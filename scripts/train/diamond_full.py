"""Full DIAMOND training: diffusion world model + RL agent in imagination.

Implements Algorithm 1 from "Diffusion for World Modeling: Visual Details
Matter in Atari" (Alonso et al., 2024).

Features:
  - Atari 100k protocol wrappers (NoopReset, EpisodicLife, ClipReward)
  - CategoricalCEMSolver for MPC evaluation (random-shooting / CEM)
  - ReplayBuffer from repo infrastructure (ring buffer, episode-aware)
  - Hydra/OmegaConf config system
  - Gradient clipping for stable training

Usage:
    python scripts/train/diamond_full.py env_name=ALE/Breakout-v5 epochs=1000
"""

import time
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch import optim

from stable_worldmodel.data.buffer import ReplayBuffer, classic_filter
from stable_worldmodel.policy import PlanConfig
from stable_worldmodel.solver import CategoricalCEMSolver
from stable_worldmodel.wm.diamond.agent import ActorCritic, RewardTermModel
from stable_worldmodel.wm.diamond.cost_model import DiamondCostModel
from stable_worldmodel.wm.diamond.diamond_agent import (
    DiamondAgent,
    compute_rl_losses,
)
from stable_worldmodel.wm.diamond.diffusion import DiscreteActionEncoder
from stable_worldmodel.wm.diamond.edm import EDMModel
from stable_worldmodel.wm.diamond.edm_sampling import sample_euler
from stable_worldmodel.envs.ale import make_atari_env


def preprocess_obs(obs):
    obs = obs[34:194, :, :]
    obs = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    obs = nn.functional.interpolate(obs, size=(64, 64), mode='area')
    obs = obs / 255.0
    return obs


def clip_grad_norm(model, max_norm=1.0):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def build_imagination_batch_from_clip(clip, L, device):
    pixels = torch.as_tensor(clip['pixels'], device=device).float()
    actions = torch.as_tensor(clip['action'], device=device).float()
    B = pixels.shape[0]
    pixels = pixels[:, :L]  # (B, L, C, H, W)
    actions = actions[:, :L].squeeze(-1)  # (B, L)
    return {'init_obs': pixels, 'init_actions': actions}


def build_diffusion_batch_from_clip(clip, L, action_encoder, device):
    pixels = torch.as_tensor(clip['pixels'], device=device).float()
    actions = torch.as_tensor(clip['action'], device=device).float()
    B = pixels.shape[0]

    next_frame = pixels[:, L]  # (B, C, H, W)
    history = pixels[:, :L].reshape(B, 3 * L, 64, 64)  # (B, C*L, H, W)

    act_seq = actions[:, :L].squeeze(-1)  # (B, L)
    act_emb = action_encoder(act_seq.unsqueeze(-1))
    cond_vec = act_emb.mean(dim=1)  # (B, cond_dim)

    batch = {
        'next_frame': next_frame,
        'history': history,
        'cond_vec': cond_vec,
    }
    return batch


def build_rterm_batch_from_clip(clip, L, H, device):
    seq_len = L + H
    pixels = torch.as_tensor(clip['pixels'], device=device).float()
    actions = torch.as_tensor(clip['action'], device=device).float()

    B = pixels.shape[0]
    frames = pixels[:, :seq_len]  # (B, seq_len, C, H, W)
    actions = actions[:, :seq_len]  # (B, seq_len)
    batch = {'frames': frames, 'actions': actions}

    if 'reward' in clip:
        r = torch.as_tensor(clip['reward'], device=device).float()[:, :seq_len]
        batch['rewards'] = r
    if 'terminated' in clip:
        t = torch.as_tensor(clip['terminated'], device=device).float()[
            :, :seq_len
        ]
        batch['terminals'] = t
    return batch


@torch.no_grad()
def evaluate(
    agent,
    env,
    device,
    solver=None,
    plan_cfg=None,
    cost_model=None,
    n_episodes=5,
    max_steps=27000,
):
    agent.eval()
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        obs_hist = []
        act_hist = []
        hx_ac = torch.zeros(
            1, agent.actor_critic.lstm.hidden_size, device=device
        )
        cx_ac = torch.zeros(
            1, agent.actor_critic.lstm.hidden_size, device=device
        )
        ep_return = 0

        for _ in range(max_steps):
            frame = preprocess_obs(obs).to(device)
            obs_hist.append(frame)
            if len(obs_hist) > agent.history_size:
                obs_hist.pop(0)

            if len(obs_hist) < agent.history_size:
                action = env.action_space.sample()
                _, _, (hx_ac, cx_ac) = agent.actor_critic(
                    frame, (hx_ac, cx_ac)
                )
            else:
                if solver is not None and cost_model is not None:
                    obs_tensor = torch.stack(obs_hist, dim=0).unsqueeze(0)
                    act_tensor = torch.tensor(
                        act_hist[-agent.history_size :], device=device
                    ).unsqueeze(0)
                    hx_rw, cx_rw = cost_model.warmup(obs_tensor, act_tensor)
                    info_dict = {
                        'pixels': obs_tensor,
                        'hx_rw': hx_rw,
                        'cx_rw': cx_rw,
                    }
                    result = solver(info_dict)
                    action = result['actions'][0, 0].item()
                else:
                    if torch.rand(1).item() < 0.01:
                        action = env.action_space.sample()
                    else:
                        logits, _, _ = agent.actor_critic(frame)
                        action = logits.argmax(dim=-1).item()

            act_hist.append(action)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            obs = next_obs
            if terminated or truncated:
                break

        returns.append(ep_return)

    agent.train()
    return np.mean(returns), np.std(returns)


def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


@hydra.main(
    version_base=None, config_path='./config', config_name='diamond_full'
)
def run(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Config:\n{OmegaConf.to_yaml(cfg)}')
    print(f'Using device: {device}')

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    L = cfg.wm.history_size
    H = cfg.wm.imagination_horizon
    action_dim = cfg.action_dim
    cond_dim = cfg.model.cond_dim

    env = make_atari_env(
        cfg.env_name,
        seed=cfg.seed,
        max_episode_steps=cfg.max_episode_steps,
    )
    print(f'Environment: {cfg.env_name}')
    print(f'Action space: {env.action_space}')

    diffusion_model = EDMModel(
        in_ch=cfg.model.in_ch,
        base_ch=cfg.model.base_ch,
        cond_dim=cond_dim,
        history_frames=L,
    ).to(device)

    reward_term_model = RewardTermModel(
        in_channels=cfg.reward_term.in_channels,
        action_dim=cfg.reward_term.action_dim,
        cond_dim=cfg.reward_term.cond_dim,
        lstm_dim=cfg.reward_term.lstm_dim,
    ).to(device)

    actor_critic = ActorCritic(
        in_channels=cfg.actor_critic.in_channels,
        action_dim=cfg.actor_critic.action_dim,
        lstm_dim=cfg.actor_critic.lstm_dim,
    ).to(device)

    action_encoder = DiscreteActionEncoder(
        num_actions=cfg.action_encoder.num_actions,
        emb_dim=cfg.action_encoder.emb_dim,
    ).to(device)

    agent = DiamondAgent(
        diffusion_model=diffusion_model,
        reward_term_model=reward_term_model,
        actor_critic=actor_critic,
        action_encoder=action_encoder,
        history_size=L,
        device=device,
    )

    cost_model = DiamondCostModel(
        diffusion=diffusion_model,
        reward_term=reward_term_model,
        action_encoder=action_encoder,
        history_size=L,
        n_denoise=cfg.wm.n_denoise_train,
        device=device,
    )

    solver = CategoricalCEMSolver(
        model=cost_model,
        num_samples=cfg.solver.num_samples,
        n_steps=cfg.solver.n_steps,
        topk=cfg.solver.topk,
        smoothing=cfg.solver.smoothing,
        alpha=cfg.solver.alpha,
        device=str(device),
        seed=cfg.seed,
    )
    solver.configure(
        action_space=env.action_space,
        n_envs=1,
        config=PlanConfig(**cfg.plan),
    )
    print(
        f'Solver: {type(solver).__name__} '
        f'(num_samples={cfg.solver.num_samples}, '
        f'n_steps={cfg.solver.n_steps})'
    )

    buffer = ReplayBuffer(
        max_steps=cfg.buffer.max_steps,
        history_len=1,
        frameskip=cfg.buffer.frameskip,
        key_filter=classic_filter,
    )

    opt_diff = optim.AdamW(
        list(diffusion_model.parameters()) + list(action_encoder.parameters()),
        lr=cfg.lr,
        weight_decay=1e-2,
    )
    opt_rterm = optim.AdamW(
        reward_term_model.parameters(), lr=cfg.lr, weight_decay=1e-2
    )
    opt_ac = optim.AdamW(actor_critic.parameters(), lr=cfg.lr, weight_decay=0)

    # ── Initial random data ──────────────────────────────────────────────
    print('Collecting initial random data...')
    n_init_episodes = 20
    collected = 0
    for _ in range(n_init_episodes):
        ep = {
            'pixels': [],
            'action': [],
            'reward': [],
            'terminated': [],
            'truncated': [],
        }
        obs, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            reward = max(-1, min(1, reward))
            done = terminated or truncated
            frame = _to_np(preprocess_obs(obs).squeeze(0))
            ep['pixels'].append(frame)
            ep['action'].append(np.array([action], dtype=np.float32))
            ep['reward'].append(np.array([reward], dtype=np.float32))
            ep['terminated'].append(np.array([terminated], dtype=bool))
            ep['truncated'].append(np.array([truncated], dtype=bool))
            obs = next_obs
            collected += 1
        buffer.write_episode(ep)
    print(
        f'Initial buffer: {buffer.num_episodes} episodes, '
        f'{buffer.num_steps_stored} steps'
    )

    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    total_steps = 0
    best_return = -float('inf')

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.time()

        # ── Collect experience ──────────────────────────────────────────
        diffusion_model.eval()
        actor_critic.eval()
        ep = {
            'pixels': [],
            'action': [],
            'reward': [],
            'terminated': [],
            'truncated': [],
        }
        obs, _ = env.reset()
        for _ in range(cfg.steps_per_epoch):
            frame = preprocess_obs(obs).to(device)
            if torch.rand(1).item() < cfg.epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    logits, _, _ = actor_critic(frame)
                    action = logits.argmax(dim=-1).item()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            reward = max(-1, min(1, reward))
            done = terminated or truncated

            frame_np = _to_np(preprocess_obs(obs).squeeze(0))
            ep['pixels'].append(frame_np)
            ep['action'].append(np.array([action], dtype=np.float32))
            ep['reward'].append(np.array([reward], dtype=np.float32))
            ep['terminated'].append(np.array([terminated], dtype=bool))
            ep['truncated'].append(np.array([truncated], dtype=bool))

            total_steps += 1
            obs = next_obs
            if done:
                buffer.write_episode(ep)
                ep = {
                    'pixels': [],
                    'action': [],
                    'reward': [],
                    'terminated': [],
                    'truncated': [],
                }
                obs, _ = env.reset()

        if len(ep['pixels']) > 0:
            buffer.write_episode(ep)

        # ── Train diffusion model ──────────────────────────────────────
        diffusion_model.train()
        diff_loss = 0.0
        for _ in range(cfg.train_steps):
            clip = buffer.sample(cfg.batch_size, history_len=L + 1)
            batch = build_diffusion_batch_from_clip(
                clip, L, action_encoder, device
            )
            opt_diff.zero_grad()
            loss = agent.train_diffusion_step(batch)
            loss.backward()
            clip_grad_norm(diffusion_model, cfg.grad_clip)
            opt_diff.step()
            diff_loss += loss.item()
        diff_loss = diff_loss / max(1, cfg.train_steps)

        # ── Train reward/termination model ─────────────────────────────
        reward_term_model.train()
        rterm_loss = 0.0
        for _ in range(cfg.train_steps):
            clip = buffer.sample(cfg.batch_size, history_len=L + H)
            batch = build_rterm_batch_from_clip(clip, L, H, device)
            opt_rterm.zero_grad()
            loss = agent.train_reward_term_step(batch)
            loss.backward()
            clip_grad_norm(reward_term_model, cfg.grad_clip)
            opt_rterm.step()
            rterm_loss += loss.item()
        rterm_loss = rterm_loss / max(1, cfg.train_steps)

        # ── Train actor-critic in imagination ──────────────────────────
        actor_critic.train()
        total_rl_loss = 0.0
        last_traj = None
        for _ in range(cfg.train_steps):
            clip = buffer.sample(cfg.batch_size, history_len=L)
            ibatch = build_imagination_batch_from_clip(clip, L, device)
            trajectory = agent.imagination_rollout(
                ibatch['init_obs'],
                ibatch['init_actions'],
                H=H,
                n_denoise=cfg.wm.n_denoise_train,
            )
            last_traj = trajectory
            opt_ac.zero_grad()
            losses = compute_rl_losses(
                trajectory,
                gamma=cfg.gamma,
                lam=cfg.lam,
                eta=cfg.entropy_weight,
            )
            losses['rl_loss'].backward()
            clip_grad_norm(actor_critic, cfg.grad_clip)
            opt_ac.step()
            total_rl_loss += losses['rl_loss'].item()
        rl_loss_avg = total_rl_loss / max(1, cfg.train_steps)

        epoch_time = time.time() - epoch_start

        if epoch % cfg.log_every == 0 or epoch == 1:
            avg_return = (
                last_traj['rewards'].mean().item()
                if last_traj is not None
                else 0
            )
            avg_value = (
                last_traj['values'].mean().item()
                if last_traj is not None
                else 0
            )
            entropy_val = (
                last_traj['entropies'].mean().item()
                if last_traj is not None
                else 0
            )
            print(
                f'Epoch {epoch:4d}/{cfg.epochs} | '
                f'{epoch_time:.1f}s | '
                f'Steps:{total_steps:5d} | '
                f'Buf:{buffer.num_steps_stored:5d} | '
                f'D:{diff_loss:.4f} | '
                f'R:{rterm_loss:.4f} | '
                f'RL:{rl_loss_avg:.4f} | '
                f'Ret:{avg_return:.3f} | '
                f'Val:{avg_value:.3f} | '
                f'Ent:{entropy_val:.4f}'
            )

        if cfg.eval_every and epoch % cfg.eval_every == 0:
            eval_return, eval_std = evaluate(
                agent,
                env,
                device,
                solver=solver if cfg.solver.n_steps > 0 else None,
                plan_cfg=PlanConfig(**cfg.plan),
                cost_model=cost_model if cfg.solver.n_steps > 0 else None,
                n_episodes=5,
            )
            mode = 'MPC' if cfg.solver.n_steps > 0 else 'policy'
            print(f'  Eval ({mode}): {eval_return:.2f} +/- {eval_std:.2f}')
            if eval_return > best_return:
                best_return = eval_return
                torch.save(agent.state_dict(), checkpoint_dir / 'best.pt')

        if epoch % 100 == 0:
            torch.save(
                {
                    'epoch': epoch,
                    'diffusion': diffusion_model.state_dict(),
                    'reward_term': reward_term_model.state_dict(),
                    'actor_critic': actor_critic.state_dict(),
                    'action_encoder': action_encoder.state_dict(),
                    'opt_diff': opt_diff.state_dict(),
                    'opt_rterm': opt_rterm.state_dict(),
                    'opt_ac': opt_ac.state_dict(),
                },
                checkpoint_dir / f'checkpoint_epoch_{epoch}.pt',
            )

    print('Training complete!')
    print(f'Best eval return: {best_return:.2f}')


if __name__ == '__main__':
    run()
