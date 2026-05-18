"""
Online TD-MPC2 training on DMControl environments.

TD-MPC2 (Temporal Difference Model Predictive Control) is a model-based RL
algorithm that learns a latent world model and uses it for planning via the
Cross Entropy Method (CEM). At each step, the agent encodes the current
observation into a latent state, optimises a sequence of actions by sampling
candidates and evaluating them with the world model, then executes the first
action from the best plan.

The world model has four components learned jointly:
  - Encoder       maps observations to a SimNorm-normalised latent state
  - Dynamics      predicts the next latent state from (z, action)
  - Reward        predicts expected reward from (z, action) as a two-hot distribution
  - Q-ensemble    estimates action-value; used both for training and CEM cost

Training alternates between environment interaction (collecting transitions)
and gradient updates on batches sampled from a replay buffer. The first
SEED_STEPS steps use random actions to warm up the buffer before the policy
takes over.

Architecture choices:
  - Two-hot encoding for rewards and values follows the TD-MPC2 paper, making
    the regression scale-invariant without reward normalisation.
  - SimNorm (simplex normalisation) in the latent space replaces LayerNorm,
    providing bounded representations that are stable for planning.
  - The discount is computed automatically from the episode length using the
    paper's heuristic: γ = clip((T/5 - 1) / (T/5), 0.95, 0.995).

The offline training script (tdmpc2.py) is the single source of truth for the
loss computation. This script imports tdmpc2_forward from it so that both
training modes stay in sync automatically.

Usage:
    python tdmpc2_online.py --domain cheetah --task run
    python tdmpc2_online.py --domain cheetah               # all cheetah tasks
    python tdmpc2_online.py --list                         # show available tasks
    python tdmpc2_online.py --domain walker --steps 1000000 --seed 1
"""

import os

os.environ['MUJOCO_GL'] = 'egl'

import argparse
import contextlib
import random
import sys
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from omegaconf import OmegaConf, open_dict
from loguru import logger as logging

from stable_worldmodel.data.buffer import ReplayBuffer
from stable_worldmodel.solver.cem import CEMSolver
from stable_worldmodel.policy import WorldModelPolicy, PlanConfig
from stable_worldmodel.wm.tdmpc2 import TDMPC2, tdmpc2_forward
from stable_worldmodel.world.env_pool import EnvPool

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

TASK_REGISTRY: dict[str, dict[str, int]] = {
    'cheetah': {
        'run': 3_000_000,
        'run-backwards': 3_000_000,
        'run-front': 3_000_000,
        'run-back': 3_000_000,
        'stand-front': 3_000_000,
        'stand-back': 3_000_000,
        'lie-down': 3_000_000,
        'jump': 3_000_000,
        'legs-up': 3_000_000,
        'flip': 3_000_000,
        'flip-backward': 3_000_000,
    },
    'walker': {
        'stand': 3_000_000,
        'walk': 1_000_000,
        'run': 3_000_000,
        'walk-backward': 3_000_000,
        'lie_down': 3_000_000,
        'flip': 3_000_000,
        'arabesque': 3_000_000,
        'legs_up': 3_000_000,
    },
    'hopper': {
        'stand': 3_000_000,
        'hop': 1_500_000,
        'hop-backward': 3_000_000,
        'flip': 3_000_000,
        'flip-backward': 3_000_000,
    },
    'quadruped': {
        'walk': 3_000_000,
        'run': 3_000_000,
    },
    'reacher': {
        'easy': 3_000_000,
        'hard': 3_000_000,
    },
    'finger': {
        'spin': 3_000_000,
        'turn_easy': 3_000_000,
        'turn_hard': 3_000_000,
    },
    'humanoid': {
        'stand': 5_000_000,
        'walk': 5_000_000,
        'run': 5_000_000,
    },
    'cartpole': {'balance': 1_000_000},
    'pendulum': {'swingup': 1_000_000},
}


ENC_KEY = 'observation'

# Domains whose gymnasium wrappers have a fixed task and don't accept a task kwarg
_TASKLESS_DOMAINS = {'cartpole', 'pendulum', 'ballincup'}

SEED_STEPS = 5_000
EVAL_FREQ = 50_000
SAVE_FREQ = 50_000
EVAL_EPS = 10
BATCH_SIZE = 256
BUFFER_CAP = 1_000_000

TRAIN_NUM_SAMPLES = 256
TRAIN_N_STEPS = 4
EVAL_NUM_SAMPLES = 512
EVAL_N_STEPS = 6
CEM_TOPK = 64
CEM_VAR_SCALE = 2.0
RECEDING_HORIZON = 1
N_COLLECT_ENVS = 1
GRAD_STEPS = 1

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_CONFIG_PATH = Path(__file__).parent / 'config' / 'tdmpc2_online.yaml'
_DEVNULL = open(os.devnull, 'w')


def load_cfg(obs_dim: int, action_dim: int, discount: float) -> OmegaConf:
    """Load the online config and override env-specific fields.

    The encoding dim is preserved from the yaml; the key is replaced with
    ENC_KEY ('observation') since online DMControl envs always use that key.
    discount is computed from the episode length at runtime.
    """
    cfg = OmegaConf.load(_CONFIG_PATH)
    enc_dim = next(iter(OmegaConf.to_container(cfg.wm.encoding).values()))
    with open_dict(cfg):
        cfg.action_dim = action_dim
        cfg.extra_dims = {ENC_KEY: obs_dim}
        cfg.wm.encoding = {ENC_KEY: enc_dim}
        cfg.wm.discount = discount
    return cfg


def _get_max_episode_steps(env: gym.Env) -> int | None:
    """
    Infer the episode time limit from the env registration or, for DMControl
    envs, from the underlying dm_control time limit attribute.
    """
    if env.spec is not None and env.spec.max_episode_steps is not None:
        return env.spec.max_episode_steps
    try:
        dmc_env = env.unwrapped.dmc_env
        return int(dmc_env._step_limit / env.unwrapped.action_repeat)
    except AttributeError:
        return None


def make_env(gym_id: str, task: str | None = None) -> gym.Env:
    """
    Create a gymnasium environment with guaranteed episode termination.

    DMControlWrapper.step always returns truncated=False, so episode end
    must come from a TimeLimit wrapper. This function applies one if the
    env registration does not include max_episode_steps.
    """
    domain = gym_id.split('/')[-1].replace('DMControl-v0', '').lower()
    env_kwargs = (
        {'task': task}
        if (task is not None and domain not in _TASKLESS_DOMAINS)
        else {}
    )
    env = gym.make(gym_id, **env_kwargs)

    max_steps = _get_max_episode_steps(env)
    if max_steps is None:
        raise RuntimeError(
            f"Could not determine max_episode_steps for '{gym_id}'. "
            'Pass max_episode_steps explicitly to gym.make(), or fix the '
            'env registration to include it.'
        )
    if env.spec is None or env.spec.max_episode_steps is None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)

    f32 = spaces.Box(
        low=env.observation_space.low.astype(np.float32),
        high=env.observation_space.high.astype(np.float32),
        shape=env.observation_space.shape,
        dtype=np.float32,
    )
    return gym.wrappers.TransformObservation(
        env, lambda o: o.astype(np.float32), f32
    )


class _ForwardContext:
    """Minimal Lightning-module shim for tdmpc2_forward."""

    def __init__(self, model: TDMPC2):
        self.model = model
        self.metrics: dict = {}

    def log_dict(self, d: dict, **_):
        self.metrics.update(
            {k: v.item() if torch.is_tensor(v) else v for k, v in d.items()}
        )


def build_policy(
    model: TDMPC2,
    env: EnvPool,
    num_samples: int = TRAIN_NUM_SAMPLES,
    n_steps: int = TRAIN_N_STEPS,
) -> WorldModelPolicy:
    solver = CEMSolver(
        model=model,
        num_samples=num_samples,
        n_steps=n_steps,
        topk=CEM_TOPK,
        var_scale=CEM_VAR_SCALE,
        device=str(DEVICE),
    )
    plan_cfg = PlanConfig(
        horizon=model.cfg.wm.horizon,
        receding_horizon=RECEDING_HORIZON,
        warm_start=True,
    )
    policy = WorldModelPolicy(solver=solver, config=plan_cfg, process={})
    policy.set_env(env)
    return policy


def update_model(
    model: TDMPC2, batch: dict, cfg: OmegaConf, optimizers: dict
) -> dict:
    model.train()

    ctx = _ForwardContext(model)
    tdmpc2_forward(ctx, batch, stage='train', cfg=cfg)

    total_loss = batch['loss']
    for opt in optimizers.values():
        opt.zero_grad(set_to_none=True)
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 20.0)
    for opt in optimizers.values():
        opt.step()

    model.eval()
    return {
        'loss': ctx.metrics.get('train/loss', total_loss.item()),
        'consistency': ctx.metrics.get('train/consist', 0.0),
        'reward_loss': ctx.metrics.get('train/reward', 0.0),
        'value_loss': ctx.metrics.get('train/value', 0.0),
        'policy_loss': ctx.metrics.get('train/policy', 0.0),
    }


def save_checkpoint(model: TDMPC2, save_dir: Path, tag: str):
    torch.save(model, save_dir / f'{tag}_model.pt')
    logging.info(f'  Checkpoint saved → {tag}')


@torch.no_grad()
def evaluate(
    model: TDMPC2, gym_id: str, task: str, n_episodes: int = EVAL_EPS
) -> float:
    pool = EnvPool([lambda: make_env(gym_id, task=task)])
    policy = build_policy(
        model, pool, num_samples=EVAL_NUM_SAMPLES, n_steps=EVAL_N_STEPS
    )
    env = pool.envs[0]
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        obs = obs.astype(np.float32)
        ep_reward = 0.0
        done = False
        for buf in policy._action_buffer:
            buf.clear()
        policy._next_init = None
        while not done:
            with contextlib.redirect_stdout(_DEVNULL):
                action = policy.get_action({ENC_KEY: obs[np.newaxis]})
            action = np.asarray(action).reshape(-1)
            obs, r, term, trunc, _ = env.step(action)
            obs = obs.astype(np.float32)
            ep_reward += r
            done = term or trunc
        rewards.append(ep_reward)
    pool.close()
    return float(np.mean(rewards))


def train_task(
    domain: str,
    task: str,
    total_steps: int,
    base_dir: Path,
    use_wandb: bool = False,
    seed: int = 42,
):
    gym_id = f'swm/{domain.capitalize()}DMControl-v0'
    save_dir = base_dir / f'{domain}_{task}'
    save_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = use_wandb and WANDB_AVAILABLE

    logging.info(f'\n{"=" * 60}')
    logging.info(
        f'TD-MPC2 | {domain}-{task} | {total_steps:,} steps | device={DEVICE}'
    )
    logging.info(f'{"=" * 60}')

    pool = EnvPool(
        [lambda: make_env(gym_id, task=task) for _ in range(N_COLLECT_ENVS)]
    )
    obs_arr = np.stack([e.reset()[0].astype(np.float32) for e in pool.envs])

    obs_dim = pool.envs[0].observation_space.shape[0]
    action_dim = pool.envs[0].action_space.shape[0]

    max_ep_steps = _get_max_episode_steps(pool.envs[0])
    discount = float(
        np.clip((max_ep_steps / 5 - 1) / (max_ep_steps / 5), 0.95, 0.995)
    )

    cfg = load_cfg(obs_dim=obs_dim, action_dim=action_dim, discount=discount)
    horizon = cfg.wm.horizon

    logging.info(
        f'  obs_dim={obs_dim} | action_dim={action_dim} | '
        f'horizon={horizon} | discount={discount:.4f} | '
        f'max_ep_steps={max_ep_steps}'
    )

    if use_wandb:
        wandb.init(
            project='tdmpc2-online',
            name=f'{domain}_{task}_seed{seed}',
            config={
                'domain': domain,
                'task': task,
                'total_steps': total_steps,
                'obs_dim': obs_dim,
                'action_dim': action_dim,
                'discount': discount,
                'seed': seed,
                **{f'wm/{k}': v for k, v in dict(cfg.wm).items()},
            },
            sync_tensorboard=False,
        )

    model = TDMPC2(cfg).to(DEVICE)
    model.eval()

    lr = cfg.optimizer.lr
    optimizers = {
        'enc': torch.optim.Adam(
            list(model.extra_encoders.parameters())
            + list(model.sim_norm.parameters()),
            lr=lr * cfg.enc_lr_scale,
        ),
        'wm': torch.optim.Adam(
            list(model.dynamics.parameters())
            + list(model.reward.parameters())
            + list(model.qs.parameters()),
            lr=lr,
        ),
        'pi': torch.optim.Adam(model.pi.parameters(), lr=lr, eps=1e-5),
    }

    buffer = ReplayBuffer(max_steps=BUFFER_CAP, history_len=horizon + 1)
    policy = build_policy(model, pool)

    best_eval = -float('inf')
    ep_reward = np.zeros(N_COLLECT_ENVS)
    ep_steps = np.zeros(N_COLLECT_ENVS, dtype=int)
    cur_obs = [[] for _ in range(N_COLLECT_ENVS)]
    cur_act = [[] for _ in range(N_COLLECT_ENVS)]
    cur_rew = [[] for _ in range(N_COLLECT_ENVS)]

    for step in range(N_COLLECT_ENVS, total_steps + 1, N_COLLECT_ENVS):
        if step <= SEED_STEPS:
            actions = np.stack(
                [e.action_space.sample().astype(np.float32) for e in pool.envs]
            )
        else:
            with contextlib.redirect_stdout(_DEVNULL):
                actions = np.asarray(
                    policy.get_action({ENC_KEY: obs_arr})
                ).reshape(N_COLLECT_ENVS, -1)

        for i in range(N_COLLECT_ENVS):
            next_obs, reward, terminated, truncated, _ = pool.envs[i].step(
                actions[i]
            )
            next_obs = next_obs.astype(np.float32)
            done = terminated or truncated
            cur_obs[i].append(obs_arr[i].copy())
            cur_act[i].append(actions[i].copy())
            cur_rew[i].append(float(reward))
            ep_reward[i] += reward
            ep_steps[i] += 1
            obs_arr[i] = next_obs
            if done:
                buffer.write_episode(
                    {
                        ENC_KEY: np.stack(cur_obs[i]).astype(np.float32),
                        'action': np.stack(cur_act[i]).astype(np.float32),
                        'reward': np.array(cur_rew[i], np.float32),
                    }
                )
                cur_obs[i].clear()
                cur_act[i].clear()
                cur_rew[i].clear()
                logging.info(
                    f'[{domain}-{task}] step={step:,} | ep_reward={ep_reward[i]:.2f} | ep_len={ep_steps[i]}'
                )
                if use_wandb:
                    wandb.log(
                        {
                            'rollout/ep_reward': ep_reward[i],
                            'rollout/ep_len': ep_steps[i],
                        },
                        step=step,
                    )
                obs_arr[i], _ = pool.envs[i].reset()
                obs_arr[i] = obs_arr[i].astype(np.float32)
                ep_reward[i], ep_steps[i] = 0.0, 0
                if step > SEED_STEPS:
                    policy._action_buffer[i].clear()
                    if policy._next_init is not None:
                        policy._next_init[i] = 0

        if step >= SEED_STEPS and len(buffer) >= BATCH_SIZE:
            for _ in range(GRAD_STEPS):
                raw = buffer.sample(BATCH_SIZE)
                batch = {
                    k: torch.as_tensor(v)
                    .pin_memory()
                    .to(DEVICE, non_blocking=True)
                    for k, v in raw.items()
                }
                metrics = update_model(model, batch, cfg, optimizers)
            if step % 5_000 == 0:
                logging.info(
                    f'[{domain}-{task}] step={step:,} | '
                    f'loss={metrics["loss"]:.4f} | '
                    f'rew={metrics["reward_loss"]:.4f} | '
                    f'val={metrics["value_loss"]:.4f} | '
                    f'pi={metrics["policy_loss"]:.4f}'
                )
                if use_wandb:
                    wandb.log(
                        {
                            'train/loss': metrics['loss'],
                            'train/reward_loss': metrics['reward_loss'],
                            'train/value_loss': metrics['value_loss'],
                            'train/policy_loss': metrics['policy_loss'],
                        },
                        step=step,
                    )

        if step % SAVE_FREQ == 0 and step >= SEED_STEPS:
            save_checkpoint(model, save_dir, tag=f'step_{step}')

        if step % EVAL_FREQ == 0 and step >= SEED_STEPS:
            eval_r = evaluate(model, gym_id, task)
            logging.info(
                f'[{domain}-{task}] *** EVAL step={step:,} | mean_reward={eval_r:.2f} ***'
            )
            if use_wandb:
                wandb.log({'eval/mean_reward': eval_r}, step=step)
            if eval_r > best_eval:
                best_eval = eval_r
                save_checkpoint(model, save_dir, tag='best')
                logging.info(f'[{domain}-{task}] New best: {best_eval:.2f}')

    save_checkpoint(model, save_dir, tag='final')
    logging.info(f'[{domain}-{task}] Done. Best eval reward: {best_eval:.2f}')
    pool.close()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if use_wandb:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(
        description='Online TD-MPC2 training on DMControl environments'
    )
    parser.add_argument(
        '--domain', type=str, help='Domain name (e.g. cheetah, walker, hopper)'
    )
    parser.add_argument(
        '--task',
        type=str,
        help='Task name. If omitted, runs all tasks in the domain.',
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=None,
        help='Override total training steps.',
    )
    parser.add_argument(
        '--base_dir',
        type=str,
        default='./models/tdmpc2',
        help='Output directory for checkpoints.',
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available domain/task combinations and exit.',
    )
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument(
        '--wandb',
        action='store_true',
        help='Log metrics to Weights & Biases (requires: pip install wandb)',
    )
    args = parser.parse_args()

    if args.list:
        for domain, tasks in TASK_REGISTRY.items():
            logging.info(f'[{domain}]: {", ".join(tasks.keys())}')
        return

    if not args.domain:
        logging.error(
            'Please specify --domain (or use --list to see options).'
        )
        sys.exit(1)

    domain = args.domain.lower()
    if domain not in TASK_REGISTRY:
        logging.error(
            f"Domain '{domain}' not found. Use --list to see options."
        )
        sys.exit(1)

    if args.task and args.task not in TASK_REGISTRY[domain]:
        logging.error(
            f"Task '{args.task}' not found in domain '{domain}'. "
            f'Available: {", ".join(TASK_REGISTRY[domain].keys())}'
        )
        sys.exit(1)

    tasks = (
        {args.task: TASK_REGISTRY[domain][args.task]}
        if args.task
        else TASK_REGISTRY[domain]
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    base_dir = Path(args.base_dir)
    for task, total_steps in tasks.items():
        if args.steps is not None:
            total_steps = args.steps
        train_task(
            domain=domain,
            task=task,
            total_steps=total_steps,
            base_dir=base_dir,
            use_wandb=args.wandb,
            seed=args.seed,
        )


if __name__ == '__main__':
    main()
