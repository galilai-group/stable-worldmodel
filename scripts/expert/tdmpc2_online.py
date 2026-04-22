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
os.environ["MUJOCO_GL"] = "egl"

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from omegaconf import OmegaConf, open_dict
from loguru import logger as logging

import stable_worldmodel as swm
from stable_worldmodel.solver.cem import CEMSolver
from stable_worldmodel.policy import WorldModelPolicy, PlanConfig
from stable_worldmodel.wm.tdmpc2 import TDMPC2, tdmpc2_forward

TASK_REGISTRY: dict[str, dict[str, int]] = {
    "cheetah": {
        "run":           3_000_000,
        "run-backwards": 3_000_000,
        "run-front":     3_000_000,
        "run-back":      3_000_000,
        "stand-front":   3_000_000,
        "stand-back":    3_000_000,
        "lie-down":      3_000_000,
        "jump":          3_000_000,
        "legs-up":       3_000_000,
        "flip":          3_000_000,
        "flip-backward": 3_000_000,
    },
    "walker": {
        "stand":         3_000_000,
        "walk":          3_000_000,
        "run":           3_000_000,
        "walk-backward": 3_000_000,
        "lie_down":      3_000_000,
        "flip":          3_000_000,
        "arabesque":     3_000_000,
        "legs_up":       3_000_000,
    },
    "hopper": {
        "stand":         3_000_000,
        "hop":           3_000_000,
        "hop-backward":  3_000_000,
        "flip":          3_000_000,
        "flip-backward": 3_000_000,
    },
    "quadruped": {
        "walk": 3_000_000,
        "run":  3_000_000,
    },
    "reacher": {
        "easy": 3_000_000,
        "hard": 3_000_000,
    },
    "finger": {
        "spin":      3_000_000,
        "turn_easy": 3_000_000,
        "turn_hard": 3_000_000,
    },
    "humanoid": {
        "stand": 5_000_000,
        "walk":  5_000_000,
        "run":   5_000_000,
    },
}


ENC_KEY = "observation"

SEED_STEPS = 5_000
EVAL_FREQ  = 50_000
SAVE_FREQ  = 50_000
EVAL_EPS   = 10
BATCH_SIZE = 256
BUFFER_CAP = 1_000_000

TRAIN_NUM_SAMPLES = 256
TRAIN_N_STEPS     = 4
EVAL_NUM_SAMPLES  = 512
EVAL_N_STEPS      = 6
CEM_TOPK          = 64
CEM_VAR_SCALE     = 2.0
RECEDING_HORIZON  = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_CONFIG_PATH = Path(__file__).parent / "config" / "tdmpc2_online.yaml"


def load_cfg(obs_dim: int, action_dim: int, discount: float) -> OmegaConf:
    """Load the online config and override env-specific fields.

    The encoding dim is preserved from the yaml; the key is replaced with
    ENC_KEY ('observation') since online DMControl envs always use that key.
    discount is computed from the episode length at runtime.
    """
    cfg = OmegaConf.load(_CONFIG_PATH)
    enc_dim = next(iter(OmegaConf.to_container(cfg.wm.encoding).values()))
    with open_dict(cfg):
        cfg.action_dim    = action_dim
        cfg.extra_dims    = {ENC_KEY: obs_dim}
        cfg.wm.encoding   = {ENC_KEY: enc_dim}
        cfg.wm.discount   = discount
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

    Note: the correct library fix is DMControlWrapper.step returning
    truncated=step.last() — this is a workaround until that lands.
    """
    env_kwargs = {"task": task} if task is not None else {}
    env = gym.make(gym_id, **env_kwargs)

    max_steps = _get_max_episode_steps(env)
    if max_steps is None:
        raise RuntimeError(
            f"Could not determine max_episode_steps for '{gym_id}'. "
            "Pass max_episode_steps explicitly to gym.make(), or fix the "
            "env registration to include it."
        )
    if env.spec is None or env.spec.max_episode_steps is None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)

    f32 = spaces.Box(
        low=env.observation_space.low.astype(np.float32),
        high=env.observation_space.high.astype(np.float32),
        shape=env.observation_space.shape, dtype=np.float32,
    )
    return gym.wrappers.TransformObservation(env, lambda o: o.astype(np.float32), f32)

# Needed to handle the action space shape expected by CEMSolver without modifying cem.py
class _EnvProxy:
    num_envs = 1
    def __init__(self, action_dim: int):
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1, action_dim), dtype=np.float32
        )


class _ForwardContext:
    """
    Satisfies the self.model / self.log_dict interface that tdmpc2_forward
    expects from a LightningModule, without requiring Lightning.
    """
    def __init__(self, model: TDMPC2):
        self.model   = model
        self.metrics: dict = {}

    def log_dict(self, d: dict, **kwargs):
        self.metrics.update({
            k: v.item() if torch.is_tensor(v) else v for k, v in d.items()
        })


def build_policy(model: TDMPC2,
                 num_samples: int = TRAIN_NUM_SAMPLES,
                 n_steps: int = TRAIN_N_STEPS) -> WorldModelPolicy:
    solver   = CEMSolver(model=model, num_samples=num_samples, n_steps=n_steps,
                         topk=CEM_TOPK, var_scale=CEM_VAR_SCALE, device=str(DEVICE))
    plan_cfg = PlanConfig(horizon=model.cfg.wm.horizon,
                          receding_horizon=RECEDING_HORIZON, warm_start=True)
    policy   = WorldModelPolicy(solver=solver, config=plan_cfg, process={})
    policy.set_env(_EnvProxy(model.cfg.action_dim))
    return policy


class SequenceReplayBuffer:
    """Stores full episodes; samples contiguous windows of length (horizon + 1)."""

    def __init__(self, capacity: int, horizon: int, obs_dim: int,
                 action_dim: int, device: torch.device):
        self.capacity   = capacity
        self.horizon    = horizon
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.device     = device
        self._episodes: list[dict] = []
        self._cur_obs:  list[np.ndarray] = []
        self._cur_act:  list[np.ndarray] = []
        self._cur_rew:  list[float]      = []
        self._total     = 0

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float):
        self._cur_obs.append(obs.copy())
        self._cur_act.append(action.copy())
        self._cur_rew.append(float(reward))

    def end_episode(self):
        if len(self._cur_obs) > self.horizon:
            ep = {"obs": np.stack(self._cur_obs).astype(np.float32),
                  "act": np.stack(self._cur_act).astype(np.float32),
                  "rew": np.array(self._cur_rew, np.float32)}
            self._episodes.append(ep)
            self._total += len(self._cur_obs)
            while self._total > self.capacity and len(self._episodes) > 1:
                self._total -= len(self._episodes.pop(0)["obs"])
        self._cur_obs.clear(); self._cur_act.clear(); self._cur_rew.clear()

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        seq_len = self.horizon + 1
        obs_b = np.zeros((batch_size, seq_len, self.obs_dim),    np.float32)
        act_b = np.zeros((batch_size, seq_len, self.action_dim), np.float32)
        rew_b = np.zeros((batch_size, seq_len),                  np.float32)
        for i in range(batch_size):
            ep    = random.choice(self._episodes)
            T     = len(ep["obs"])
            start = random.randint(0, max(0, T - seq_len))
            end   = min(start + seq_len, T)
            n     = end - start
            obs_b[i, :n] = ep["obs"][start:end]
            act_b[i, :n] = ep["act"][start:end]
            rew_b[i, :n] = ep["rew"][start:end]
        return {ENC_KEY:  torch.as_tensor(obs_b, device=self.device),
                "action": torch.as_tensor(act_b, device=self.device),
                "reward": torch.as_tensor(rew_b, device=self.device)}

    def __len__(self) -> int:
        return self._total

def update_model(model: TDMPC2, batch: dict, cfg: OmegaConf,
                 optimizers: dict) -> dict:
    model.train()

    ctx = _ForwardContext(model)
    tdmpc2_forward(ctx, batch, stage="train", cfg=cfg)

    total_loss = batch["loss"]
    for opt in optimizers.values():
        opt.zero_grad(set_to_none=True)
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 20.0)
    for opt in optimizers.values():
        opt.step()

    model.eval()
    return {"loss":        ctx.metrics.get("train/loss",    total_loss.item()),
            "consistency": ctx.metrics.get("train/consist", 0.0),
            "reward_loss": ctx.metrics.get("train/reward",  0.0),
            "value_loss":  ctx.metrics.get("train/value",   0.0),
            "policy_loss": ctx.metrics.get("train/policy",  0.0)}

def save_checkpoint(model: TDMPC2, save_dir: Path, tag: str):
    torch.save(model, save_dir / f"{tag}_model.pt")
    logging.info(f"  Checkpoint saved → {tag}")


@torch.no_grad()
def evaluate(model: TDMPC2, gym_id: str, task: str,
             n_episodes: int = EVAL_EPS) -> float:
    env    = make_env(gym_id, task=task)
    policy = build_policy(model, num_samples=EVAL_NUM_SAMPLES, n_steps=EVAL_N_STEPS)
    rewards = []
    for _ in range(n_episodes):
        obs, _    = env.reset()
        obs       = obs.astype(np.float32)
        ep_reward = 0.0
        done      = False
        policy._action_buffer.clear()
        policy._next_init = None
        while not done:
            action = policy.get_action({ENC_KEY: obs[np.newaxis]})
            action = np.asarray(action).reshape(-1)
            obs, r, term, trunc, _ = env.step(action)
            obs        = obs.astype(np.float32)
            ep_reward += r
            done       = term or trunc
        rewards.append(ep_reward)
    env.close()
    return float(np.mean(rewards))


def train_task(domain: str, task: str, total_steps: int, base_dir: Path):
    gym_id   = f"swm/{domain.capitalize()}DMControl-v0"
    save_dir = base_dir / f"{domain}_{task}"
    save_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"\n{'='*60}")
    logging.info(f"TD-MPC2 | {domain}-{task} | {total_steps:,} steps | device={DEVICE}")
    logging.info(f"{'='*60}")

    env    = make_env(gym_id, task=task)
    obs, _ = env.reset()
    obs    = obs.astype(np.float32)

    obs_dim    = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    max_ep_steps = _get_max_episode_steps(env)
    discount     = float(np.clip(
        (max_ep_steps / 5 - 1) / (max_ep_steps / 5), 0.95, 0.995
    ))

    cfg    = load_cfg(obs_dim=obs_dim, action_dim=action_dim, discount=discount)
    horizon = cfg.wm.horizon

    logging.info(f"  obs_dim={obs_dim} | action_dim={action_dim} | "
                 f"horizon={horizon} | discount={discount:.4f} | "
                 f"max_ep_steps={max_ep_steps}")

    model = TDMPC2(cfg).to(DEVICE)
    model.eval()

    lr = cfg.optimizer.lr
    optimizers = {
        "enc": torch.optim.Adam(
            list(model.extra_encoders.parameters()) + list(model.sim_norm.parameters()),
            lr=lr * cfg.enc_lr_scale),
        "wm":  torch.optim.Adam(
            list(model.dynamics.parameters())
            + list(model.reward.parameters())
            + list(model.qs.parameters()),
            lr=lr),
        "pi":  torch.optim.Adam(model.pi.parameters(), lr=lr, eps=1e-5),
    }

    buffer = SequenceReplayBuffer(BUFFER_CAP, horizon, obs_dim, action_dim, DEVICE)
    policy = build_policy(model)

    ep_reward, ep_steps, best_eval = 0.0, 0, -float("inf")

    for step in range(1, total_steps + 1):

        if step <= SEED_STEPS:
            action = env.action_space.sample().astype(np.float32)
        else:
            action = policy.get_action({ENC_KEY: obs[np.newaxis]})
            action = np.asarray(action).reshape(-1)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_obs = next_obs.astype(np.float32)
        done     = terminated or truncated

        buffer.add(obs, action, reward)
        ep_reward += reward
        ep_steps  += 1
        obs        = next_obs

        if done:
            buffer.end_episode()
            logging.info(
                f"[{domain}-{task}] step={step:,} | "
                f"ep_reward={ep_reward:.2f} | ep_len={ep_steps}"
            )
            obs, _    = env.reset()
            obs       = obs.astype(np.float32)
            ep_reward, ep_steps = 0.0, 0
            if step > SEED_STEPS:
                policy._action_buffer.clear()
                policy._next_init = None

        if step >= SEED_STEPS and len(buffer) >= BATCH_SIZE * (horizon + 1):
            metrics = update_model(model, buffer.sample(BATCH_SIZE), cfg, optimizers)
            if step % 5_000 == 0:
                logging.info(
                    f"[{domain}-{task}] step={step:,} | "
                    f"loss={metrics['loss']:.4f} | "
                    f"rew={metrics['reward_loss']:.4f} | "
                    f"val={metrics['value_loss']:.4f} | "
                    f"pi={metrics['policy_loss']:.4f}"
                )

        if step % SAVE_FREQ == 0 and step >= SEED_STEPS:
            save_checkpoint(model, save_dir, tag=f"step_{step}")

        if step % EVAL_FREQ == 0 and step >= SEED_STEPS:
            eval_r = evaluate(model, gym_id, task)
            logging.info(
                f"[{domain}-{task}] *** EVAL step={step:,} | mean_reward={eval_r:.2f} ***"
            )
            if eval_r > best_eval:
                best_eval = eval_r
                save_checkpoint(model, save_dir, tag="best")
                logging.info(f"[{domain}-{task}] New best: {best_eval:.2f}")

    save_checkpoint(model, save_dir, tag="final")
    logging.info(f"[{domain}-{task}] Done. Best eval reward: {best_eval:.2f}")
    env.close()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(
        description="Online TD-MPC2 training on DMControl environments"
    )
    parser.add_argument("--domain", type=str,
                        help="Domain name (e.g. cheetah, walker, hopper)")
    parser.add_argument("--task",   type=str,
                        help="Task name. If omitted, runs all tasks in the domain.")
    parser.add_argument("--steps",  type=int, default=None,
                        help="Override total training steps.")
    parser.add_argument("--base_dir", type=str, default="./models/tdmpc2",
                        help="Output directory for checkpoints.")
    parser.add_argument("--list", action="store_true",
                        help="List all available domain/task combinations and exit.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    args = parser.parse_args()

    if args.list:
        for domain, tasks in TASK_REGISTRY.items():
            logging.info(f"[{domain}]: {', '.join(tasks.keys())}")
        return

    if not args.domain:
        logging.error("Please specify --domain (or use --list to see options).")
        sys.exit(1)

    domain = args.domain.lower()
    if domain not in TASK_REGISTRY:
        logging.error(f"Domain '{domain}' not found. Use --list to see options.")
        sys.exit(1)

    if args.task and args.task not in TASK_REGISTRY[domain]:
        logging.error(
            f"Task '{args.task}' not found in domain '{domain}'. "
            f"Available: {', '.join(TASK_REGISTRY[domain].keys())}"
        )
        sys.exit(1)

    tasks = {args.task: TASK_REGISTRY[domain][args.task]} if args.task \
            else TASK_REGISTRY[domain]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    base_dir = Path(args.base_dir)
    for task, total_steps in tasks.items():
        if args.steps is not None:
            total_steps = args.steps
        train_task(domain=domain, task=task,
                   total_steps=total_steps, base_dir=base_dir)


if __name__ == "__main__":
    main()