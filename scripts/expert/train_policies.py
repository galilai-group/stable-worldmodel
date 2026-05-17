"""Script for training SAC expert policies on DMControl environments."""

import os

os.environ['MUJOCO_GL'] = 'egl'

import argparse
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from loguru import logger
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
import stable_worldmodel

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# Define default architectures and hyperparameters

ARCH_SMALL = {'net_arch': [256, 256]}
ARCH_MEDIUM = {'net_arch': [400, 300]}
ARCH_LARGE = {'net_arch': [1024, 1024]}

DEFAULT_CFG = {
    'batch_size': 256,
    'policy_kwargs': ARCH_SMALL,
    'learning_starts': 10000,
}

QUADRUPED_CFG = {
    'batch_size': 1024,
    'gradient_steps': 1,
    'learning_starts': 10000,
    'policy_kwargs': ARCH_MEDIUM,
    'tau': 0.005,
}

WALKER_CFG = {
    'batch_size': 1024,
    'gradient_steps': 2,
    'train_freq': 1,
    'policy_kwargs': ARCH_MEDIUM,
    'learning_starts': 10000,
    'tau': 0.005,
}

HUMANOID_CFG = {
    'batch_size': 1024,
    'gradient_steps': 2,
    'policy_kwargs': ARCH_LARGE,
    'learning_starts': 25000,
}

# Registry mapping domains and tasks to their SAC hyperparameters
PARAMS_REGISTRY = {
    'pendulum': {
        'swingup': {
            **DEFAULT_CFG,
            'gradient_steps': 2,
            'batch_size': 1024,
            'learning_starts': 25000,
            'policy_kwargs': ARCH_MEDIUM,
            'total_timesteps': 750_000,
        }
    },
    'ballincup': {'default': {**DEFAULT_CFG, 'total_timesteps': 750_000}},
    'cartpole': {'default': {**DEFAULT_CFG, 'total_timesteps': 750_000}},
    'quadruped': {
        'walk': {**QUADRUPED_CFG, 'total_timesteps': 2_500_000},
        'run': {**QUADRUPED_CFG, 'total_timesteps': 3_500_000},
    },
    'cheetah': {
        'run': {**DEFAULT_CFG, 'total_timesteps': 1_500_000},
        'run-backward': {**DEFAULT_CFG, 'total_timesteps': 1_500_000},
        'run-front': {**DEFAULT_CFG, 'total_timesteps': 1_500_000},
        'run-back': {**DEFAULT_CFG, 'total_timesteps': 1_500_000},
        'stand-front': {**DEFAULT_CFG, 'total_timesteps': 1_500_000},
        'stand-back': {**DEFAULT_CFG, 'total_timesteps': 1_500_000},
        'lie-down': {**DEFAULT_CFG, 'total_timesteps': 1_500_000},
        'jump': {**DEFAULT_CFG, 'total_timesteps': 1_500_000},
        'legs-up': {**DEFAULT_CFG, 'total_timesteps': 1_500_000},
        'flip': {**DEFAULT_CFG, 'total_timesteps': 1_500_000},
        'flip-backward': {**DEFAULT_CFG, 'total_timesteps': 1_500_000},
    },
    'reacher': {
        'easy': {
            **DEFAULT_CFG,
            'total_timesteps': 750_000,
            'learning_starts': 5000,
        },
        'hard': {
            **DEFAULT_CFG,
            'total_timesteps': 1_000_000,
            'learning_starts': 5000,
        },
    },
    'walker': {
        'stand': {**WALKER_CFG, 'total_timesteps': 1_000_000},
        'walk': {**WALKER_CFG, 'total_timesteps': 1_000_000},
        'run': {**WALKER_CFG, 'total_timesteps': 1_500_000},
        'walk-backward': {**WALKER_CFG, 'total_timesteps': 1_500_000},
        'lie_down': {**WALKER_CFG, 'total_timesteps': 1_500_000},
        'flip': {**WALKER_CFG, 'total_timesteps': 2_500_000},
        'arabesque': {**WALKER_CFG, 'total_timesteps': 2_500_000},
        'legs_up': {**WALKER_CFG, 'total_timesteps': 2_500_000},
    },
    'hopper': {
        'stand': {
            **DEFAULT_CFG,
            'batch_size': 1024,
            'gradient_steps': 2,
            'tau': 0.005,
            'total_timesteps': 2_000_000,
        },
        'hop': {
            **DEFAULT_CFG,
            'batch_size': 1024,
            'gradient_steps': 2,
            'tau': 0.005,
            'total_timesteps': 2_500_000,
        },
        'hop-backward': {
            **DEFAULT_CFG,
            'batch_size': 1024,
            'gradient_steps': 2,
            'tau': 0.005,
            'total_timesteps': 4_000_000,
        },
        'flip': {
            **DEFAULT_CFG,
            'batch_size': 1024,
            'gradient_steps': 2,
            'tau': 0.005,
            'total_timesteps': 4_000_000,
        },
        'flip-backward': {
            **DEFAULT_CFG,
            'batch_size': 1024,
            'gradient_steps': 2,
            'tau': 0.005,
            'total_timesteps': 4_000_000,
        },
    },
    'finger': {
        'spin': {**DEFAULT_CFG, 'total_timesteps': 1_500_000},
        'turn_easy': {**DEFAULT_CFG, 'total_timesteps': 1_500_000},
        'turn_hard': {**DEFAULT_CFG, 'total_timesteps': 3_000_000},
    },
    'humanoid': {
        'stand': {**HUMANOID_CFG, 'total_timesteps': 5_000_000},
        'walk': {**HUMANOID_CFG, 'total_timesteps': 5_000_000},
        'run': {**HUMANOID_CFG, 'total_timesteps': 5_000_000},
    },
}


class RewardLoggerCallback(BaseCallback):
    """Logs episode rewards to .npy and optionally to wandb."""

    def __init__(
        self,
        save_path: str,
        save_freq: int = 5_000,
        use_wandb: bool = False,
        train_log_freq: int = 500,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_path = Path(save_path)
        self.save_freq = save_freq
        self.use_wandb = use_wandb
        self.train_log_freq = train_log_freq
        self._log: list[list[float]] = []
        self._last_save = 0
        self._last_train_log = 0
        self._ep_rew_acc: np.ndarray | None = None
        self._completed_ep_rews: list[float] = []

    def _on_training_start(self) -> None:
        self._ep_rew_acc = np.zeros(self.training_env.num_envs)

    def _on_step(self) -> bool:
        rewards = np.asarray(self.locals.get('rewards', np.zeros(self.training_env.num_envs)))
        dones = np.asarray(self.locals.get('dones', np.zeros(self.training_env.num_envs, dtype=bool)))

        self._ep_rew_acc += rewards
        for i, done in enumerate(dones):
            if done:
                ep_rew = float(self._ep_rew_acc[i])
                self._completed_ep_rews.append(ep_rew)
                self._log.append([float(self.num_timesteps), ep_rew])
                self._ep_rew_acc[i] = 0.0

        if self.use_wandb and self.num_timesteps - self._last_train_log >= self.train_log_freq:
            metrics: dict = {}
            if self._completed_ep_rews:
                metrics['rollout/ep_rew_mean'] = float(np.mean(self._completed_ep_rews[-100:]))
            metrics.update({
                k: v for k, v in self.model.logger.name_to_value.items()
                if k.startswith('train/')
            })
            if metrics:
                wandb.log(metrics, step=self.num_timesteps)
            self._last_train_log = self.num_timesteps

        if self.num_timesteps - self._last_save >= self.save_freq:
            self._flush()
            self._last_save = self.num_timesteps
        return True

    def _on_training_end(self) -> None:
        self._flush()

    def _flush(self) -> None:
        if self._log:
            np.save(self.save_path, np.array(self._log, dtype=np.float32))


class DMControlTrainer:
    """Trainer class for running Soft Actor-Critic (SAC) on DMControl environments.

    This class orchestrates environment creation, observation normalization, model
    initialization, and the execution of the training loop using Stable Baselines3.
    """

    def __init__(
        self,
        domain_name: str,
        task_name: str,
        config: dict[str, Any],
        base_dir: str = './models/sac_dmcontrol',
        n_envs: int = 4,
        seed: int = 0,
        use_wandb: bool = False,
    ):
        """Initializes the DMControlTrainer.

        Args:
            domain_name (str): The name of the DMControl domain (e.g., 'walker').
            task_name (str): The specific task within the domain (e.g., 'run').
            config (Dict[str, Any]): Dictionary containing SAC hyperparameters and total_timesteps.
            base_dir (str, optional): Base directory for saving models. Defaults to "./models/sac_dmcontrol".
            n_envs (int, optional): Number of parallel envs for data collection. Defaults to 4.
            seed (int, optional): Random seed. Defaults to 0.
            use_wandb (bool, optional): Whether to log to Weights & Biases. Defaults to False.
        """
        self.domain = domain_name
        self.task = task_name
        self.config = config.copy()
        self.total_timesteps = self.config.pop('total_timesteps')
        self.n_envs = n_envs
        self.seed = seed
        self.use_wandb = use_wandb and WANDB_AVAILABLE

        _GYM_NAME = {'ballincup': 'BallInCup'}
        gym_name = _GYM_NAME.get(self.domain, self.domain.capitalize())
        self.gym_id = f'swm/{gym_name}DMControl-v0'
        self.save_dir = Path(base_dir) / f'{self.domain.lower()}_{self.task}'
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def make_env(self) -> gym.Env:
        """Creates and wraps the specific DMControl environment.

        Handles domain-specific edge cases for environments that do not accept a task
        argument, casts observations to float32, and attaches a monitoring wrapper.

        Returns:
            gym.Env: The configured Gymnasium environment.
        """
        env_kwargs = {'task': self.task}

        if any(
            d in self.gym_id.lower()
            for d in ['pendulum', 'cartpole', 'ballincup']
        ):
            env_kwargs.pop('task', None)

        env = gym.make(self.gym_id, **env_kwargs)

        f32_space = gym.spaces.Box(
            low=env.observation_space.low,
            high=env.observation_space.high,
            shape=env.observation_space.shape,
            dtype=np.float32,
        )
        env = gym.wrappers.TransformObservation(
            env, lambda obs: obs.astype(np.float32), f32_space
        )
        return Monitor(env)

    def train(self):
        """Executes the SAC training loop.

        Sets up the vectorized environment, applies observation normalization, initializes
        the SAC model with the injected configuration, and trains the agent. Saves periodic
        checkpoints, the final expert policy, and normalization statistics.
        """
        logger.info(
            f'Training {self.domain} | Task: {self.task} | Steps: {self.total_timesteps} | n_envs: {self.n_envs} | seed: {self.seed}'
        )
        logger.info(f'Device: {self.device} | Config: {self.config}')

        if self.use_wandb:
            wandb.init(
                project='dmcontrol-sac',
                name=f'{self.domain}_{self.task}_seed{self.seed}',
                config={
                    **self.config,
                    'domain': self.domain,
                    'task': self.task,
                    'total_timesteps': self.total_timesteps,
                    'n_envs': self.n_envs,
                    'seed': self.seed,
                },
                sync_tensorboard=False,
            )
        elif not WANDB_AVAILABLE and self.use_wandb:
            logger.warning('wandb not installed — skipping. Run: pip install wandb')

        env_fns = [self.make_env for _ in range(self.n_envs)]
        if self.n_envs > 1:
            vec_env = SubprocVecEnv(env_fns)
        else:
            vec_env = DummyVecEnv(env_fns)
        vec_env = VecNormalize(
            vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0
        )

        sac_kwargs = {
            'policy': 'MlpPolicy',
            'env': vec_env,
            'verbose': 0,
            'learning_rate': 3e-4,
            'buffer_size': 1_000_000,
            'ent_coef': 'auto',
            'device': self.device,
            'seed': self.seed,
        }
        sac_kwargs.update(self.config)

        model = SAC(**sac_kwargs)

        checkpoint_callback = CheckpointCallback(
            save_freq=max(200_000 // self.n_envs, 1),
            save_path=str(self.save_dir),
            name_prefix=f'sac_{self.domain}_{self.task}',
        )
        reward_callback = RewardLoggerCallback(
            save_path=str(self.save_dir / 'rewards.npy'),
            save_freq=5_000,
            use_wandb=self.use_wandb,
        )
        callbacks = CallbackList([checkpoint_callback, reward_callback])

        try:
            model.learn(
                total_timesteps=self.total_timesteps,
                callback=callbacks,
                progress_bar=True,
                log_interval=10,
            )

            model.save(self.save_dir / 'expert_policy')
            vec_env.save(str(self.save_dir / 'vec_normalize.pkl'))
            logger.success(
                f'Completed training for {self.domain}::{self.task}'
            )

        except Exception as e:
            logger.error(
                f'Failed training for {self.domain}::{self.task}. Error: {e}'
            )
            raise e
        finally:
            vec_env.close()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if self.use_wandb:
                wandb.finish()


def main():
    logger.info(
        f'Initialized stable_worldmodel environments (from {stable_worldmodel.__file__})'
    )
    parser = argparse.ArgumentParser(
        description='Train SAC expert policies on DMControl'
    )
    parser.add_argument(
        '--domain', type=str, help='Domain name (e.g., walker, cheetah)'
    )
    parser.add_argument(
        '--task',
        type=str,
        help='Specific task. If empty, runs all tasks in the domain.',
    )
    parser.add_argument(
        '--base_dir',
        type=str,
        default=None,
        help='Output directory (default: ./models/sac_dmcontrol)',
    )
    parser.add_argument(
        '--n_envs',
        type=int,
        default=4,
        help='Number of parallel environments (default: 4)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed (default: 0)',
    )
    parser.add_argument(
        '--wandb',
        action='store_true',
        help='Log metrics to Weights & Biases (requires: pip install wandb)',
    )
    parser.add_argument(
        '--list', action='store_true', help='List all available configurations'
    )

    args = parser.parse_args()

    if args.list:
        for domain, tasks in PARAMS_REGISTRY.items():
            logger.info(f'[{domain}]: {", ".join(tasks.keys())}')
        return

    if not args.domain:
        logger.error('Please specify a --domain (or use --list)')
        sys.exit(1)

    domain = args.domain.lower()

    if domain not in PARAMS_REGISTRY:
        logger.error(
            f"Domain '{domain}' not found. Use --list to see available options."
        )
        sys.exit(1)

    tasks_to_run = []
    if args.task:
        if args.task not in PARAMS_REGISTRY[domain]:
            logger.error(f"Task '{args.task}' not found in {domain} config.")
            sys.exit(1)
        tasks_to_run = [args.task]
    else:
        tasks_to_run = list(PARAMS_REGISTRY[domain].keys())

    for task in tasks_to_run:
        config = PARAMS_REGISTRY[domain][task]
        trainer = DMControlTrainer(
            domain_name=domain,
            task_name=task,
            config=config,
            base_dir=args.base_dir or './models/sac_dmcontrol',
            n_envs=args.n_envs,
            seed=args.seed,
            use_wandb=args.wandb,
        )
        trainer.train()


if __name__ == '__main__':
    main()
