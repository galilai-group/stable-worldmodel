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
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import stable_worldmodel


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
        'run': {**DEFAULT_CFG, 'total_timesteps': 750_000},
        'run-backward': {**DEFAULT_CFG, 'total_timesteps': 750_000},
        'run-front': {**DEFAULT_CFG, 'total_timesteps': 750_000},
        'run-back': {**DEFAULT_CFG, 'total_timesteps': 750_000},
        'stand-front': {**DEFAULT_CFG, 'total_timesteps': 1_000_000},
        'stand-back': {**DEFAULT_CFG, 'total_timesteps': 1_000_000},
        'lie-down': {**DEFAULT_CFG, 'total_timesteps': 1_000_000},
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
        'spin': {**DEFAULT_CFG, 'total_timesteps': 750_000},
        'turn_easy': {**DEFAULT_CFG, 'total_timesteps': 1_000_000},
        'turn_hard': {**DEFAULT_CFG, 'total_timesteps': 1_500_000},
    },
    'humanoid': {
        'stand': {**HUMANOID_CFG, 'total_timesteps': 5_000_000},
        'walk': {**HUMANOID_CFG, 'total_timesteps': 5_000_000},
        'run': {**HUMANOID_CFG, 'total_timesteps': 5_000_000},
    },
}


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
    ):
        """Initializes the DMControlTrainer.

        Args:
            domain_name (str): The name of the DMControl domain (e.g., 'walker').
            task_name (str): The specific task within the domain (e.g., 'run').
            config (Dict[str, Any]): Dictionary containing SAC hyperparameters and total_timesteps.
            base_dir (str, optional): Base directory for saving models. Defaults to "./models/sac_dmcontrol".
        """
        self.domain = domain_name
        self.task = task_name
        self.config = config.copy()
        self.total_timesteps = self.config.pop('total_timesteps')

        self.gym_id = f'swm/{self.domain.capitalize()}DMControl-v0'
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
            f'Training {self.domain} | Task: {self.task} | Steps: {self.total_timesteps}'
        )
        logger.info(f'Device: {self.device} | Config: {self.config}')

        vec_env = DummyVecEnv([self.make_env])
        vec_env = VecNormalize(
            vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0
        )

        sac_kwargs = {
            'policy': 'MlpPolicy',
            'env': vec_env,
            'verbose': 1,
            'learning_rate': 3e-4,
            'buffer_size': 1_000_000,
            'ent_coef': 'auto',
            'device': self.device,
        }
        sac_kwargs.update(self.config)

        model = SAC(**sac_kwargs)

        checkpoint_callback = CheckpointCallback(
            save_freq=200_000,
            save_path=str(self.save_dir),
            name_prefix=f'sac_{self.domain}_{self.task}',
        )

        try:
            model.learn(
                total_timesteps=self.total_timesteps,
                callback=checkpoint_callback,
                progress_bar=True,
                log_interval=30,
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
        default='./models/sac_dmcontrol',
        help='Output directory',
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
        logger.error(f"Domain '{domain}' not found in configs.")
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
            base_dir=args.base_dir,
        )
        trainer.train()


if __name__ == '__main__':
    main()
