import os
import argparse
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
except ImportError:
    wandb = None
    WandbCallback = None


def train_expert(
    env_id: str,
    total_timesteps: int,
    seed: int = 42,
    track: bool = False,
    project_name: str = 'stable-worldmodel',
):
    """
    Trains a Soft Actor-Critic (SAC) expert policy on a continuous control Fetch environment.
    SAC natively excels at continuous Cartesian robotic control arrays.
    """
    print('===================================================')
    print(f' Training Expert Policy for {env_id}')
    print(f' Setup: SAC | {total_timesteps} Timesteps | Seed: {seed}')
    print('===================================================')

    env = Monitor(gym.make(env_id))
    eval_env = Monitor(gym.make(env_id))

    model = SAC(
        'MlpPolicy',
        env,
        verbose=1,
        seed=seed,
        tensorboard_log=f'./logs/tensorboard/{env_id.replace("/", "_")}_sac/',
    )

    save_path = f'./policies/{env_id.replace("/", "_")}_expert'
    os.makedirs(save_path, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=save_path,
        eval_freq=5000,
        deterministic=True,
        render=False,
    )

    callbacks = [eval_callback]

    if track and wandb is None:
        raise ImportError(
            'wandb is required for tracking. Install it with: pip install wandb'
        )

    if track:
        wandb.init(
            project=project_name,
            name=f'SAC_{env_id.replace("/", "_")}',
            config={
                'env': env_id,
                'algo': 'SAC',
                'seed': seed,
                'timesteps': total_timesteps,
            },
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
        wandb_callback = WandbCallback(
            model_save_path=save_path, model_save_freq=5000, verbose=2
        )
        callbacks.append(wandb_callback)

    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList(callbacks),
        progress_bar=True,
    )

    model.save(f'{save_path}/final_model')

    if track:
        wandb.finish()

    print(f'Training complete. Models saved to {save_path}')
    env.close()
    eval_env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train an RL Expert Policy for Fetch Environments'
    )
    parser.add_argument(
        '--env',
        type=str,
        default='swm/FetchReach-v3',
        help='Target SWM Environment ID',
    )
    parser.add_argument(
        '--timesteps',
        type=int,
        default=100000,
        help='Total environment steps to execute',
    )
    parser.add_argument('--seed', type=int, default=42, help='RNG seed')
    parser.add_argument(
        '--track',
        action='store_true',
        help='Log training metrics natively to Weights & Biases',
    )
    parser.add_argument(
        '--project',
        type=str,
        default='stable-worldmodel',
        help='WandB Cloud project name',
    )

    args = parser.parse_args()

    train_expert(args.env, args.timesteps, args.seed, args.track, args.project)
