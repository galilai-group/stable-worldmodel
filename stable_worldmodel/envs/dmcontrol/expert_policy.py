import pickle
import numpy as np
from stable_worldmodel.policy import BasePolicy


class SB3Normalizer:
    """Normalizer for observations using Stable Baselines3 VecNormalize statistics."""

    def __init__(
        self,
        stats_path: str,
        clip_obs: float = 10.0,
        epsilon: float = 1e-8,
    ):
        with open(stats_path, 'rb') as f:
            vec_normalize = pickle.load(f)

        self.mean = vec_normalize.obs_rms.mean
        self.var = vec_normalize.obs_rms.var
        self.epsilon = epsilon
        self.clip_obs = clip_obs

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self.normalize(obs)

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        norm_obs = (obs - self.mean) / np.sqrt(self.var + self.epsilon)
        return np.clip(norm_obs, -self.clip_obs, self.clip_obs)


class ExpertPolicy(BasePolicy):
    def __init__(
        self,
        ckpt_path: str,
        vec_normalize_path: str,
        noise_std: float = 0.0,
        seed: int = None,
        device: str = 'cpu',
        **kwargs,
    ):
        """
        Expert Policy (RL) for DMControl Suite.
        Policies have been trained using Soft Actor-Critic (SAC) algorithm from stable_baselines3.

        Args:
            ckpt_path (str): Path to the stable_baselines3 .zip file of the trained policy.
            vec_normalize_path (str): Path to the .pkl file containing the normalization statistics.
            noise_std (float): Standard deviation of Gaussian noise added to actions.
            seed (int): Random seed for action noise.
            device (str): Device to load the model on, e.g., 'cpu' or 'cuda'.
        """
        super().__init__(**kwargs)

        try:
            import stable_baselines3 as sb3
        except ImportError:
            raise ImportError(
                'stable_baselines3 is required to use the ExpertPolicy. '
                "Please install it via 'uv add stable-baselines3'."
            )

        self.model = sb3.SAC.load(ckpt_path, device=device)
        self.normalizer = SB3Normalizer(vec_normalize_path)
        self.rng = np.random.default_rng(seed)
        self.noise_std = noise_std
        self.type = 'expert'

        if seed is not None:
            self.model.set_random_seed(seed)

    def set_env(self, env):
        self.env = env

    def get_action(self, info_dict, **kwargs):
        assert 'observation' in info_dict, (
            'Observation key missing in info_dict'
        )

        obs = info_dict['observation'].squeeze()

        if len(obs.shape) != 2:
            raise ValueError(
                f'Expected observation shape (num_envs, obs_dim), got {obs.shape}'
            )

        normalized_obs = self.normalizer(obs)
        actions, _ = self.model.predict(normalized_obs, deterministic=False)
        if self.noise_std > 0:
            noise = self.rng.normal(0, self.noise_std, size=actions.shape)
            actions = actions + noise
        return np.clip(actions, -1.0, 1.0).astype(np.float32)


if __name__ == '__main__':
    policy = ExpertPolicy(
        ckpt_path='../stable-expert/models/sac_dmcontrol/cheetah/expert_policy.zip',
        vec_normalize_path='../stable-expert/models/sac_dmcontrol/cheetah/vec_normalize.pkl',
    )

    print(policy)
