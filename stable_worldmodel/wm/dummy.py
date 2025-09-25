import torch
import numpy as np


class DummyWorldModel(torch.nn.Module):
    def __init__(self, image_shape, action_dim):
        super().__init__()
        self.encoder = torch.nn.Linear(np.prod(image_shape), 10)
        self.predictor = torch.nn.Linear(10 + action_dim, 10)

    def encode(self, obs):
        if type(obs["pixels"]) is np.ndarray:
            obs["pixels"] = torch.from_numpy(obs["pixels"]).float()
        obs["embedding"] = self.encoder(obs["pixels"].flatten(1))
        return obs

    def predict(self, obs, actions, timestep=None):
        """predict next s_t+H embedding given s_t + action sequence
        i.e rollout the dynamics model for H steps
        """

        z_obs = obs["embedding"]

        if torch.is_tensor(actions):
            return self.predictor(torch.cat([z_obs, actions], 1))

        elif type(actions) in [tuple, list]:
            for a in actions:
                z_obs = self.predictor(torch.cat([z_obs, a], 1))
            return z_obs
