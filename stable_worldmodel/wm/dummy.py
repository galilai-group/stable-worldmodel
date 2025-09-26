import torch
import numpy as np

import stable_pretraining as spt


def transform(info_dict):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = spt.data.transforms.Compose(
        spt.data.transforms.ToImage(
            mean=mean,
            std=std,
            source="pixels",
            target="pixels",
        ),
        spt.data.transforms.ToImage(
            mean=mean,
            std=std,
            source="goal",
            target="goal",
        ),
    )

    return transform(info_dict)


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
        """Predict next s_t+H embedding given s_t + action sequence i.e rollout the dynamics model for H steps."""
        z_obs = obs["embedding"]

        if torch.is_tensor(actions):
            return self.predictor(torch.cat([z_obs, actions], 1))

        elif type(actions) in [tuple, list]:
            for a in actions:
                z_obs = self.predictor(torch.cat([z_obs, a], 1))
            return z_obs

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        horizon = action_candidates.shape[1]
        actions = action_candidates.float()  # (B,T,A) -> (B*T,A)

        info_dict = transform(info_dict)

        # (n_envs, C, H, W)
        obs = info_dict["pixels"].flatten(1)
        goal = info_dict["goal"].flatten(1)

        embedding = self.encoder(obs)
        goal = self.encoder(goal)

        # -- predict next states
        preds = embedding
        for t in range(horizon):
            preds = self.predictor(torch.cat([preds, actions[:, t]], 1))

        # -- compute cost as distance to goal
        # REM: SHOULD BE A COST PER ENV
        cost = torch.nn.functional.mse_loss(preds, goal, reduction="none").mean(1)

        return cost
