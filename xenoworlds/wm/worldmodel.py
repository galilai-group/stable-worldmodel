from typing import Protocol

import torch

class WorldModel(Protocol):

    def encode(self, obs: dict) -> dict:...
    def predict(self, z_obs:dict, actions:torch.Tensor, timestep=None) -> dict:...

def decode_rollout():
    # if decoder exists
    raise NotImplementedError