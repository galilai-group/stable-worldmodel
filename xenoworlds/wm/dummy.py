import torch
import numpy as np


class DummyWorldModel(torch.nn.Module):
    def __init__(self, image_shape, action_dim):
        super().__init__()
        self.encoder = torch.nn.Linear(np.prod(image_shape), 10)
        self.predictor = torch.nn.Linear(10 + action_dim, 10)

    def forward(self, states, actions):
        encoding = self.encoder(states.flatten(1))
        if torch.is_tensor(actions):
            return self.predictor(torch.cat([encoding, actions], 1))
        elif type(actions) in [tuple, list]:
            for a in actions:
                encoding = self.predictor(torch.cat([encoding, a], 1))
            return encoding

    def encode(self, states):
        return self.encoder(states.flatten(1))
