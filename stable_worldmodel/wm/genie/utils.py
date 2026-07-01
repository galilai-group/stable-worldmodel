import torch
from torch import nn
from typing import Iterator
import math

def cosine_schedule(u: torch.Tensor | float) -> torch.Tensor | float:
    if isinstance(u, torch.Tensor): cos = torch.cos
    if isinstance(u, float): cos = math.cos
    return cos(u * math.pi / 2)


def init_weights(modules: Iterator[nn.Module]):
    std = 0.02
    
    def _init_linear(m_: nn.Linear):
        m_.weight.data.normal_(mean=0.0, std=std)
        if m_.bias is not None: m_.bias.data.zero_()

    def _init_embedding(m_: nn.Embedding):
        m_.weight.data.normal_(mean=0.0, std=std)
        if m_.padding_idx is not None: m_.weight.data[m_.padding_idx].zero_()

    for module in modules:
        if isinstance(module, nn.Linear): _init_linear(module)
        elif isinstance(module, nn.Embedding): _init_embedding(module)
