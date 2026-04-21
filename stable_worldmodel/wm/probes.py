"""Helper functions for probing wm latent spaces."""

import torch
from torch import nn


def attach_probe(model, key, probe):
    """Attach a probe to the model under the given key."""
    assert isinstance(probe, nn.Module), 'Probe must be a nn.Module'
    if not hasattr(model, '_probes'):
        model._probes = nn.ModuleDict()
    model._probes[key] = probe


def get_probe(model, key):
    """Get the probe attached to the model under the given key."""
    if hasattr(model, '_probes'):
        return model._probes[key] if key in model._probes else None

    return None


def load_probe(model, key, path):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    is_module = isinstance(payload, nn.Module)

    if is_module:
        attach_probe(model, key, payload)
    elif isinstance(payload, dict):
        probe = get_probe(model, key)
        if probe is None:
            raise ValueError(f'No probe found for key {key} in model')

        probe.load_state_dict(payload)
    return


__all__ = [
    'attach_probe',
    'get_probe',
    'load_probe',
]
