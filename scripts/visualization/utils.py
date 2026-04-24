"""Shared grid utilities for environment visualization."""

import torch
import torch.nn as nn
import numpy as np
from loguru import logger as logging

from stable_worldmodel.envs.ogbench.cube_env import CubeEnv
from stable_worldmodel.envs.pusht.env import PushT
from stable_worldmodel.envs.two_room.env import TwoRoomEnv


class LeWMAdapter(nn.Module):
    """Wraps LeWM for visualization.

    Skips NaN-valued action at reset boundaries.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        # LeWM has no extra_encoders dict; provide an empty one for compatibility
        self.extra_encoders = {}

    def encode(self, info, pixels_key='pixels', target='emb'):
        encode_info = dict(info)
        if (
            'action' in encode_info
            and isinstance(encode_info['action'], torch.Tensor)
            and encode_info['action'].isnan().any()
        ):
            encode_info.pop('action')
        result = self.model.encode(encode_info)
        info.update(result)
        return info


class PreJEPAAdapter(nn.Module):
    """Wraps PreJEPA for visualization.

    Filters out NaN-valued keys (e.g. action at reset) from the extra encoders
    before calling encode, so only valid state observations are embedded.
    Exposes a flat (B, T, P*d) interface for predict.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self._n_patches = None
        self._d = None

    def encode(self, info, pixels_key='pixels', target='emb'):
        emb_keys = [
            k
            for k in self.model.extra_encoders
            if k in info
            and isinstance(info[k], torch.Tensor)
            and not info[k].isnan().any()
        ]
        return self.model.encode(
            info, pixels_key=pixels_key, target=target, emb_keys=emb_keys
        )

    def predict(self, emb, act_emb=None):
        """(B, T, P*d) -> (B, T, P*d). act_emb is unused (already baked into emb)."""
        B, T = emb.shape[:2]
        emb_4d = emb.reshape(B, T, self._n_patches, self._d)
        preds = self.model.predict(emb_4d)
        return preds.flatten(start_dim=2)


def get_state_from_grid(env, grid_element, dim: int | list = 0):
    """Convert grid element to full state vector.

    Args:
        env: The environment instance (PushT, TwoRoomEnv, or CubeEnv).
        grid_element: The grid coordinates to convert.
        dim: Dimension index or list of dimension indices to vary.

    Returns:
        Full state vector with grid_element values at specified dimensions.
    """
    # first retrieve the reference state depending on the env type
    if isinstance(dim, int):
        dim = [dim]
    if isinstance(env, PushT):
        reference_state = np.concatenate(
            [
                env.variation_space['agent']['start_position'].value.tolist(),
                env.variation_space['block']['start_position'].value.tolist(),
                [env.variation_space['block']['angle'].value],
                env.variation_space['agent']['velocity'].value.tolist(),
            ]
        )
        # get the positions of the block and the agent closer
        reference_state[2:4] = reference_state[0:2] + 0.3 * (
            reference_state[2:4] - reference_state[0:2]
        )
    elif isinstance(env, TwoRoomEnv):
        reference_state = env.variation_space['agent']['position'].value
    elif isinstance(env, CubeEnv):
        qpos0 = env._model.qpos0.copy()
        qvel0 = np.zeros(env._model.nv, dtype=qpos0.dtype)
        reference_state = np.concatenate([qpos0, qvel0])
    else:
        raise NotImplementedError(
            f'get_state_from_grid not implemented for env type: {type(env)}'
        )

    # computing the state from a grid element
    grid_state = reference_state.copy()
    for i, d in enumerate(dim):
        grid_state[d] = grid_element[i]
    if isinstance(env, PushT):
        # relative position of agent and block remains the same
        # we set the position of the block accordingly
        grid_state[2:4] = grid_state[0:2] + (
            reference_state[2:4] - reference_state[0:2]
        )
    elif isinstance(env, TwoRoomEnv):
        # TODO should check position is feasible
        pass
    elif isinstance(env, CubeEnv):
        # TODO should check position is feasible
        pass
    return grid_state


def get_rotation_states(env, n_steps: int = 100):
    """Generate PushT states with fixed agent/block positions and varying block angle.

    Agent is fixed at the bottom-right corner of its spawn range, the T-block
    is fixed at the centre of its spawn range, and the block angle sweeps
    uniformly from 0 to 2π.

    Args:
        env: A PushT environment instance.
        n_steps: Number of angle samples in [0, 2π).

    Returns:
        angles: (N,) array of block angles in radians.
        state_list: List of N full state vectors.
    """
    if not isinstance(env, PushT):
        raise NotImplementedError(
            f'Rotation state sweep not implemented for env type: {type(env)}'
        )

    agent_space = env.variation_space['agent']['start_position']
    block_space = env.variation_space['block']['start_position']

    # Agent fixed at bottom-right corner (high end of its spawn range)
    agent_pos = agent_space.high.copy()

    # Block fixed at centre of its spawn range
    block_pos = (block_space.low + block_space.high) / 2.0

    # Zero velocity
    vel = np.zeros(2, dtype=np.float32)

    angles = np.linspace(0, 2 * np.pi, n_steps, endpoint=False)
    state_list = [
        np.concatenate([agent_pos, block_pos, [angle], vel])
        for angle in angles
    ]

    return angles, state_list


def get_state_grid(env, grid_size: int = 10):
    """Generate a grid of states for the environment.

    Args:
        env: The environment instance (PushT, TwoRoomEnv, or CubeEnv).
        grid_size: Number of points along each dimension.

    Returns:
        Tuple of (grid, state_grid) where:
        - grid: (N, 2) array of grid coordinates
        - state_grid: List of full state vectors for each grid point
    """
    logging.info(f'Generating state grid for env type: {type(env)}')

    if isinstance(env, PushT):
        dim = [0, 1]  # Agent X, Y
        # Extract low/high limits for the specified dims
        min_val = [
            env.variation_space['agent']['start_position'].low[d] for d in dim
        ]
        max_val = [
            env.variation_space['agent']['start_position'].high[d] for d in dim
        ]
        range_val = [max_v - min_v for min_v, max_v in zip(min_val, max_val)]
        # decrease range a bit to avoid unreachable states
        min_val = [min_v + 0.15 * r for min_v, r in zip(min_val, range_val)]
        max_val = [max_v - 0.15 * r for max_v, r in zip(max_val, range_val)]
    elif isinstance(env, TwoRoomEnv):
        dim = [0, 1]  # Agent X, Y
        # Extract low/high limits for the specified dims
        min_val = [
            env.variation_space['agent']['position'].low[d] for d in dim
        ]
        max_val = [
            env.variation_space['agent']['position'].high[d] for d in dim
        ]
        # decrease range a bit to avoid unreachable states
        range_val = [max_v - min_v for min_v, max_v in zip(min_val, max_val)]
        min_val = [min_v + 0.1 * r for min_v, r in zip(min_val, range_val)]
        max_val = [max_v - 0.1 * r for max_v, r in zip(max_val, range_val)]
    elif isinstance(env, CubeEnv):
        env._mode = 'data_collection'
        cube_pos_start = int(
            np.asarray(env._model.joint('object_joint_0').qposadr).reshape(-1)[
                0
            ]
        )
        dim = [cube_pos_start, cube_pos_start + 1]
        qpos0 = env._model.qpos0
        cube_xy = qpos0[cube_pos_start : cube_pos_start + 2]
        bounds = np.asarray(env._object_sampling_bounds, dtype=np.float64)
        half_range = np.minimum(cube_xy - bounds[0], bounds[1] - cube_xy)
        if np.any(half_range <= 0.0):
            min_val = bounds[0].tolist()
            max_val = bounds[1].tolist()
        else:
            min_val = (cube_xy - half_range).tolist()
            max_val = (cube_xy + half_range).tolist()
    else:
        raise NotImplementedError(
            f'State grid generation not implemented for env type: {type(env)}'
        )

    # Create linear spaces for each dimension
    linspaces = [
        np.linspace(mn, mx, grid_size) for mn, mx in zip(min_val, max_val)
    ]

    # Create the meshgrid and reshape to (N, 2)
    # Using indexing='ij' ensures x varies with axis 0, y with axis 1
    mesh = np.meshgrid(*linspaces, indexing='ij')
    grid = np.stack(mesh, axis=-1).reshape(-1, len(dim))

    # Convert grid points to full state vectors
    state_grid = [get_state_from_grid(env, x, dim) for x in grid]

    return grid, state_grid
