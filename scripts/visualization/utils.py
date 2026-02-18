"""Shared grid utilities for environment visualization."""

import numpy as np
from loguru import logger as logging

from stable_worldmodel.envs.ogbench.cube_env import CubeEnv
from stable_worldmodel.envs.pusht.env import PushT
from stable_worldmodel.envs.two_room.env import TwoRoomEnv


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
