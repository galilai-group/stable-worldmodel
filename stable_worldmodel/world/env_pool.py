"""EnvPool: vectorized env wrapper with selective stepping.

A lightweight replacement for ``gymnasium.vector.SyncVectorEnv`` with two
differences tailored to ``World``:

1. ``reset(mask=...)`` and ``step(mask=...)`` can skip individual envs —
   useful for the ``wait`` reset mode where terminated envs freeze until
   every env has finished.
2. The stacked info dict is pre-allocated on the first reset and updated
   in-place afterwards. Tensor/array values are shaped ``(num_envs, 1, ...)``
   so consumers can rely on a ``(batch, time, ...)`` convention without the
   pool re-stacking every step.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector.utils import batch_space


class EnvPool:
    """Batched env runner with selective stepping.

    Args:
        env_fns: List of zero-arg factories, one per env. Each is called
            once and the result is kept for the lifetime of the pool.
    """

    def __init__(self, env_fns: list):
        self.envs = [fn() for fn in env_fns]
        self._single_env = self.envs[0]
        self._stacked_infos: dict[str, Any] | None = None
        self.seeds = np.zeros(len(self.envs), dtype=np.int64)
        # Cache batched spaces — rebuilding them per-access creates a fresh
        # unseeded space each call, so .seed() / .sample() never advances RNG.
        self._action_space = batch_space(
            self._single_env.action_space, len(self.envs)
        )
        self._observation_space = batch_space(
            self._single_env.observation_space, len(self.envs)
        )

    @property
    def num_envs(self) -> int:
        """Number of envs in the pool."""
        return len(self.envs)

    @property
    def action_space(self) -> gym.Space:
        """Batched action space (``batch_space(single_action_space, num_envs)``)."""
        return self._action_space

    @property
    def single_action_space(self) -> gym.Space:
        """Action space of a single env."""
        return self._single_env.action_space

    @property
    def observation_space(self) -> gym.Space:
        """Batched observation space."""
        return self._observation_space

    @property
    def single_observation_space(self) -> gym.Space:
        """Observation space of a single env."""
        return self._single_env.observation_space

    @property
    def variation_space(self):
        """Variation space from the unwrapped env, or ``None`` if not defined."""
        return getattr(self._single_env.unwrapped, 'variation_space', None)

    @property
    def single_variation_space(self):
        """Variation space for a single env (alias of ``variation_space``)."""
        return self.variation_space

    def reset(
        self,
        seed: int | list[int | None] | None = None,
        options: dict | list[dict | None] | None = None,
        mask: np.ndarray | None = None,
    ) -> tuple[None, dict]:
        """Reset envs and return the stacked info dict.

        Args:
            seed: Base int (each env gets ``seed + i``), a per-env list,
                or ``None``.
            options: Shared dict or per-env list.
            mask: If provided, only envs where ``mask[i]`` is truthy are
                reset. Others keep their current state in the stacked
                info buffer.
        """
        seeds = _broadcast_arg(seed, self.num_envs, increment=True)
        opts = _broadcast_arg(options, self.num_envs)

        per_env_infos = [None] * self.num_envs
        for i, env in enumerate(self.envs):
            if mask is not None and not mask[i]:
                continue
            _, per_env_infos[i] = env.reset(seed=seeds[i], options=opts[i])
            if seeds[i] is not None:
                self.seeds[i] = seeds[i]

        if self._stacked_infos is None or mask is None:
            self._stacked_infos = _stack_fresh(per_env_infos)
        else:
            for i, info in enumerate(per_env_infos):
                if info is not None:
                    _write_env_info(self._stacked_infos, i, info)

        return None, self._stacked_infos

    def step(
        self, actions: np.ndarray, mask: np.ndarray | None = None
    ) -> tuple[None, np.ndarray, np.ndarray, np.ndarray, dict]:
        """Step envs and return ``(None, rewards, terminateds, truncateds, infos)``.

        Args:
            actions: Array of shape ``(num_envs, ...)`` — one action per env.
            mask: If provided, only envs where ``mask[i]`` is truthy are
                stepped. Masked envs contribute zero reward and ``False``
                termination/truncation, and their slot in the stacked
                info buffer is left unchanged.
        """
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminateds = np.zeros(self.num_envs, dtype=bool)
        truncateds = np.zeros(self.num_envs, dtype=bool)

        for i, env in enumerate(self.envs):
            if mask is not None and not mask[i]:
                continue
            _, rewards[i], terminateds[i], truncateds[i], info = env.step(
                actions[i]
            )
            _write_env_info(self._stacked_infos, i, info)

        return None, rewards, terminateds, truncateds, self._stacked_infos

    def close(self):
        """Close every env in the pool."""
        for env in self.envs:
            env.close()


def _broadcast_arg(arg, n: int, increment: bool = False) -> list:
    if arg is None:
        return [None] * n
    if isinstance(arg, list):
        return arg
    if isinstance(arg, np.ndarray):
        return list(arg)
    if increment and isinstance(arg, int):
        return [arg + i for i in range(n)]
    return [arg] * n


def _stack_fresh(per_env_infos: list[dict]) -> dict[str, Any]:
    """Build stacked info arrays from a full set of per-env infos.

    Tensor/array values get a leading time dim of 1 after the env dim,
    yielding shape (N, 1, ...) so downstream consumers can rely on a
    (batch, time, ...) convention.
    """
    keys = per_env_infos[0].keys()
    stacked = {}
    for k in keys:
        vals = [info[k] for info in per_env_infos]
        first = vals[0]
        if isinstance(first, torch.Tensor):
            stacked[k] = torch.stack(vals).unsqueeze(1)
        elif isinstance(first, np.ndarray):
            stacked[k] = np.stack(vals)[:, None, ...]
        elif isinstance(first, (bool, int, float, np.number)):
            stacked[k] = np.array(vals)[:, None]
        else:
            stacked[k] = [[v] for v in vals]
    return stacked


def _write_env_info(stacked: dict, idx: int, info: dict) -> None:
    """Write a single env's info into pre-allocated stacked arrays in-place."""
    for k, v in info.items():
        if k not in stacked:
            continue
        buf = stacked[k]
        if isinstance(buf, torch.Tensor):
            if not isinstance(v, torch.Tensor):
                v = torch.as_tensor(v, dtype=buf.dtype, device=buf.device)
            buf[idx, 0] = v
        elif isinstance(buf, np.ndarray):
            buf[idx, 0] = v
        elif isinstance(buf, list):
            buf[idx][0] = v
