"""Stream MineRL demonstrations into an SWM Lance dataset.

MineRL exposes demonstrations as ordered ``(state, action, reward,
next_state, done)`` tuples.  This converter deliberately uses the *current*
state's ``pov`` RGB image and its corresponding action, which gives an
action-conditioned world model the transition ``(o_t, a_t) -> o_{t+1}``.

The MineRL package is imported only when conversion is requested.  This keeps
it an optional dependency: users who only work with existing SWM environments
do not need a Minecraft installation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from stable_worldmodel.data.formats.lance import LanceWriter


class MineRLDataPipeline(Protocol):
    """Small subset of MineRL's data pipeline used by this converter."""

    def get_trajectory_names(self) -> list[str]: ...

    def load_data(self, stream_name: str) -> Iterable[tuple[Any, ...]]: ...


@dataclass(frozen=True)
class MineRLConversionSummary:
    """A compact, serialisable conversion result."""

    output_path: Path
    trajectories: int
    transitions: int
    action_dim: int


class ActionVectorizer:
    """Flatten MineRL's nested action dictionaries deterministically.

    MineRL actions vary by task but are normally a tree of scalar/array-valued
    controls such as ``forward``, ``attack`` and ``camera``.  Sorting mapping
    keys gives one stable action layout for every trajectory.  The first action
    fixes the dimension; a later schema change fails early rather than silently
    training a model on misaligned controls.
    """

    def __init__(self) -> None:
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            raise RuntimeError('Action vectorizer has not seen an action yet.')
        return self._dim

    def __call__(self, action: Any) -> np.ndarray:
        vector = _flatten_numeric(action)
        if self._dim is None:
            self._dim = int(vector.size)
        elif vector.size != self._dim:
            raise ValueError(
                'MineRL action schema changed within this conversion: '
                f'expected {self._dim} values, got {vector.size}.'
            )
        return vector


def _flatten_numeric(value: Any) -> np.ndarray:
    """Convert a nested numeric MineRL action to one float32 vector."""
    if isinstance(value, Mapping):
        parts = [_flatten_numeric(value[key]) for key in sorted(value)]
        return np.concatenate(parts) if parts else np.empty(0, np.float32)

    if isinstance(value, (tuple, list)):
        parts = [_flatten_numeric(item) for item in value]
        return np.concatenate(parts) if parts else np.empty(0, np.float32)

    array = np.asarray(value)
    if array.dtype.kind not in 'biuf':
        raise TypeError(
            'MineRL action values must be numeric. Got '
            f'{array.dtype!s}; encode categorical controls before conversion.'
        )
    return array.astype(np.float32, copy=False).reshape(-1)


def _rgb_observation(state: Mapping[str, Any]) -> np.ndarray:
    """Extract MineRL's first-person RGB observation as HWC uint8."""
    if 'pov' not in state:
        raise KeyError("MineRL state has no 'pov' RGB observation.")
    frame = np.asarray(state['pov'])
    if frame.ndim != 3:
        raise ValueError(
            "MineRL state['pov'] must be a rank-3 RGB image, got "
            f'{frame.shape}.'
        )
    # Accept CHW input from a custom loader while storing the SWM writer's
    # canonical HWC image layout.
    if frame.shape[0] in (1, 3) and frame.shape[-1] not in (1, 3):
        frame = np.moveaxis(frame, 0, -1)
    if frame.shape[-1] != 3:
        raise ValueError(
            "MineRL state['pov'] must have three RGB channels, got "
            f'{frame.shape}.'
        )
    return frame.astype(np.uint8, copy=False)


def trajectory_to_episode(
    transitions: Iterable[tuple[Any, ...]],
    *,
    episode_id: int,
    action_vectorizer: ActionVectorizer,
) -> dict[str, list[Any]]:
    """Materialise one MineRL trajectory in SWM's per-step episode layout."""
    episode = {
        'observation': [],
        'action': [],
        'reward': [],
        'done': [],
        'episode_id': [],
        'timestep': [],
    }
    for timestep, transition in enumerate(transitions):
        if len(transition) < 5:
            raise ValueError(
                'MineRL transition must be (state, action, reward, '
                'next_state, done).'
            )
        state, action, reward, _next_state, done = transition[:5]
        if not isinstance(state, Mapping):
            raise TypeError('MineRL state must be a mapping containing `pov`.')
        episode['observation'].append(_rgb_observation(state))
        episode['action'].append(action_vectorizer(action))
        episode['reward'].append(np.float32(reward))
        episode['done'].append(bool(done))
        episode['episode_id'].append(np.int32(episode_id))
        episode['timestep'].append(np.int32(timestep))

    if not episode['observation']:
        raise ValueError(f'MineRL trajectory {episode_id} contains no steps.')
    return episode


def _make_pipeline(
    environment: str,
    data_dir: str | Path | None,
) -> MineRLDataPipeline:
    try:
        import minerl
    except ImportError as exc:
        raise ImportError(
            'MineRL conversion requires the optional dependency. Install it '
            "with `pip install 'stable-worldmodel[minerl]'`."
        ) from exc
    return minerl.data.make(environment, data_dir=data_dir, num_workers=1)


def convert_minerl(
    output_path: str | Path,
    *,
    environment: str,
    data_dir: str | Path | None = None,
    trajectory_names: Iterable[str] | None = None,
    mode: str = 'error',
    pipeline: MineRLDataPipeline | None = None,
) -> MineRLConversionSummary:
    """Convert MineRL/BASALT trajectories into a Lance-backed SWM dataset.

    Args:
        output_path: Destination Lance URI/path.
        environment: MineRL task name, e.g. ``MineRLBasaltFindCave-v0``.
        data_dir: Root containing the downloaded MineRL demonstrations.
        trajectory_names: Optional deterministic subset of trajectory names.
        mode: Lance writer mode — ``'error'`` is the safe default.
        pipeline: Optional MineRL-compatible pipeline injection, intended for
            tests and custom data sources.

    Returns:
        Conversion counts and the flattened action dimension.
    """
    source = pipeline or _make_pipeline(environment, data_dir)
    names = list(
        source.get_trajectory_names()
        if trajectory_names is None
        else trajectory_names
    )
    if not names:
        raise ValueError('No MineRL trajectories selected for conversion.')

    vectorizer = ActionVectorizer()
    transitions = 0
    destination = Path(output_path)
    with LanceWriter(destination, mode=mode) as writer:
        for episode_id, name in enumerate(names):
            episode = trajectory_to_episode(
                source.load_data(name),
                episode_id=episode_id,
                action_vectorizer=vectorizer,
            )
            transitions += len(episode['observation'])
            writer.write_episode(episode)

    return MineRLConversionSummary(
        output_path=destination,
        trajectories=len(names),
        transitions=transitions,
        action_dim=vectorizer.dim,
    )


__all__ = [
    'ActionVectorizer',
    'MineRLConversionSummary',
    'convert_minerl',
    'trajectory_to_episode',
]
