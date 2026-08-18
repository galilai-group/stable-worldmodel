"""Register all Atari Learning Environment games as gymnasium envs
and expose ``make_atari_env`` using ``ale_py.vector_env.AtariVectorEnv``.

All standard Atari preprocessing (noop reset, fire reset, episodic life,
reward clipping, frame skip, resize) is handled natively by the ALE C++
vector environment.  Emits a warning if ``ale-py`` is not installed.
"""

import warnings

try:
    import ale_py
    import gymnasium as gym

    gym.register_envs(ale_py)
except ImportError:
    warnings.warn(
        'ale-py not found; ALE/* envs are unavailable. '
        "Install with: pip install 'stable-worldmodel[env]' "
        'or pip install ale-py.',
        stacklevel=2,
    )

from stable_worldmodel.envs.ale.atari_wrappers import (
    AtariEnvAdapter,
    make_atari_env,
)

__all__ = [
    'AtariEnvAdapter',
    'make_atari_env',
]
