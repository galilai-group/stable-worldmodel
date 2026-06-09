"""Register all Atari Learning Environment games as gymnasium envs
and expose Atari 100k protocol wrappers.

Importing this subpackage exposes the standard ``ALE/<Game>-v5`` ids via
``gym.make`` and the Atari 100k wrappers (``NoopResetEnv``,
``EpisodicLifeEnv``, ``FireResetEnv``, ``ClipRewardEnv``) together with
the ``make_atari_env`` factory.  Emits a warning if ``ale-py`` is not
installed.
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
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    NoopResetEnv,
    make_atari_env,
)

__all__ = [
    'ClipRewardEnv',
    'EpisodicLifeEnv',
    'FireResetEnv',
    'NoopResetEnv',
    'make_atari_env',
]
