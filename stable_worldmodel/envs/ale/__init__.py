"""Register all Atari Learning Environment games as gymnasium envs.

Importing this subpackage exposes the standard ``ALE/<Game>-v5`` ids via
``gym.make``. Emits a warning if ``ale-py`` is not installed.
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
