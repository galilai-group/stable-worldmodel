from .tasks import TASK_SPECS


__all__ = ['ManiSkillWrapper', 'TASK_SPECS']


def __getattr__(name):
    # Expose ManiSkillWrapper lazily so importing this package (e.g. to read
    # TASK_SPECS during env registration) does not pull in env.py — which
    # imports mani_skill only when actually instantiated, but should still not
    # run at `import stable_worldmodel` time.
    if name == 'ManiSkillWrapper':
        from .env import ManiSkillWrapper

        return ManiSkillWrapper
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
