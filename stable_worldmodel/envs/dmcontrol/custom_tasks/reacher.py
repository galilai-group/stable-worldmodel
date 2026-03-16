import numpy as np
from dm_control.suite import reacher

_DEFAULT_QPOS_THRESHOLD = 0.05


class ReacherQPosMatchTask(reacher.Reacher):
    """Reacher task that terminates (success) when qpos matches a target qpos.

    The target qpos is set externally via the env's `set_target_qpos()`. Until
    it is set, the task never terminates early.

    Args:
        target_size: Tolerance radius for the standard finger-to-target reward.
        random: Random seed or RandomState.
        qpos_threshold: Per-joint absolute tolerance (in radians) used to
            determine whether the current qpos matches the target qpos.
    """

    def __init__(
        self, target_size, random=None, qpos_threshold=_DEFAULT_QPOS_THRESHOLD
    ):
        super().__init__(target_size=target_size, random=random)
        self.target_qpos = None
        self.qpos_threshold = qpos_threshold

    def get_termination(self, physics):
        if self.target_qpos is None:
            return None
        diff = np.abs(physics.data.qpos - self.target_qpos)
        if np.all(diff < self.qpos_threshold):
            return 0.0
        return None
