# Adapted from https://github.com/nicklashansen/newt/blob/main/tdmpc2/envs/dmcontrol.py
from dm_control.suite import hopper
from dm_control.utils import rewards


# Minimal height of torso over foot above which stand reward is 1.
_STAND_HEIGHT = 0.6

# Hopping speed above which hop reward is 1.
_HOP_SPEED = 2

# Angular momentum above which reward is 1.
_SPIN_SPEED = 5


class Physics(hopper.Physics):
    def angmomentum(self):
        """Returns the angular momentum of torso of the Cheetah about Y axis."""
        return self.named.data.subtree_angmom['torso'][1]


class CustomHopper(hopper.Hopper):
    """Custom Hopper tasks."""

    def __init__(self, goal='hop-backward', random=None):
        hopping = goal == 'hop'
        super().__init__(hopping, random)
        self._goal = goal

    def _hop_backward_reward(self, physics):
        standing = rewards.tolerance(physics.height(), (_STAND_HEIGHT, 2))
        hopping = rewards.tolerance(
            physics.speed(),
            bounds=(-float('inf'), -_HOP_SPEED / 2),
            margin=_HOP_SPEED / 4,
            value_at_margin=0.5,
            sigmoid='linear',
        )
        return standing * hopping

    def _flip_reward(self, physics, forward=True):
        reward = rewards.tolerance(
            (1.0 if forward else -1.0) * physics.angmomentum(),
            bounds=(_SPIN_SPEED, float('inf')),
            margin=_SPIN_SPEED / 2,
            value_at_margin=0,
            sigmoid='linear',
        )
        return reward

    def get_reward(self, physics):
        if self._goal in ('stand', 'hop'):
            return super().get_reward(physics)
        elif self._goal == 'hop-backward':
            return self._hop_backward_reward(physics)
        elif self._goal == 'flip':
            return self._flip_reward(physics, forward=True)
        elif self._goal == 'flip-backward':
            return self._flip_reward(physics, forward=False)
        else:
            raise NotImplementedError(f'Goal {self._goal} is not implemented.')
