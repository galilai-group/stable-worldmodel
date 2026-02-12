# Adapted from https://github.com/nicklashansen/newt/blob/main/tdmpc2/envs/dmcontrol.py

from dm_control.suite import cheetah
from dm_control.utils import rewards

_CHEETAH_JUMP_HEIGHT = 1.2
_CHEETAH_LIE_HEIGHT = 0.25
_CHEETAH_SPIN_SPEED = 8


class Physics(cheetah.Physics):
    """Physics simulation with additional features for the Cheetah domain."""

    def angmomentum(self):
        """Returns the angular momentum of torso of the Cheetah about Y axis."""
        return self.named.data.subtree_angmom['torso'][1]


class CustomCheetah(cheetah.Cheetah):
    """Custom Cheetah tasks."""

    def __init__(self, goal='run-backward', move_speed=0, random=None):
        super().__init__(random)
        self._goal = goal
        self._move_speed = move_speed

    def _run_backward_reward(self, physics):
        return rewards.tolerance(
            physics.speed(),
            bounds=(-float('inf'), -self._move_speed),
            margin=self._move_speed,
            value_at_margin=0,
            sigmoid='linear',
        )

    def _stand_one_foot_reward(self, physics, foot):
        """Note: `foot` is the foot that is *not* on the ground."""
        torso_height = physics.named.data.xpos['torso', 'z']
        foot_height = physics.named.data.xpos[foot, 'z']
        height_reward = rewards.tolerance(
            (torso_height + foot_height) / 2,
            bounds=(_CHEETAH_JUMP_HEIGHT, float('inf')),
            margin=_CHEETAH_JUMP_HEIGHT / 2,
        )
        horizontal_speed_reward = rewards.tolerance(
            physics.speed(),
            bounds=(-self._move_speed, self._move_speed),
            margin=self._move_speed,
            value_at_margin=0,
            sigmoid='linear',
        )
        stand_reward = (5 * height_reward + horizontal_speed_reward) / 6
        return stand_reward

    def _stand_front_reward(self, physics):
        return self._stand_one_foot_reward(physics, 'bfoot')

    def _stand_back_reward(self, physics):
        return self._stand_one_foot_reward(physics, 'ffoot')

    def _jump_reward(self, physics):
        front_reward = self._stand_front_reward(physics)
        back_reward = self._stand_back_reward(physics)
        jump_reward = (front_reward + back_reward) / 2
        return jump_reward

    def _run_one_foot_reward(self, physics, foot):
        """Note: `foot` is the foot that is *not* on the ground."""
        torso_height = physics.named.data.xpos['torso', 'z']
        foot_height = physics.named.data.xpos[foot, 'z']
        torso_up = rewards.tolerance(
            torso_height,
            bounds=(_CHEETAH_JUMP_HEIGHT, float('inf')),
            margin=_CHEETAH_JUMP_HEIGHT / 2,
        )
        foot_up = rewards.tolerance(
            foot_height,
            bounds=(_CHEETAH_JUMP_HEIGHT, float('inf')),
            margin=_CHEETAH_JUMP_HEIGHT / 2,
        )
        up_reward = (3 * foot_up + 2 * torso_up) / 5
        if self._move_speed == 0:
            return up_reward
        horizontal_speed_reward = rewards.tolerance(
            physics.speed(),
            bounds=(self._move_speed, float('inf')),
            margin=self._move_speed,
            value_at_margin=0,
            sigmoid='linear',
        )
        return up_reward * (5 * horizontal_speed_reward + 1) / 6

    def _run_front_reward(self, physics):
        return self._run_one_foot_reward(physics, 'bfoot')

    def _run_back_reward(self, physics):
        return self._run_one_foot_reward(physics, 'ffoot')

    def _lie_down_reward(self, physics):
        torso_height = physics.named.data.xpos['torso', 'z']
        feet_height = (
            physics.named.data.xpos['ffoot', 'z']
            + physics.named.data.xpos['bfoot', 'z']
        ) / 2
        torso_down = rewards.tolerance(
            torso_height,
            bounds=(-float('inf'), _CHEETAH_LIE_HEIGHT),
            margin=_CHEETAH_LIE_HEIGHT,
            value_at_margin=0,
            sigmoid='linear',
        )
        feet_down = rewards.tolerance(
            feet_height,
            bounds=(-float('inf'), _CHEETAH_LIE_HEIGHT),
            margin=_CHEETAH_LIE_HEIGHT,
            value_at_margin=0,
            sigmoid='linear',
        )
        lie_down_reward = (3 * torso_down + feet_down) / 4
        return lie_down_reward

    def _legs_up_reward(self, physics):
        torso_height = physics.named.data.xpos['torso', 'z']
        torso_down = rewards.tolerance(
            torso_height,
            bounds=(-float('inf'), _CHEETAH_LIE_HEIGHT),
            margin=_CHEETAH_LIE_HEIGHT / 2,
        )
        get_up = self._run_one_foot_reward(physics, 'bfoot')
        legs_up_reward = (5 * torso_down + get_up) / 6
        return legs_up_reward

    def _flip_reward(self, physics, forward=True):
        spin_reward = rewards.tolerance(
            (1.0 if forward else -1.0) * physics.angmomentum(),
            bounds=(_CHEETAH_SPIN_SPEED, float('inf')),
            margin=_CHEETAH_SPIN_SPEED,
            value_at_margin=0,
            sigmoid='linear',
        )
        horizontal_speed_reward = rewards.tolerance(
            (1.0 if forward else -1.0) * physics.speed(),
            bounds=(self._move_speed, float('inf')),
            margin=self._move_speed,
            value_at_margin=0,
            sigmoid='linear',
        )
        flip_reward = (2 * spin_reward + horizontal_speed_reward) / 3
        return flip_reward

    def get_reward(self, physics):
        if self._goal == 'run':
            return super().get_reward(physics)
        elif self._goal == 'run-backward':
            return self._run_backward_reward(physics)
        elif self._goal == 'stand-front':
            return self._stand_front_reward(physics)
        elif self._goal == 'stand-back':
            return self._stand_back_reward(physics)
        elif self._goal == 'jump':
            return self._jump_reward(physics)
        elif self._goal == 'run-front':
            return self._run_front_reward(physics)
        elif self._goal == 'run-back':
            return self._run_back_reward(physics)
        elif self._goal == 'lie-down':
            return self._lie_down_reward(physics)
        elif self._goal == 'legs-up':
            return self._legs_up_reward(physics)
        elif self._goal == 'flip':
            return self._flip_reward(physics, forward=True)
        elif self._goal == 'flip-backward':
            return self._flip_reward(physics, forward=False)
        else:
            raise NotImplementedError(f'Goal {self._goal} is not implemented.')
