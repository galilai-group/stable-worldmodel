"""TwoRoomDoorKey Navigation Environment.

Continuous analog of MiniGrid's DoorKey, implemented as a thin subclass of
:class:`stable_worldmodel.envs.two_room.env.TwoRoomEnv`:

- Agent and target start in opposite rooms separated by a wall with one or
  more door openings. Doors start locked (impassable, rendered with
  ``door.locked_color``).
- A key is randomly placed in the agent's room.
- When the agent gets within the success radius of the key (the SAME threshold
  used to reach the target, ``self._success_dist``), the key disappears, doors
  unlock (rendered with ``door.unlocked_color``), and the agent can pass
  through to reach the target.

All collision physics, wall/door rendering, and the variation-space scaffolding
come from the parent. DoorKey only overrides what genuinely differs:
variation space (adds ``key.*``, splits ``door.color`` into locked/unlocked,
drops ``task.min_steps``, adds a constraint on ``target.position``); the four
subclass hooks (``_door_is_passable``, ``_door_color_for_index``,
``_render_extras``, ``_after_step_state_update``); and observation/info to
expose key state.
"""

from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces
from stable_worldmodel import spaces as swm_spaces
from stable_worldmodel.envs.two_room.env import TwoRoomEnv

DEFAULT_VARIATIONS = (
    'wall.axis',
    'door.number',
    'door.position',
    'agent.position',
    'key.position',
    'target.position',
    'agent.color',
    'key.color',
    'target.color',
    'door.locked_color',
    'door.unlocked_color',
)


class TwoRoomDoorKeyEnv(TwoRoomEnv):
    def __init__(
        self,
        render_mode: str = 'rgb_array',
        render_target: bool = False,
        init_value: dict | None = None,
        img_size: int = TwoRoomEnv.DEFAULT_IMG_SIZE,
    ):
        # Initialize DoorKey-specific runtime state BEFORE super().__init__()
        # so the parent's bring-up (which may invoke `_door_is_passable()` or
        # `_render_extras()` indirectly during early construction) sees a
        # valid `has_key` / `key_position`.
        self.has_key = False
        self.key_position = torch.zeros(2, dtype=torch.float32)

        super().__init__(
            render_mode=render_mode,
            render_target=render_target,
            init_value=init_value,
            img_size=img_size,
        )
        self.env_name = 'TwoRoomDoorKey'

        # Extend observation space: parent's 10 + key_xy(2) + has_key(1) = 13.
        state_dim = 2 + 2 + self.MAX_DOOR * 2 + 2 + 1
        self.observation_space = spaces.Box(
            low=0,
            high=self.IMG_SIZE,
            shape=(state_dim,),
            dtype=np.float32,
        )

    # Variation Space

    def _build_variation_space(self):
        parent = super()._build_variation_space()
        pos_min = float(self.BORDER_SIZE)
        pos_max = float(self.IMG_SIZE - self.BORDER_SIZE - 1)

        # Key render size scales with img_size via self._scale so the key looks
        # the same at any resolution (matches the parent's agent/target radius
        # bounds, incl. the scaled lower floor). Key PICKUP is NOT a separate
        # tunable radius: it uses the env success radius (self._success_dist),
        # the same threshold as reaching the target — see _after_step_state_update.
        r_def = self._default_dot_std
        r_lo = max(1.0 * self._scale, 0.5 * r_def)
        r_hi = 3.0 * r_def
        # Key default in agent's room — agent default is (W*0.6, W*0.6), so
        # place the key a bit further from the wall on both axes. Valid for
        # both wall orientations.
        key_def_v = max(pos_min + 1, self.WALL_CENTER * 0.4)

        key_dict = swm_spaces.Dict(
            {
                'color': swm_spaces.RGBBox(
                    init_value=np.array([255, 215, 0], dtype=np.uint8)
                ),
                'radius': swm_spaces.Box(
                    low=np.array([r_lo], dtype=np.float32),
                    high=np.array([r_hi], dtype=np.float32),
                    init_value=np.array([r_def], dtype=np.float32),
                    shape=(1,),
                    dtype=np.float32,
                ),
                'position': swm_spaces.Box(
                    low=np.array([pos_min, pos_min], dtype=np.float32),
                    high=np.array([pos_max, pos_max], dtype=np.float32),
                    shape=(2,),
                    dtype=np.float32,
                    init_value=np.array(
                        [key_def_v, key_def_v], dtype=np.float32
                    ),
                    constrain_fn=self._constrain_key_in_agent_room,
                ),
            },
            sampling_order=['color', 'radius', 'position'],
        )

        # Door: reuse parent's number/size/position by reference; swap the
        # single `color` for a locked/unlocked pair.
        parent_door = parent['door']
        door_dict = swm_spaces.Dict(
            {
                'locked_color': swm_spaces.RGBBox(
                    init_value=np.array([200, 0, 0], dtype=np.uint8)
                ),
                'unlocked_color': swm_spaces.RGBBox(
                    init_value=np.array([255, 255, 255], dtype=np.uint8)
                ),
                'number': parent_door['number'],
                'size': parent_door['size'],
                'position': parent_door['position'],
            },
            sampling_order=[
                'locked_color',
                'unlocked_color',
                'number',
                'size',
                'position',
            ],
        )

        # Target: parent has no active constrain_fn; DoorKey requires the
        # target to be in the opposite room from the agent. Use the parent's
        # scaled default position so the geometry stays scale-invariant.
        parent_target = parent['target']
        parent_target_init = parent_target['position'].init_value
        target_dict = swm_spaces.Dict(
            {
                'color': parent_target['color'],
                'radius': parent_target['radius'],
                'position': swm_spaces.Box(
                    low=np.array([pos_min, pos_min], dtype=np.float32),
                    high=np.array([pos_max, pos_max], dtype=np.float32),
                    shape=(2,),
                    dtype=np.float32,
                    init_value=np.array(parent_target_init, dtype=np.float32),
                    constrain_fn=self._constrain_target_in_other_room,
                ),
            },
            sampling_order=['color', 'radius', 'position'],
        )

        return swm_spaces.Dict(
            {
                'agent': parent['agent'],
                'key': key_dict,
                'target': target_dict,
                'wall': parent['wall'],
                'door': door_dict,
                'background': parent['background'],
                'rendering': parent['rendering'],
                # 'task' (min_steps) intentionally dropped — not used in DoorKey.
            },
            sampling_order=[
                'background',
                'wall',
                'agent',
                'key',
                'door',
                'target',
                'rendering',
            ],
        )

    # Gym API

    def reset(self, seed=None, options=None):
        options = dict(options) if options else {}
        # Use DoorKey's broader default-resample set when caller didn't specify.
        if 'variation' not in options:
            options['variation'] = DEFAULT_VARIATIONS

        key_state = options.get('key_state', None)

        # Reset has_key BEFORE super so the parent's cache refresh sees doors
        # locked (its `door_passable_cache = self._door_is_passable()` reads
        # `self.has_key`).
        self.has_key = False

        # Parent's reset: samples variation_space, sets agent/target positions,
        # caches params, refreshes door_passable_cache, renders _target_img,
        # returns (obs, info).
        super().reset(seed=seed, options=options)

        # Wire up key position from options or from the freshly sampled space.
        if key_state is not None:
            self.key_position = torch.as_tensor(key_state, dtype=torch.float32)
        else:
            self.key_position = torch.as_tensor(
                self.variation_space['key']['position'].value,
                dtype=torch.float32,
            )

        # Re-render the target image now that the key position is known
        # (parent's render ran before key_position was set, so the key would
        # have been drawn at the stale/zero location).
        self._target_img = self._render_frame(agent_pos=self.target_position)

        obs = self._get_obs()
        info = self._get_info()
        info['distance_to_target'] = float(
            torch.norm(self.agent_position - self.target_position)
        )
        return obs, info

    # Hooks for subclasses

    def _door_is_passable(self) -> torch.Tensor:
        return torch.full(
            (self.MAX_DOOR,), bool(self.has_key), dtype=torch.bool
        )

    def _door_color_for_index(self, i: int) -> np.ndarray:
        space = self.variation_space['door']
        return (
            space['unlocked_color'].value
            if self.has_key
            else space['locked_color'].value
        )

    def _render_extras(
        self, img: torch.Tensor, not_wall_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.has_key:
            return img
        key_color = self.variation_space['key']['color'].value
        key_r = float(self.variation_space['key']['radius'].value.item())
        key_dot = self._gaussian_dot(self.key_position, key_r)
        if not_wall_mask is not None:
            key_dot = key_dot * not_wall_mask
        return self._alpha_blend(img, key_dot, key_color)

    def _after_step_state_update(
        self, pre_pos: torch.Tensor, post_pos: torch.Tensor
    ):
        if self.has_key:
            return
        # Pick the key up using the SAME radius as reaching the target
        # (self._success_dist = _REF_SUCCESS_DIST * scale) — scale-consistent,
        # not a fixed pixel count and not a separate tunable.
        pickup_r = self._success_dist
        if float(torch.norm(post_pos - self.key_position)) < pickup_r:
            self.has_key = True
            # Refresh the cache so the NEXT step's physics treats doors as
            # passable. (This step's collision was computed with the doors
            # still locked, which is correct.)
            self.door_passable_cache = self._door_is_passable()

    # Observation / info

    def _get_obs(self):
        base = super()._get_obs()  # (10,)
        extra = torch.tensor(
            [
                float(self.key_position[0]),
                float(self.key_position[1]),
                float(self.has_key),
            ],
            dtype=torch.float32,
        )
        return torch.cat([base, extra])  # (13,)

    def _get_info(self):
        info = super()._get_info()
        info['key_position'] = self.key_position.detach().cpu().numpy()
        info['has_key'] = np.float32(self.has_key)
        return info

    # Constraints

    def _constrain_key_in_agent_room(self, key_pos):
        """Key must be in the agent's room and outside the wall zone."""
        agent_pos = self.variation_space['agent']['position'].value
        wall_axis = int(self.variation_space['wall']['axis'].value)
        wall_thickness = int(self.variation_space['wall']['thickness'].value)
        half_thickness = wall_thickness // 2

        # Key is a point (pickup is a center-distance test): exclude only the
        # actual wall slab, no radius padding.
        wall_min = self.WALL_CENTER - half_thickness
        wall_max = self.WALL_CENTER + half_thickness

        if wall_axis == 1:  # vertical wall — rooms split on x
            agent_side = agent_pos[0] < self.WALL_CENTER
            key_side = key_pos[0] < self.WALL_CENTER
            if agent_side != key_side:
                return False
            if wall_min <= key_pos[0] <= wall_max:
                return False
        else:  # horizontal wall — rooms split on y
            agent_side = agent_pos[1] < self.WALL_CENTER
            key_side = key_pos[1] < self.WALL_CENTER
            if agent_side != key_side:
                return False
            if wall_min <= key_pos[1] <= wall_max:
                return False
        return True

    def _constrain_target_in_other_room(self, target_pos):
        """Target must be in the room opposite the agent and outside the
        wall zone. DoorKey only makes sense as a cross-room task."""
        agent_pos = self.variation_space['agent']['position'].value
        wall_axis = int(self.variation_space['wall']['axis'].value)
        wall_thickness = int(self.variation_space['wall']['thickness'].value)
        half_thickness = wall_thickness // 2

        # Target is a point (success is a center-distance test): exclude only
        # the actual wall slab, no radius padding.
        wall_min = self.WALL_CENTER - half_thickness
        wall_max = self.WALL_CENTER + half_thickness

        if wall_axis == 1:  # vertical wall
            agent_side = agent_pos[0] < self.WALL_CENTER
            target_side = target_pos[0] < self.WALL_CENTER
            if agent_side == target_side:
                return False
            if wall_min <= target_pos[0] <= wall_max:
                return False
        else:  # horizontal wall
            agent_side = agent_pos[1] < self.WALL_CENTER
            target_side = target_pos[1] < self.WALL_CENTER
            if agent_side == target_side:
                return False
            if wall_min <= target_pos[1] <= wall_max:
                return False
        return True

    # Convenience setters

    def _set_key_state(self, key_state):
        self.key_position = torch.tensor(key_state, dtype=torch.float32)
        self.variation_space['key']['position'].set_value(
            np.array(key_state, dtype=np.float32)
        )

    def _set_has_key(self, has_key):
        self.has_key = bool(has_key)
        # Refresh the cache so subsequent physics sees the new lock state.
        self.door_passable_cache = self._door_is_passable()
