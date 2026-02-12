import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw

from stable_worldmodel import spaces as swm_spaces
from stable_worldmodel.envs.pinpad.constants import (
    COLORS,
    X_BOUND,
    Y_BOUND,
    RENDER_SCALE,
    TASK_NAMES,
    LAYOUTS,
)


DEFAULT_VARIATIONS = (
    'agent.spawn',
    'agent.target_pad',
)


# TODO: Re-enable targets to be sequences of pads instead of single pads
class PinPad(gym.Env):
    def __init__(
        self,
        seed=None,
        init_value=None,
        render_mode='rgb_array',  # For backward compatibility; not used
    ):
        # Build variation space
        self.variation_space = self._build_variation_space()
        if init_value is not None:
            self.variation_space.set_init_value(init_value)

        # To be initialized in reset()
        self.task = None
        self.layout = None
        self.pads = None
        self.player = None
        self.target_pad = None

    def _build_variation_space(self):
        return swm_spaces.Dict(
            {
                'agent': swm_spaces.Dict(
                    {
                        'spawn': swm_spaces.Box(
                            low=np.array([1.5, 1.5], dtype=np.float64),
                            high=np.array(
                                [X_BOUND - 1.5, Y_BOUND - 1.5],
                                dtype=np.float64,
                            ),
                            init_value=np.array(
                                [X_BOUND / 2, Y_BOUND / 2], dtype=np.float64
                            ),
                            shape=(2,),
                            dtype=np.float64,
                        ),
                        # The number of pads is dynamic based on the task,
                        # so we generate the index as a float in [0, 1) and then
                        # scale it to the number of pads before truncating it to an int
                        'target_pad': swm_spaces.Box(
                            low=0.0,
                            high=1.0,
                            init_value=0.0,
                            shape=(),
                            dtype=np.float64,
                        ),
                    }
                ),
                'grid': swm_spaces.Dict(
                    {
                        'task': swm_spaces.Discrete(
                            n=len(TASK_NAMES),
                            start=0,
                            init_value=0,  # 0 = 'three', 5 = 'eight'
                        ),
                    }
                ),
            },
            sampling_order=['grid', 'agent'],
        )

    def _setup_layout(self, task):
        layout = LAYOUTS[task]
        self.layout = np.array(
            [list(line) for line in layout.split('\n')]
        ).T  # Transposes so that actions are (dx, dy)
        assert self.layout.shape == (X_BOUND, Y_BOUND), (
            f'Layout shape should be ({X_BOUND}, {Y_BOUND}), got {self.layout.shape}'
        )

    def _setup_pads(self):
        self.pads = sorted(
            list(set(self.layout.flatten().tolist()) - set('* #\n'))
        )

    @property
    def action_space(self):
        return gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float64,
        )

    @property
    def observation_space(self):
        return gym.spaces.Box(
            low=0,
            high=255,
            shape=(Y_BOUND * RENDER_SCALE, X_BOUND * RENDER_SCALE, 3),
            dtype=np.uint8,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        # Reset variation space
        options = options or {}
        swm_spaces.reset_variation_space(
            self.variation_space,
            seed,
            options,
            DEFAULT_VARIATIONS,
        )

        # Update task if it changed or if this is the first reset
        task_idx = int(self.variation_space['grid']['task'].value)
        new_task = TASK_NAMES[task_idx]
        if new_task != self.task or self.task is None:
            self.task = new_task
            self._setup_layout(self.task)
            self._setup_pads()

        # Set player position directly from variation space
        spawn_position = self.variation_space['agent']['spawn'].value
        self.player = tuple(spawn_position)

        # Set target pad from variation space using linear binning
        target_pad_value = float(
            self.variation_space['agent']['target_pad'].value
        )
        target_pad_idx = int(target_pad_value * len(self.pads))
        assert target_pad_idx >= 0 and target_pad_idx < len(self.pads), (
            f'Target pad index {target_pad_idx} is out of range for {len(self.pads)} pads'
        )
        self.target_pad = self.pads[target_pad_idx]
        self.target_position = self._get_target_position(self.target_pad)
        self.goal = self.render(player_position=self.target_position)

        # Gets return values
        obs = self.render()
        info = self._get_info()
        return obs, info

    def step(self, action):
        # Moves player
        x = np.clip(self.player[0] + action[0], 1.5, X_BOUND - 1.5)
        y = np.clip(self.player[1] + action[1], 1.5, Y_BOUND - 1.5)
        tile = self.layout[int(x)][int(y)]
        if (
            tile != '#'
        ):  # TODO: Add linear interpolation in case of wall collision
            self.player = (float(x), float(y))

        # Makes observation
        agent_in_target_pad = self._agent_in_target_pad(
            self.player, self.target_pad
        )
        obs = self.render()
        reward = 10.0 if agent_in_target_pad else 0.0
        terminated = agent_in_target_pad  # TODO: Maybe always set to false?
        truncated = False
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def _get_target_position(self, target_pad):
        target_cells = np.array(
            list(zip(*np.where(self.layout == target_pad))), dtype=np.float64
        )
        target_cell_centers = target_cells + 0.5
        center_cell = np.array([X_BOUND / 2, Y_BOUND / 2], dtype=np.float64)
        farthest_idx = np.argmax(
            np.linalg.norm(target_cell_centers - center_cell, axis=1)
        )
        farthest_from_center = target_cell_centers[farthest_idx]
        return farthest_from_center

    def _agent_in_target_pad(self, player, target_pad):
        # Gets all cells that overlap with the agent
        corner_deltas = np.array(
            [
                (-0.5, -0.5),
                (-0.5, 0.5),
                (0.5, -0.5),
                (0.5, 0.5),
            ],
            dtype=np.float64,
        )
        corner_positions = player + corner_deltas
        distinct_corner_positions = [
            tuple(pos)
            for pos in np.unique(corner_positions.astype(int), axis=0)
        ]

        # Gets all cells from the target pad
        target_cells = np.array(
            list(zip(*np.where(self.layout == target_pad))), dtype=np.float64
        )
        target_cells = [tuple(pos) for pos in target_cells.astype(int)]

        # Checks that the agent is entirely within the target pad
        for pos in distinct_corner_positions:
            if pos not in target_cells:
                return False
        return True

    def _get_info(self):
        info = {
            'agent_position': np.array(self.player),
            'target_position': np.array(self.target_position),
            'goal': self.goal,
        }
        return info

    def render(self, player_position=None):
        # Sets up grid
        grid = np.zeros((X_BOUND, Y_BOUND, 3), np.uint8) + 255
        white = np.array([255, 255, 255])
        if player_position is None:
            player_position = self.player

        # Colors all cells except agent
        for (x, y), char in np.ndenumerate(self.layout):
            if char == '#':
                grid[x, y] = (192, 192, 192)  # Gray
            elif char in self.pads:
                color = np.array(COLORS[char])
                color = (
                    color
                    if self._agent_in_target_pad(player_position, char)
                    else (10 * color + 90 * white) / 100
                )
                grid[x, y] = color

        # Scales up and transposes grid
        image = np.repeat(np.repeat(grid, RENDER_SCALE, 0), RENDER_SCALE, 1)
        image = image.transpose((1, 0, 2))

        # Places agent with anti-aliasing
        image_pil = Image.fromarray(image, mode='RGB')
        draw = ImageDraw.Draw(image_pil)
        x, y = player_position
        draw.rectangle(
            [
                (x - 0.5) * RENDER_SCALE,
                (y - 0.5) * RENDER_SCALE,
                (x + 0.5) * RENDER_SCALE,
                (y + 0.5) * RENDER_SCALE,
            ],
            fill=(0, 0, 0),  # Agent is black
        )
        image = np.asarray(image_pil)
        return image
