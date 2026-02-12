import collections

import gymnasium as gym
import numpy as np

from loguru import logger as logging
import cv2

from stable_worldmodel import spaces as swm_spaces


DEFAULT_VARIATIONS = (
    'agent.spawn',
    'agent.target_pad',
)


# TODO: Re-enable targets to be sequences of pads instead of single pads
class PinPadDiscrete(gym.Env):

    COLORS = {
        '1': (255,   0,   0),
        '2': (  0, 255,   0),
        '3': (  0,   0, 255),
        '4': (255, 255,   0),
        '5': (255,   0, 255),
        '6': (  0, 255, 255),
        '7': (128,   0, 128),
        '8': (  0, 128, 128),
    }
    X_BOUND = 16
    Y_BOUND = 16
    RENDER_SCALE = 14

    def __init__(self, seed=None, init_value=None):
        # Task mapping for Discrete space
        self.task_names = ['three', 'four', 'five', 'six', 'seven', 'eight']
        
        # Build variation space
        self.variation_space = self._build_variation_space()
        if init_value is not None:
            self.variation_space.set_init_value(init_value)

        # To be initialized in reset()
        self.task = None
        self.layout = None
        self.pads = None
        self.spawns = None
        self.player = None
        self.target_pad = None

    def _build_variation_space(self):
        # Spawn locations don't include walls
        max_spawns = self.X_BOUND * self.Y_BOUND - 2 * (self.X_BOUND + self.Y_BOUND - 2)
        
        return swm_spaces.Dict(
            {
                'agent': swm_spaces.Dict(
                    {
                        'spawn': swm_spaces.Discrete(
                            n=max_spawns,
                            start=0,
                            init_value=0,
                        ),
                        # The number of pads is dynamic based on the task,
                        # so we generate the index as a float in [0, 1) and then
                        # scale it to the number of pads before truncating it to an int
                        'target_pad': swm_spaces.Box(
                            low=0.0,
                            high=1.0,
                            init_value=0.0,
                            shape=(),
                            dtype=np.float32,
                        ),
                    }
                ),
                'grid': swm_spaces.Dict(
                    {
                        'task': swm_spaces.Discrete(
                            n=len(self.task_names),
                            start=0,
                            init_value=0,  # 0 = 'three', 5 = 'eight'
                        ),
                    }
                ),
            },
            sampling_order=['grid', 'agent'],
        )

    def _setup_layout(self, task):
        layout = {
            'three': LAYOUT_THREE,
            'four': LAYOUT_FOUR,
            'five': LAYOUT_FIVE,
            'six': LAYOUT_SIX,
            'seven': LAYOUT_SEVEN,
            'eight': LAYOUT_EIGHT,
        }[task]
        self.layout = np.array([list(line) for line in layout.split('\n')]).T  # Transposes so that actions are (dx, dy)
        assert self.layout.shape == (self.X_BOUND, self.Y_BOUND), (
            f"Layout shape should be ({self.X_BOUND}, {self.Y_BOUND}), got {self.layout.shape}"
        )

    def _setup_pads_and_spawns(self):
        self.pads = sorted(list(set(self.layout.flatten().tolist()) - set('* #\n')))
        self.spawns = []
        for (x, y), char in np.ndenumerate(self.layout):
            if char != '#':
                self.spawns.append((x, y))

    @property
    def action_space(self):
        return gym.spaces.Discrete(5)  # [0, 5)

    @property
    def observation_space(self):
        return gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.X_BOUND * self.RENDER_SCALE, self.Y_BOUND * self.RENDER_SCALE, 3),
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
        new_task = self.task_names[task_idx]
        if new_task != self.task or self.task is None:
            self.task = new_task
            self._setup_layout(self.task)
            self._setup_pads_and_spawns()
        
        # Set player position from variation space (index into spawns)
        spawn_idx = int(self.variation_space['agent']['spawn'].value)
        assert spawn_idx >= 0 and spawn_idx < len(self.spawns), (
            f"Spawn index {spawn_idx} is out of range for {len(self.spawns)} spawns"
        )
        self.player = self.spawns[spawn_idx]
        
        # Set target pad from variation space using linear binning
        target_pad_value = float(self.variation_space['agent']['target_pad'].value)
        target_pad_idx = int(target_pad_value * len(self.pads))
        assert target_pad_idx >= 0 and target_pad_idx < len(self.pads), (
            f"Target pad index {target_pad_idx} is out of range for {len(self.pads)} pads"
        )
        self.target_pad = self.pads[target_pad_idx]
        self.goal = self._get_goal(self.target_pad)

        # Gets return values
        obs = self.render()
        info = {'agent_position': np.array(self.player), 'goal': self.goal}
        return obs, info

    def step(self, action):
        # Moves player
        move = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)][action]
        x = np.clip(self.player[0] + move[0], 0, self.X_BOUND - 1)
        y = np.clip(self.player[1] + move[1], 0, self.Y_BOUND - 1)
        tile = self.layout[x][y]
        if tile != '#':
            self.player = (x, y)

        # Gets reward
        reward = 0.0
        if tile == self.target_pad:
            reward += 10.0

        # Makes observation
        obs = self.render()
        terminated = tile == self.target_pad
        truncated = False
        info = {'agent_position': np.array(self.player), 'goal': self.goal}
        return obs, reward, terminated, truncated, info

    def _get_goal(self, target_pad):
        target_cells = list(zip(*np.where(self.layout == target_pad)))
        center_cell = (self.X_BOUND // 2, self.Y_BOUND // 2)
        farthest_idx = np.argmax(
            np.linalg.norm(np.array(target_cells) - np.array(center_cell), axis=1)
        )
        farthest_from_center = target_cells[farthest_idx]
        return self.render(player_position=farthest_from_center)

    def render(self, player_position=None):
        # Sets up grid
        grid = np.zeros((self.X_BOUND, self.Y_BOUND, 3), np.uint8) + 255
        white = np.array([255, 255, 255])
        if player_position is None:
            player_position = self.player
        current = self.layout[player_position[0]][player_position[1]]

        # Colors all cells
        for (x, y), char in np.ndenumerate(self.layout):
            if char == '#':
                grid[x, y] = (192, 192, 192)  # Gray
            elif char in self.pads:
                color = np.array(self.COLORS[char])
                color = color if char == current else (10 * color + 90 * white) / 100
                grid[x, y] = color
        grid[player_position] = (0, 0, 0)

        # Scales up
        image = np.repeat(np.repeat(grid, self.RENDER_SCALE, 0), self.RENDER_SCALE, 1)
        return image.transpose((1, 0, 2))


LAYOUT_THREE = """
################
#1111      3333#
#1111      3333#
#1111      3333#
#1111      3333#
#              #
#              #
#              #
#              #
#              #
#              #
#     2222     #
#     2222     #
#     2222     #
#     2222     #
################
""".strip('\n')

LAYOUT_FOUR = """
################
#1111      4444#
#1111      4444#
#1111      4444#
#1111      4444#
#              #
#              #
#              #
#              #
#              #
#              #
#3333      2222#
#3333      2222#
#3333      2222#
#3333      2222#
################
""".strip('\n')

LAYOUT_FIVE = """
################
#          4444#
#          4444#
#111       4444#
#111           #
#111           #
#111        555#
#           555#
#           555#
#333        555#
#333           #
#333           #
#333       2222#
#          2222#
#          2222#
################
""".strip('\n')

LAYOUT_SIX = """
################
#111        555#
#111        555#
#111        555#
#              #
#              #
#33          66#
#33          66#
#33          66#
#33          66#
#              #
#              #
#444        222#
#444        222#
#444        222#
################
""".strip('\n')

LAYOUT_SEVEN = """
################
#111        444#
#111        444#
#11          44#
#              #
#              #
#33          55#
#33          55#
#33          55#
#33          55#
#              #
#              #
#66          22#
#666  7777  222#
#666  7777  222#
################
""".strip('\n')

LAYOUT_EIGHT = """
################
#111  8888  444#
#111  8888  444#
#11          44#
#              #
#              #
#33          55#
#33          55#
#33          55#
#33          55#
#              #
#              #
#66          22#
#666  7777  222#
#666  7777  222#
################
""".strip('\n')
