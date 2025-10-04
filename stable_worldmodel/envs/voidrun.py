from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces

import stable_worldmodel as swm


DEFAULT_VARIATIONS = ("board.prob_gravel", "agent.position", "goal.position")


@dataclass(frozen=True)
class Action:
    LEFT: int = 0
    RIGHT: int = 1
    DOWN: int = 2
    UP: int = 3


class VoidRunEnv(gym.Env):
    """Discrete grid environment with a 1x1 agent cell."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        seed: int | None = None,
        render_mode: str = "human",
    ) -> None:
        super().__init__()

        self.render_mode = render_mode

        self.max_size = 50
        self.step_size = 1  # fixed for 1x1 agent

        self._rng = np.random.default_rng(seed)
        self._fig = None
        self._ax = None
        self._goal = None

        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(low=0, high=3, shape=(self.max_size, self.max_size), dtype=np.int8),
                "perception": spaces.MultiDiscrete([self.max_size, self.max_size]),
            }
        )

        # Variation space without radius; agent is always 1x1
        self.variation_space = swm.spaces.Dict(
            {
                "agent": swm.spaces.Dict(
                    {
                        "color": swm.spaces.RGBBox(
                            init_value=np.array([255, 0, 0], dtype=np.uint8),
                        ),
                        "position": swm.spaces.MultiDiscrete(
                            [self.max_size, self.max_size],
                            init_value=np.array([10, 10], dtype=np.int32),
                            constrain_fn=self.check_location,
                        ),
                        "prob_break": swm.spaces.Box(
                            low=np.array(0.5, dtype=np.float32),
                            high=np.array(1.0, dtype=np.float32),
                            init_value=np.array(1.0, dtype=np.float32),
                            dtype=np.float32,
                        ),
                    },
                    sampling_order=["color", "position", "prob_break"],
                ),
                "goal": swm.spaces.Dict(
                    {
                        "color": swm.spaces.RGBBox(
                            init_value=np.array([52, 235, 201], dtype=np.uint8),
                        ),
                        "position": swm.spaces.MultiDiscrete(
                            [self.max_size, self.max_size],
                            init_value=[5, 5],
                            constrain_fn=self.check_location,
                        ),
                    },
                    sampling_order=["color", "position"],
                ),
                "board": swm.spaces.Dict(
                    {
                        "size": swm.spaces.Discrete(self.max_size - 10, start=10, init_value=20),
                        "prob_gravel": swm.spaces.Box(
                            low=np.array(0.0, dtype=np.float32),
                            high=np.array(1.0, dtype=np.float32),
                            init_value=np.array(0.45, dtype=np.float32),
                            dtype=np.float32,
                        ),
                        "prob_break": swm.spaces.Box(
                            low=np.array(0.5, dtype=np.float32),
                            high=np.array(1.0, dtype=np.float32),
                            init_value=np.array(1.0, dtype=np.float32),
                            dtype=np.float32,
                        ),
                        "sand_color": swm.spaces.RGBBox(
                            init_value=np.array([242, 218, 130], dtype=np.uint8),
                        ),
                        "gravel_color": swm.spaces.RGBBox(
                            init_value=np.array([128, 128, 128], dtype=np.uint8),
                        ),
                        "void_color": swm.spaces.RGBBox(
                            init_value=np.array([0, 0, 0], dtype=np.uint8),
                        ),
                    },
                    sampling_order=[
                        "size",
                        "prob_gravel",
                        "sand_color",
                        "gravel_color",
                        "void_color",
                    ],
                ),
            },
            sampling_order=["board", "agent", "goal"],
        )

    # -------------------- Core API ----------

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.observation_space.seed(seed)
        self.action_space.seed(seed)

        if hasattr(self, "variation_space"):
            self.variation_space.seed(seed)

        options = options or {}

        self.variation_space.reset()

        if "variation" not in options:
            options["variation"] = DEFAULT_VARIATIONS
        elif not isinstance(options["variation"], Sequence):
            raise ValueError("variation option must be a Sequence containing variations names to sample")

        self.variation_space.update(options["variation"])

        assert self.variation_space.check(debug=True), "Variation values must be within variation space!"

        self._reset_state()

        obs = self._get_obs()
        info = {
            "newly_voided": 0,
            "in_void": False,
            "goal": self._goal,
            "steps": self.steps,
            "goal_pos": self.goal_pos,
        }

        return obs, info

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError("Invalid action")

        old_r, old_c = self.player_row, self.player_col

        dr, dc = 0, 0
        if action == Action.LEFT:
            dc = -self.step_size
        elif action == Action.RIGHT:
            dc = +self.step_size
        elif action == Action.DOWN:
            dr = +self.step_size
        elif action == Action.UP:
            dr = -self.step_size

        size = self.variation_space["board"]["size"].value
        # 1x1 agent can occupy any cell within [0, size-1]
        new_r = int(np.clip(old_r + dr, 0, size - 1))
        new_c = int(np.clip(old_c + dc, 0, size - 1))

        # Move the agent
        self.player_row, self.player_col = new_r, new_c
        self.player_y = self.player_row + 0.5
        self.player_x = self.player_col + 0.5

        # Void the cell the agent just left (like scraping behind)
        newly_voided = self._void_cell(old_r, old_c)

        #

        in_void = self.board[new_r, new_c] == 0

        reward = float(newly_voided)
        self.steps += 1

        terminated = self.check_termination()
        truncated = bool(in_void)

        obs = self._get_obs()
        info = {
            "newly_voided": newly_voided,
            "in_void": bool(in_void),
            "goal": self._goal,
            "steps": self.steps,
            "goal_pos": self.goal_pos,
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, bool(terminated), bool(truncated), info

    def render(self, mode: str | None = None):
        mode = mode or self.render_mode or "human"
        size = self.variation_space["board"]["size"].value
        if self._fig is None or self._ax is None:
            self._fig, self._ax = plt.subplots(figsize=(size * 0.4, size * 0.4))

        ax = self._ax
        ax.clear()

        self.render_board(ax=ax)

        # Draw 1x1 agent as a square
        if self.board[self.player_row, self.player_col] > 0:
            rect = plt.Rectangle(
                (self.player_x - 0.5, self.player_y - 0.5),
                1.0,
                1.0,
                fill=True,
                facecolor=self.variation_space["agent"]["color"].value / 255.0,
                edgecolor=None,
                zorder=3,
                antialiased=False,
            )
            ax.add_patch(rect)

        ax.set_xlim(0, size)
        ax.set_ylim(0, size)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor((1.0, 1.0, 1.0))

        self._fig.tight_layout(pad=0)

        if mode == "human":
            plt.pause(0.001)
            plt.draw()
            return None
        if mode == "rgb_array":
            self._fig.canvas.draw()
            h, w = self._fig.canvas.get_width_height()
            buf = np.frombuffer(self._fig.canvas.buffer_rgba(), dtype=np.uint8)
            buf = buf.reshape(h, w, 4)[..., :3]
            return buf.copy()
        raise NotImplementedError(f"Render mode {mode} not supported.")

    def close(self) -> None:
        if self._fig is not None:
            plt.close(self._fig)
            self._fig, self._ax = None, None

    # -------------------- Helpers --------------------

    def _reset_state(self) -> None:
        self.board = self.generate_board().astype(np.int8)
        self.player_row, self.player_col = self.variation_space["agent"]["position"].value
        self.player_y = self.player_row + 0.5
        self.player_x = self.player_col + 0.5
        self.steps = 0
        self.generate_goal()

    def _get_obs(self) -> dict[str, Any]:
        return {
            "board": self.board.copy(),
            "perception": np.array([self.player_row, self.player_col], dtype=np.int32),
        }

    def generate_board(self) -> np.ndarray:
        prob_gravel = self.variation_space["board"]["prob_gravel"].value.item()
        probs = [0.0, 1 - prob_gravel, prob_gravel]
        if not np.isclose(sum(probs), 1.0):
            raise ValueError("Probabilities must sum to 1")

        size = self.variation_space["board"]["size"].value
        board = np.zeros((self.max_size, self.max_size), dtype=np.int8)
        board[:size, :size] = self._rng.choice([0, 1, 2], size=(size, size), p=probs)
        return board

    def render_board(self, ax: plt.Axes | None = None) -> None:
        void_color = self.variation_space["board"]["void_color"].value
        sand_color = self.variation_space["board"]["sand_color"].value
        gravel_color = self.variation_space["board"]["gravel_color"].value
        goal_color = self.variation_space["goal"]["color"].value
        lut = np.array([void_color, sand_color, gravel_color, goal_color], dtype=float) / 255.0

        size = self.variation_space["board"]["size"].value
        board = self.board[:size, :size]
        board[self.goal_pos[0], self.goal_pos[1]] = 3

        img = lut[board]
        if ax is None:
            _, ax = plt.subplots(figsize=(board.shape[1] * 0.2, board.shape[0] * 0.2))

        ax.imshow(img, interpolation="nearest", origin="lower", extent=[0, size, 0, size])
        ax.set_xticks([])
        ax.set_yticks([])

    def _void_cell(self, r: int, c: int) -> int:
        """Void the single cell at (r, c) and return 1 if it was newly voided, else 0."""

        prob_break = self.variation_space["agent"]["prob_break"].value.item()
        should_void = self._rng.random() < prob_break

        if should_void and self.board[r, c] != 0:
            self.board[r, c] = 0
            return 1
        return 0

    def check_termination(self) -> bool:
        """
        Success = all blocks are void except under the agent,
        AND the agent is at the goal position.
        For 1x1 agent, 'under the agent' is just its current cell.
        """
        size = self.variation_space["board"]["size"].value
        r, c = self.player_row, self.player_col

        board_copy = self.board[:size, :size].copy()
        board_copy[r, c] = 0  # ignore agent cell
        all_voided = np.count_nonzero(board_copy) == 0

        at_goal = (r, c) == self.goal_pos
        return bool(all_voided and at_goal)

    def set_state(
        self,
        board: np.ndarray,
        player_pos: tuple[int, int],
        *,
        validate: bool = True,
        render: bool = False,
    ) -> dict[str, Any]:
        if validate:
            size = self.variation_space["board"]["size"].value
            if board.shape != (size, size):
                raise ValueError("Invalid board shape")
            r, c = player_pos
            if not (0 <= r < size and 0 <= c < size):
                raise ValueError("player_pos out of bounds")

        self.board = board.astype(np.int8, copy=False)
        self.player_row, self.player_col = map(int, player_pos)
        self.player_y, self.player_x = self.player_row + 0.5, self.player_col + 0.5
        self.steps = 0
        if render:
            self.render()
        return self._get_obs()

    def generate_goal(self, *, cell_value: int = 3) -> None:
        prev_board, prev_row, prev_col = (
            self.board.copy(),
            self.player_row,
            self.player_col,
        )
        size = self.variation_space["board"]["size"].value
        prev_y, prev_x, prev_steps = self.player_y, self.player_x, self.steps
        try:
            self.goal_pos = self.variation_space["goal"]["position"].value
            r, c = self.goal_pos[0], self.goal_pos[1]
            board = np.zeros((size, size), dtype=np.int8)
            board[r, c] = cell_value
            _ = self.set_state(board, (r, c), validate=True, render=False)
            self._goal = self.render(mode=self.render_mode)

        finally:
            self.board, self.player_row, self.player_col = (
                prev_board,
                prev_row,
                prev_col,
            )
            self.player_y, self.player_x, self.steps = prev_y, prev_x, prev_steps

    def check_location(self, x):
        size = int(self.variation_space.value["board"]["size"])
        # 1x1 agent can exist anywhere inside the board
        return (0 <= x[0] < size) and (0 <= x[1] < size)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
