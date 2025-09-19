import math
from typing import Optional, Dict, Any, Tuple

import torch
import gymnasium as gym
import numpy as np
import random


InfoType = Dict[str, Any]
ObsType = torch.Tensor


def check_wall_intersect(
    pos1,
    pos2,
    wall_x,
    hole_y,
    wall_width,
    door_space,
    border_wall_loc,
    img_size,
    add_noise=True,
):
    """
    Parameters:
        pos1: [2]
        pos2: [2]
        wall_x: []
        hole_y: []
        wall_width: int
        door_space: int
        border_wall_loc: int
        img_size: int
    Returns:
        intersect: [2]
        intersect_w_noise: [2]
    """

    # first, we check to see if the point bumps into the middle wall's width
    left_wall_corner, right_wall_corner = (
        wall_x - wall_width // 2,
        wall_x + wall_width // 2,
    )
    door_bot, door_top = hole_y - door_space, hole_y + door_space

    # check if it's moving upwards and crosses the door top horizontal line
    if pos2[1] - pos1[1] > 0 and pos2[1] > door_top and pos1[1] < door_top:
        # get x intercept with door top horizontal line
        intersect = check_horizontal_wall_intersect(
            pos1, pos2, door_top, None, door_space
        )
        # if x intercept occurs between the left and right wall
        if (
            intersect is not None
            and left_wall_corner <= intersect[0]
            and intersect[0] <= right_wall_corner
        ):
            # add downward noise and return early
            noise = torch.randn(2, device=pos1.device) * 0.5
            noise[1] = noise[1].abs() * -1
            return intersect, intersect + noise

    # check if it's moving downwards and croseses the door bot horizontal line
    if pos2[1] - pos1[1] < 0 and pos2[1] < door_bot and pos1[1] > door_bot:
        # get x intercept with door bot horizontal line
        intersect = check_horizontal_wall_intersect(
            pos1, pos2, door_bot, None, door_space
        )
        # if x intercept occurs between the left and right wall
        if (
            intersect is not None
            and left_wall_corner <= intersect[0]
            and intersect[0] <= right_wall_corner
        ):
            # add upward noise and return early
            noise = torch.randn(2, device=pos1.device) * 0.5
            noise[1] = noise[1].abs()
            return intersect, intersect + noise

    # next, we check to see if point bumps into border and wall proper
    left_wall, left_hole = border_wall_loc - 1, None
    right_wall, right_hole = img_size - border_wall_loc, None
    if wall_x > pos1[0]:
        right_wall, right_hole = wall_x - wall_width // 2, hole_y
    else:
        left_wall, left_hole = wall_x + wall_width // 2, hole_y

    top_wall, top_hole = border_wall_loc - 1, None
    bot_wall, bot_hole = img_size - border_wall_loc, None

    vertical_intersect = check_vertical_wall_intersect(
        pos1, pos2, left_wall, left_hole, door_space
    )
    if vertical_intersect is None:
        vertical_intersect = check_vertical_wall_intersect(
            pos1, pos2, right_wall, right_hole, door_space
        )

    horizontal_intersect = check_horizontal_wall_intersect(
        pos1, pos2, top_wall, top_hole, door_space
    )
    if horizontal_intersect is None:
        horizontal_intersect = check_horizontal_wall_intersect(
            pos1, pos2, bot_wall, bot_hole, door_space
        )

    if vertical_intersect is not None:
        sign = torch.sign(pos1[0] - vertical_intersect[0])
        vertical_noise = torch.randn(2, device=pos1.device) * 0.5
        vertical_noise[0] = vertical_noise[0].abs() * sign

    if horizontal_intersect is not None:
        sign = torch.sign(pos1[1] - horizontal_intersect[1])
        horizontal_noise = torch.randn(2, device=pos1.device) * 0.5
        horizontal_noise[1] = horizontal_noise[1].abs() * sign

    if vertical_intersect is not None and horizontal_intersect is not None:
        # return the intersection that happens first
        if torch.norm(pos1 - vertical_intersect) < torch.norm(
            pos1 - horizontal_intersect
        ):
            intersect = vertical_intersect
            noise = vertical_noise
        else:
            intersect = horizontal_intersect
            noise = horizontal_noise
    elif vertical_intersect is not None:
        intersect = vertical_intersect
        noise = vertical_noise
    elif horizontal_intersect is not None:
        intersect = horizontal_intersect
        noise = horizontal_noise
    else:
        return None, None

    intersect_w_noise = intersect + noise
    # we make sure after adding noise, we don't cross another wall
    intersect_w_noise[0] = torch.clamp(
        intersect_w_noise[0], min=left_wall, max=right_wall
    )
    intersect_w_noise[1] = torch.clamp(intersect_w_noise[1], min=top_wall, max=bot_wall)

    if intersect_w_noise[0] <= left_wall:
        intersect_w_noise[0] = left_wall + 0.3
    if intersect_w_noise[0] >= right_wall:
        intersect_w_noise[0] = right_wall - 0.3
    if intersect_w_noise[1] <= top_wall:
        intersect_w_noise[1] = top_wall + 0.3
    if intersect_w_noise[1] >= bot_wall:
        intersect_w_noise[1] = bot_wall - 0.3

    return intersect, intersect_w_noise


class DotWall(gym.Env):
    def __init__(
        self,
        rng: Optional[np.random.Generator] = None,
        border_wall_loc: int = 5,
        wall_width: int = 3,
        door_space: int = 4,
        wall_padding: int = 20,
        img_size: int = 64,
        fix_wall: bool = True,
        cross_wall: bool = True,
        level: str = "medium",
        n_steps: int = 200,
        action_step_mean: float = 1.0,
        max_step_norm: float = 2.45,
        device: Optional[torch.device] = None,
        fix_wall_location: Optional[int] = 32,
        fix_door_location: Optional[int] = 10,
    ):
        super().__init__()
        self.cross_wall = cross_wall
        self.level = level
        self.img_size = img_size
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.dot_std = 1.3
        self.padding = self.dot_std * 2
        self.wall_width = wall_width
        self.door_space = door_space
        self.wall_padding = wall_padding
        self.border_wall_loc = border_wall_loc
        self.border_padding = self.border_wall_loc - 1 + self.padding
        self.n_steps = n_steps
        self.action_step_mean = action_step_mean
        self.rng = rng or np.random.default_rng()

        self.action_space = gym.spaces.Box(
            low=-max_step_norm, high=max_step_norm, shape=(2,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(2, img_size, img_size), dtype=np.float32
        )

        self.fix_wall = fix_wall
        self.fix_wall_location = fix_wall_location
        self.fix_door_location = fix_door_location

    @property
    def np_random(self):
        return self.rng

    def render(self):
        return self._get_obs()

    def reset(self, location=None) -> Tuple[ObsType, InfoType]:
        self.wall_x, self.hole_y = self._generate_wall()
        self.left_wall_x = self.wall_x - self.wall_width // 2
        self.right_wall_x = self.wall_x + self.wall_width // 2

        self.wall_img = self._render_walls(self.wall_x, self.hole_y)
        if location is None:
            self._generate_start_and_target()
        else:
            self.dot_position = location

        self.position_history = [self.dot_position]
        obs = self._render_dot_and_wall()
        info = self._build_info()
        return obs, info

    def _build_info(self) -> InfoType:
        return {
            "dot_position": self.dot_position,
            "target_position": self.target_position,
            "target_obs": self.get_target_obs(),
        }

    def _get_obs(self):
        return self._render_dot_and_wall()

    def get_target_obs(self):
        return self._render_dot_and_wall_target(self.target_position)

    def step(self, action: np.array) -> Tuple[ObsType, float, bool, bool, InfoType]:
        action = torch.tensor(action, device=self.device)
        self.dot_position = self._calculate_next_position(action)
        self.position_history.append(self.dot_position)
        obs = self._render_dot_and_wall()
        done = (self.dot_position - self.target_position).pow(2).mean() < 1.0
        truncated = len(self.position_history) >= self.n_steps
        return obs, 0.0, done, truncated, self._build_info()

    def _calculate_next_position(self, action):
        next_dot_position = self._generate_transition(self.dot_position, action)
        intersect, intersect_w_noise = check_wall_intersect(
            self.dot_position,
            next_dot_position,
            self.wall_x,
            self.hole_y,
            wall_width=self.wall_width,
            door_space=self.door_space,
            border_wall_loc=self.border_wall_loc,
            img_size=self.img_size,
        )
        if intersect is not None:
            next_dot_position = intersect_w_noise
        return next_dot_position

    def _generate_transition(self, location, action):
        next_location = location + action  # [..., :-1] * action[..., -1]
        return next_location

    def _generate_wall(self):
        wall_loc = torch.tensor(self.fix_wall_location, device=self.device)
        door_loc = torch.tensor(self.fix_door_location, device=self.device)

        return wall_loc, door_loc

    def _generate_start_and_target(self):
        # We leave 2 * self.dot_std margin when generating state, and don't let the
        # dot approach the border.
        n_steps = self.n_steps
        if self.cross_wall:
            if self.level == "easy":
                # we make sure start and goal are within (n_steps/2) steps from door

                avg_dist_n_steps = n_steps * self.action_step_mean

                assert (
                    self.wall_padding - self.wall_width // 2 - self.border_wall_loc
                    >= math.ceil(avg_dist_n_steps * 3 / 4)
                )

                start_min_x = self.left_wall_x - math.ceil(avg_dist_n_steps * 3 / 4)
                start_max_x = self.left_wall_x - math.ceil(avg_dist_n_steps * 1 / 4)
                target_min_x = self.right_wall_x + math.ceil(avg_dist_n_steps * 1 / 4)
                target_max_x = self.right_wall_x + math.ceil(avg_dist_n_steps * 3 / 4)
                min_y = max(
                    self.hole_y - math.ceil(avg_dist_n_steps * 3 / 4),
                    self.border_padding,
                )
                max_y = min(
                    self.hole_y + math.ceil(avg_dist_n_steps * 3 / 4),
                    self.img_size - 1 - self.border_padding,
                )
            else:
                start_min_x = self.border_padding
                start_max_x = self.left_wall_x - self.padding
                target_min_x = self.right_wall_x + self.padding
                target_max_x = self.img_size - 1 - self.border_padding
                min_y, max_y = (
                    self.border_padding,
                    self.img_size - 1 - self.border_padding,
                )

            start_x = start_min_x + random.random() * (start_max_x - start_min_x)
            target_x = target_min_x + random.random() * (target_max_x - target_min_x)

            start_y = torch.tensor(
                min_y + random.random() * (max_y - min_y), device=self.device
            )
            target_y = torch.tensor(
                min_y + random.random() * (max_y - min_y), device=self.device
            )

            if random.random() < 0.5:  # inverse travel direction 50% of time
                start_x, target_x = target_x, start_x

            self.dot_position = torch.stack([start_x, start_y])
            self.target_position = torch.stack([target_x, target_y])
        else:
            raise NotImplementedError
            effective_range = (self.img_size - 1) - 2 * self.padding
            location = (
                torch.from_numpy(
                    self.rng.random(size=(4,)) * effective_range + self.padding
                )
                .to(self.device)
                .float()
            )
            if self.level == "easy":
                # make the target location to be within a certain distance from start
                min_dist_from_start = math.ceil(n_steps * 2 / 3)
                max_dist_from_start = math.ceil(n_steps * 3 / 2)
                # generate random angle
                angle = (torch.rand(1) * 2 * torch.pi).to(location.device)
                # generate a random distance c within the range
                dist = (
                    torch.rand(1) * (max_dist_from_start - min_dist_from_start)
                    + min_dist_from_start
                ).to(location.device)
                # set new x and y for goal
                location[2] = location[0] + dist * torch.cos(angle)
                location[3] = location[1] + dist * torch.sin(angle)
                location = torch.clamp(
                    location, min=self.padding, max=self.img_size - 1 - self.padding
                )

            self.dot_position = location[:2]
            self.target_position = location[2:]

    def _render_walls(self, wall_loc, hole_loc):
        # Generates an image of the wall with the door and specified wall thickness.
        x = torch.arange(0, self.img_size, device=self.device)
        y = torch.arange(0, self.img_size, device=self.device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")

        # Calculate the range for the wall based on the wall_width
        half_width = self.wall_width // 2

        # Create the wall mask centered at wall_loc with the given wall_width
        wall_mask = (grid_x >= (wall_loc - half_width)) & (
            grid_x <= (wall_loc + half_width)
        )

        # Door logic remains the same
        door_mask = (hole_loc - self.door_space <= grid_y) & (
            grid_y <= hole_loc + self.door_space
        )

        # Combine the wall and door masks
        res = wall_mask & ~door_mask

        # Convert boolean mask to float
        res = res.float()

        # Set border walls
        border_wall_loc = self.border_wall_loc
        res[:, border_wall_loc - 1] = 1
        res[:, -border_wall_loc] = 1
        res[border_wall_loc - 1, :] = 1
        res[-border_wall_loc, :] = 1

        # to byte

        res = (res * 255.0).clamp(0, 255).to(torch.uint8)

        return res

    def _render_dot(self, location):
        x = torch.linspace(
            0, self.img_size - 1, steps=self.img_size, device=self.device
        )
        y = torch.linspace(
            0, self.img_size - 1, steps=self.img_size, device=self.device
        )
        xx, yy = torch.meshgrid(x, y, indexing="xy")
        c = torch.stack([xx, yy], dim=-1)
        # img = torch.exp(
        #     -(c - location).norm(dim=-1).pow(2) / (2 * self.dot_std * self.dot_std)
        # ) / (2 * math.pi * self.dot_std * self.dot_std)
        img = (
            (
                torch.exp(
                    -(c - location).norm(dim=-1).pow(2)
                    / (2 * self.dot_std * self.dot_std)
                )
                * 255.0
            )
            .clamp(0, 255)
            .to(torch.uint8)
        )
        return img

    def _render_dot_and_wall(self):
        dot_img = self._render_dot(self.dot_position)
        obs_output = torch.stack([dot_img, self.wall_img], dim=0)
        return obs_output

    def _render_dot_and_wall_target(self, location):
        dot_img = self._render_dot(location)
        obs_output = torch.stack([dot_img, self.wall_img], dim=0)
        return obs_output
