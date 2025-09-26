import os

from gymnasium.envs.registration import register

from . import data, policy, solver, wrappers, wm, utils, spaces
from .policy import PlanConfig
from .utils import pretraining
from .world import World

register(
    id="swm/ImagePositioning-v1",
    entry_point="stable_worldmodel.envs.image_positioning:ImagePositioning",
)

register(
    id="swm/PushT-v1",
    entry_point="stable_worldmodel.envs.pusht:PushT",
)

register(
    id="swm/SimplePointMaze-v0",
    entry_point="stable_worldmodel.envs.simple_point_maze:SimplePointMazeEnv",
)
