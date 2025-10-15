__version__ = "0.0.1a0"

from gymnasium.envs import registration

from stable_worldmodel import (
    data,
    envs,
    policy,
    solver,
    spaces,
    utils,
    wm,
    wrappers,
)
from stable_worldmodel.policy import PlanConfig
from stable_worldmodel.utils import pretraining
from stable_worldmodel.world import World


WORLDS = set()


def register(id, entry_point, **kwargs):
    registration.register(id=id, entry_point=entry_point, kwargs=kwargs)
    WORLDS.add(id)


# register(
#     id="swm/ImagePositioning-v1",
#     entry_point="stable_worldmodel.envs.image_positioning:ImagePositioning",
# )

register(
    id="swm/PushT-v1",
    entry_point="stable_worldmodel.envs.pusht:PushT",
)

register(
    id="swm/SimplePointMaze-v0",
    entry_point="stable_worldmodel.envs.simple_point_maze:SimplePointMazeEnv",
)

register(
    id="swm/TwoRoom-v0",
    entry_point="stable_worldmodel.envs.two_room:TwoRoomEnv",
)

register(
    id="swm/VoidRun-v0",
    entry_point="stable_worldmodel.envs.voidrun:VoidRunEnv",
)

register(
    id="swm/OGBCube-v0",
    entry_point="stable_worldmodel.envs.ogbench_cube:CubeEnv",
)

register(
    id="swm/OGBAntMaze-v0",
    entry_point="stable_worldmodel.envs.ogbench_antmaze:AntMazeEnv",
)

register(
    id="swm/OGBAntMaze-Inside-v0",
    entry_point="stable_worldmodel.envs.ogbench_antmaze:AntMazeEnv",
    inside=True,
    camera_name="back",
)


__all__ = [
    "World",
    "PlanConfig",
    "pretraining",
    "spaces",
    "utils",
    "envs",
    "data",
    "policy",
    "solver",
    "wrappers",
    "wm",
]
