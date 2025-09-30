from gymnasium.envs import registration
from . import data, policy, solver, wrappers, wm, utils, spaces
from .policy import PlanConfig
from .utils import pretraining
from .world import World

WORLDS = set()


def register(id, entry_point):
    registration.register(id=id, entry_point=entry_point)

    if id in WORLDS:
        raise ValueError(f"World {id} already registered.")

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
