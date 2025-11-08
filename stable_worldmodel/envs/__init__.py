__version__ = "0.0.1a0"

from gymnasium.envs import registration


WORLDS = set()


def register(id, entry_point):
    registration.register(id=id, entry_point=entry_point)
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

# register(
#     id="swm/VoidRun-v0",
#     entry_point="stable_worldmodel.envs.voidrun:VoidRunEnv",
# )

register(
    id="swm/OGBCube-v0",
    entry_point="stable_worldmodel.envs.ogbench_cube:CubeEnv",
)

register(
    id="swm/OGBScene-v0",
    entry_point="stable_worldmodel.envs.ogbench_scene:SceneEnv",
)

register(
    id="swm/PFRocketLanding-v0",
    entry_point="stable_worldmodel.envs.rocket_landing.pyflyt_rocketlanding:RocketLandingEnv",
)
