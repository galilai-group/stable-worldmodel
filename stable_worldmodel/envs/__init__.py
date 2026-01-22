__version__ = "0.0.1a0"

from gymnasium.envs import registration


WORLDS = set()


def register(id, entry_point):
    registration.register(id=id, entry_point=entry_point)
    WORLDS.add(id)


##############
# CONTINUOUS #
##############

# register(
#     id="swm/ImagePositioning-v1",
#     entry_point="stable_worldmodel.envs.image_positioning:ImagePositioning",
# )

# register(
#     id="swm/VoidRun-v0",
#     entry_point="stable_worldmodel.envs.voidrun:VoidRunEnv",
# )


register(
    id="swm/PushT-v1",
    entry_point="stable_worldmodel.envs.pusht.env:PushT",
)

register(
    id="swm/SimplePointMaze-v0",
    entry_point="stable_worldmodel.envs.simple_point_maze:SimplePointMazeEnv",
)

register(
    id="swm/TwoRoom-v0",
    entry_point="stable_worldmodel.envs.two_room.env:TwoRoomEnv",
)

register(
    id="swm/OGBCube-v0",
    entry_point="stable_worldmodel.envs.ogbench_manip.cube_env:CubeEnv",
)

register(
    id="swm/OGBScene-v0",
    entry_point="stable_worldmodel.envs.ogbench_manip.scene_env:SceneEnv",
)

register(
    id="swm/PFRocketLanding-v0",
    entry_point="stable_worldmodel.envs.rocket_landing.pyflyt_rocketlanding:RocketLandingEnv",
)

register(
    id="swm/HumanoidDMControl-v0",
    entry_point="stable_worldmodel.envs.dmcontrol.humanoid:HumanoidDMControlWrapper",
)

register(
    id="swm/CheetahDMControl-v0",
    entry_point="stable_worldmodel.envs.dmcontrol.cheetah:CheetahDMControlWrapper",
)

register(
    id="swm/HopperDMControl-v0",
    entry_point="stable_worldmodel.envs.dmcontrol.hopper:HopperDMControlWrapper",
)

register(
    id="swm/ReacherDMControl-v0",
    entry_point="stable_worldmodel.envs.dmcontrol.reacher:ReacherDMControlWrapper",
)

register(
    id="swm/WalkerDMControl-v0",
    entry_point="stable_worldmodel.envs.dmcontrol.walker:WalkerDMControlWrapper",
)

register(
    id="swm/AcrobotDMControl-v0",
    entry_point="stable_worldmodel.envs.dmcontrol.acrobot:AcrobotDMControlWrapper",
)

register(
    id="swm/PendulumDMControl-v0",
    entry_point="stable_worldmodel.envs.dmcontrol.pendulum:PendulumDMControlWrapper",
)

register(
    id="swm/CartpoleDMControl-v0",
    entry_point="stable_worldmodel.envs.dmcontrol.cartpole:CartpoleDMControlWrapper",
)


############
# DISCRETE #
############

register(
    id="swm/SimpleNavigation-v0",
    entry_point="stable_worldmodel.envs.simple_nav.env:SimpleNavigationEnv",
)

register(
    id="swm/PushT-Discrete-v1",
    entry_point="stable_worldmodel.envs.pusht:PushTDiscrete",
)
