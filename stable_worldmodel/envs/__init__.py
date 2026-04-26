from gymnasium.envs import registration


WORLDS = set()


def register(id, entry_point, **kwargs):
    registration.register(id=id, entry_point=entry_point, **kwargs)
    WORLDS.add(id)


##############
# CONTINUOUS #
##############

# register(
#     id="swm/ImagePositioning-v1",
#     entry_point="stable_worldmodel.envs.image_positioning:ImagePositioning",
# )

register(
    id='swm/PushT-v1',
    entry_point='stable_worldmodel.envs.pusht.env:PushT',
)

register(
    id='swm/SimplePointMaze-v0',
    entry_point='stable_worldmodel.envs.simple_point_maze:SimplePointMazeEnv',
)

register(
    id='swm/TwoRoom-v1',
    entry_point='stable_worldmodel.envs.two_room.env:TwoRoomEnv',
)

register(
    id='swm/OGBCube-v0',
    entry_point='stable_worldmodel.envs.ogbench.cube_env:CubeEnv',
)

register(
    id='swm/OGBScene-v0',
    entry_point='stable_worldmodel.envs.ogbench.scene_env:SceneEnv',
)

register(
    id='swm/OGBPointMaze-v0',
    entry_point='stable_worldmodel.envs.ogbench.pointmaze_env:PointMazeEnv',
)

register(
    id='swm/PFRocketLanding-v0',
    entry_point='stable_worldmodel.envs.rocket_landing.pyflyt_rocketlanding:RocketLandingEnv',
)

register(
    id='swm/HumanoidDMControl-v0',
    entry_point='stable_worldmodel.envs.dmcontrol.humanoid:HumanoidDMControlWrapper',
)

register(
    id='swm/CheetahDMControl-v0',
    entry_point='stable_worldmodel.envs.dmcontrol.cheetah:CheetahDMControlWrapper',
)

register(
    id='swm/HopperDMControl-v0',
    entry_point='stable_worldmodel.envs.dmcontrol.hopper:HopperDMControlWrapper',
)

register(
    id='swm/ReacherDMControl-v0',
    entry_point='stable_worldmodel.envs.dmcontrol.reacher:ReacherDMControlWrapper',
)

register(
    id='swm/WalkerDMControl-v0',
    entry_point='stable_worldmodel.envs.dmcontrol.walker:WalkerDMControlWrapper',
)

register(
    id='swm/AcrobotDMControl-v0',
    entry_point='stable_worldmodel.envs.dmcontrol.acrobot:AcrobotDMControlWrapper',
)

register(
    id='swm/PendulumDMControl-v0',
    entry_point='stable_worldmodel.envs.dmcontrol.pendulum:PendulumDMControlWrapper',
)

register(
    id='swm/CartpoleDMControl-v0',
    entry_point='stable_worldmodel.envs.dmcontrol.cartpole:CartpoleDMControlWrapper',
)

register(
    id='swm/BallInCupDMControl-v0',
    entry_point='stable_worldmodel.envs.dmcontrol.ball_in_cup:BallInCupDMControlWrapper',
)

register(
    id='swm/FingerDMControl-v0',
    entry_point='stable_worldmodel.envs.dmcontrol.finger:FingerDMControlWrapper',
)

register(
    id='swm/ManipulatorDMControl-v0',
    entry_point='stable_worldmodel.envs.dmcontrol.manipulator:ManipulatorDMControlWrapper',
)

register(
    id='swm/QuadrupedDMControl-v0',
    entry_point='stable_worldmodel.envs.dmcontrol.quadruped:QuadrupedDMControlWrapper',
)

register(
    id='swm/Piecewise-v0',
    entry_point='stable_worldmodel.envs.piecewise.piecewise_env:PiecewiseEnv',
)

_FETCH_ENTRY = 'stable_worldmodel.envs.gymnasium_robotics.fetch:FetchWrapper'

# Sparse reward + flattened Box obs (default; good for behavior cloning / simple SAC)
for _swm_id, _gym_id in [
    ('swm/FetchReach-v3', 'FetchReach-v4'),
    ('swm/FetchPush-v3', 'FetchPush-v4'),
    ('swm/FetchSlide-v3', 'FetchSlide-v4'),
    ('swm/FetchPickAndPlace-v3', 'FetchPickAndPlace-v4'),
]:
    register(id=_swm_id, entry_point=_FETCH_ENTRY, kwargs={'env_id': _gym_id})

# Dense reward + flattened Box obs (standard SAC with shaped reward)
for _swm_id, _gym_id in [
    ('swm/FetchReachDense-v3', 'FetchReachDense-v4'),
    ('swm/FetchPushDense-v3', 'FetchPushDense-v4'),
    ('swm/FetchSlideDense-v3', 'FetchSlideDense-v4'),
    ('swm/FetchPickAndPlaceDense-v3', 'FetchPickAndPlaceDense-v4'),
]:
    register(id=_swm_id, entry_point=_FETCH_ENTRY, kwargs={'env_id': _gym_id})

# Sparse reward + Dict obs (observation/achieved_goal/desired_goal preserved)
# Required by goal-conditioned algorithms such as SB3's HerReplayBuffer.
for _swm_id, _gym_id in [
    ('swm/FetchReachDict-v3', 'FetchReach-v4'),
    ('swm/FetchPushDict-v3', 'FetchPush-v4'),
    ('swm/FetchSlideDict-v3', 'FetchSlide-v4'),
    ('swm/FetchPickAndPlaceDict-v3', 'FetchPickAndPlace-v4'),
]:
    register(
        id=_swm_id,
        entry_point=_FETCH_ENTRY,
        kwargs={'env_id': _gym_id, 'flatten': False},
    )

############
# DISCRETE #
############


register(
    id='swm/PushT-Discrete-v1',
    entry_point='stable_worldmodel.envs.pusht:PushTDiscrete',
)

for _swm_id, _entry in [
    (
        'swm/CraftaxPixels-v1',
        'stable_worldmodel.envs.craftax.craftax:CraftaxPixelsWrapper',
    ),
    (
        'swm/CraftaxSymbolic-v1',
        'stable_worldmodel.envs.craftax.craftax:CraftaxSymbolicWrapper',
    ),
    (
        'swm/CraftaxClassicPixels-v1',
        'stable_worldmodel.envs.craftax.craftax:CraftaxClassicPixelsWrapper',
    ),
    (
        'swm/CraftaxClassicSymbolic-v1',
        'stable_worldmodel.envs.craftax.craftax:CraftaxClassicSymbolicWrapper',
    ),
]:
    register(id=_swm_id, entry_point=_entry)
