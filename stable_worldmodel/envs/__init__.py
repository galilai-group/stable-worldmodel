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
    id='swm/PickCubeManiSkill-v0',
    entry_point='stable_worldmodel.envs.maniskill.table_top.pick_cube:PickCubeManiSkillWrapper',
)

register(
    id='swm/PushCubeManiSkill-v0',
    entry_point='stable_worldmodel.envs.maniskill.table_top.push_cube:PushCubeManiSkillWrapper',
)

register(
    id='swm/PullCubeManiSkill-v0',
    entry_point='stable_worldmodel.envs.maniskill.table_top.pull_cube:PullCubeManiSkillWrapper',
)

register(
    id='swm/PullCubeToolManiSkill-v0',
    entry_point='stable_worldmodel.envs.maniskill.table_top.pull_cube_tool:PullCubeToolManiSkillWrapper',
)

register(
    id='swm/LiftPegUprightManiSkill-v0',
    entry_point='stable_worldmodel.envs.maniskill.table_top.lift_peg_upright:LiftPegUprightManiSkillWrapper',
)

register(
    id='swm/PegInsertionSideManiSkill-v0',
    entry_point='stable_worldmodel.envs.maniskill.table_top.peg_insertion_side:PegInsertionSideManiSkillWrapper',
)

register(
    id='swm/PlugChargerManiSkill-v0',
    entry_point='stable_worldmodel.envs.maniskill.table_top.plug_charger:PlugChargerManiSkillWrapper',
)

register(
    id='swm/PokeCubeManiSkill-v0',
    entry_point='stable_worldmodel.envs.maniskill.table_top.poke_cube:PokeCubeManiSkillWrapper',
)

register(
    id='swm/RollBallManiSkill-v0',
    entry_point='stable_worldmodel.envs.maniskill.table_top.roll_ball:RollBallManiSkillWrapper',
)

register(
    id='swm/StackCubeManiSkill-v0',
    entry_point='stable_worldmodel.envs.maniskill.table_top.stack_cube:StackCubeManiSkillWrapper',
)

register(
    id='swm/StackPyramidManiSkill-v0',
    entry_point='stable_worldmodel.envs.maniskill.table_top.stack_pyramid:StackPyramidManiSkillWrapper',
)

register(
    id='swm/PlaceSphereManiSkill-v0',
    entry_point='stable_worldmodel.envs.maniskill.table_top.place_sphere:PlaceSphereManiSkillWrapper',
)

register(
    id='swm/PushTManiSkill-v0',
    entry_point='stable_worldmodel.envs.maniskill.table_top.push_t:PushTManiSkillWrapper',
)

register(
    id='swm/AssemblingKitsManiSkill-v0',
    entry_point='stable_worldmodel.envs.maniskill.table_top.assembling_kits:AssemblingKitsManiSkillWrapper',
)

register(
    id='swm/PickSingleYCBManiSkill-v0',
    entry_point='stable_worldmodel.envs.maniskill.table_top.pick_single_ycb:PickSingleYCBManiSkillWrapper',
)

register(
    id='swm/PickCubeSO100ManiSkill-v0',
    entry_point='stable_worldmodel.envs.maniskill.table_top.pick_cube_so100:PickCubeSO100ManiSkillWrapper',
)

register(
    id='swm/PickCubeWidowXAIManiSkill-v0',
    entry_point='stable_worldmodel.envs.maniskill.table_top.pick_cube_widowxai:PickCubeWidowXAIManiSkillWrapper',
)

register(
    id='swm/TwoRobotPickCubeManiSkill-v0',
    entry_point='stable_worldmodel.envs.maniskill.table_top.two_robot_pick_cube:TwoRobotPickCubeManiSkillWrapper',
)

register(
    id='swm/TwoRobotStackCubeManiSkill-v0',
    entry_point='stable_worldmodel.envs.maniskill.table_top.two_robot_stack_cube:TwoRobotStackCubeManiSkillWrapper',
)


############
# DISCRETE #
############

register(
    id='swm/SimpleNavigation-v0',
    entry_point='stable_worldmodel.envs.simple_nav.env:SimpleNavigationEnv',
)

register(
    id='swm/PushT-Discrete-v1',
    entry_point='stable_worldmodel.envs.pusht:PushTDiscrete',
)
