from stable_worldmodel.envs.maniskill.table_top.base import TableTopManiSkillWrapper


class PickCubeWidowXAIManiSkillWrapper(TableTopManiSkillWrapper):

    task_id = 'PickCubeWidowXAI-v1'
    default_control_mode = 'pd_joint_delta_pos'
    actors = {
        'cube': {'attrs': ('cube',), 'rgb': (1.0, 0.0, 0.0)},
        'goal': {'attrs': ('goal_site', 'goal'), 'rgb': (0.0, 1.0, 0.0)},
    }
