from stable_worldmodel.envs.maniskill.table_top.base import TableTopManiSkillWrapper


class PickCubeSO100ManiSkillWrapper(TableTopManiSkillWrapper):

    task_id = 'PickCubeSO100-v1'
    default_control_mode = 'pd_joint_delta_pos'
    actors = {
        'cube': {'attrs': ('cube',), 'rgb': (1.0, 0.0, 0.0), 'physics': True},
        'goal': {'attrs': ('goal_site', 'goal'), 'rgb': (0.0, 1.0, 0.0)},
    }
