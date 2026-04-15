from stable_worldmodel.envs.maniskill.table_top.base import TableTopManiSkillWrapper


class TwoRobotPickCubeManiSkillWrapper(TableTopManiSkillWrapper):

    task_id = 'TwoRobotPickCube-v1'
    actors = {
        'cube': {'attrs': ('cube',), 'rgb': (1.0, 0.0, 0.0), 'physics': True},
        'goal': {'attrs': ('goal_site',), 'rgb': (0.0, 1.0, 0.0)},
    }
