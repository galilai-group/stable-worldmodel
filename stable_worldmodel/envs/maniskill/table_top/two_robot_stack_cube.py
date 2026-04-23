from stable_worldmodel.envs.maniskill.table_top.base import TableTopManiSkillWrapper


class TwoRobotStackCubeManiSkillWrapper(TableTopManiSkillWrapper):

    task_id = 'TwoRobotStackCube-v1'
    actors = {
        'cubeA': {'attrs': ('cubeA',), 'rgb': (0.047, 0.165, 0.627)},
        'cubeB': {'attrs': ('cubeB',), 'rgb': (0.0, 1.0, 0.0)},
        'goal':  {'attrs': ('goal_region',), 'rgb': (0.85, 0.1, 0.1)},
    }
