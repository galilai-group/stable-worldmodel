from stable_worldmodel.envs.maniskill.table_top.base import TableTopManiSkillWrapper


class StackCubeManiSkillWrapper(TableTopManiSkillWrapper):

    task_id = 'StackCube-v1'
    actors = {
        'cubeA': {'attrs': ('cubeA',), 'rgb': (1.0, 0.0, 0.0), 'physics': True},
        'cubeB': {'attrs': ('cubeB',), 'rgb': (0.0, 1.0, 0.0), 'physics': True},
    }
