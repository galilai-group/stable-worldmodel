from stable_worldmodel.envs.maniskill.table_top.base import TableTopManiSkillWrapper


class StackPyramidManiSkillWrapper(TableTopManiSkillWrapper):

    task_id = 'StackPyramid-v1'
    actors = {
        'cubeA': {'attrs': ('cubeA',), 'rgb': (1.0, 0.0, 0.0), 'physics': True},
        'cubeB': {'attrs': ('cubeB',), 'rgb': (0.0, 1.0, 0.0), 'physics': True},
        'cubeC': {'attrs': ('cubeC',), 'rgb': (0.0, 0.0, 1.0), 'physics': True},
    }
