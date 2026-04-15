from stable_worldmodel.envs.maniskill.table_top.base import TableTopManiSkillWrapper


class PullCubeToolManiSkillWrapper(TableTopManiSkillWrapper):

    task_id = 'PullCubeTool-v1'
    actors = {
        'cube': {'attrs': ('cube',), 'rgb': (0.047, 0.165, 0.627), 'physics': True},
        'tool': {'attrs': ('l_shape_tool',), 'rgb': (1.0, 0.0, 0.0), 'physics': True,
                 'mass': (0.05, 1.0, 0.2)},
    }
