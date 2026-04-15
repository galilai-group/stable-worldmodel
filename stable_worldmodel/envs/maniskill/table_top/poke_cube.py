from stable_worldmodel.envs.maniskill.table_top.base import TableTopManiSkillWrapper


class PokeCubeManiSkillWrapper(TableTopManiSkillWrapper):

    task_id = 'PokeCube-v1'
    actors = {
        'cube': {'attrs': ('cube',), 'rgb': (1.0, 0.0, 0.0), 'physics': True},
        'peg':  {'attrs': ('peg',), 'rgb': (0.047, 0.165, 0.627), 'physics': True,
                 'mass': (0.05, 0.5, 0.15)},
        'goal': {'attrs': ('goal_region',), 'rgb': (0.85, 0.1, 0.1)},
    }
