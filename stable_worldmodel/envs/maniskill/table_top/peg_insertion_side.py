from stable_worldmodel.envs.maniskill.table_top.base import TableTopManiSkillWrapper


class PegInsertionSideManiSkillWrapper(TableTopManiSkillWrapper):

    task_id = 'PegInsertionSide-v1'
    actors = {
        'peg': {'attrs': ('peg',), 'rgb': (0.925, 0.451, 0.341), 'physics': True,
                'mass': (0.05, 0.5, 0.15)},

        'box': {'attrs': ('box',), 'rgb': (0.929, 0.965, 0.976)},
    }
