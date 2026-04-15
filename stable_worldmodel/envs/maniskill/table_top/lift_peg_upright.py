from stable_worldmodel.envs.maniskill.table_top.base import TableTopManiSkillWrapper


class LiftPegUprightManiSkillWrapper(TableTopManiSkillWrapper):

    task_id = 'LiftPegUpright-v1'
    actors = {


        'peg': {'attrs': ('peg',), 'rgb': (0.690, 0.055, 0.055), 'physics': True,
                'mass': (0.05, 0.5, 0.15)},
    }
