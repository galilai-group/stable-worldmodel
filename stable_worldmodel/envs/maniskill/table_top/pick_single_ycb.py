from stable_worldmodel.envs.maniskill.table_top.base import TableTopManiSkillWrapper


class PickSingleYCBManiSkillWrapper(TableTopManiSkillWrapper):

    task_id = 'PickSingleYCB-v1'
    actors = {


        'obj':  {'attrs': ('obj',), 'rgb': (0.8, 0.8, 0.8), 'physics': True,
                 'mass': (0.02, 1.0, 0.2)},
        'goal': {'attrs': ('goal_site',), 'rgb': (0.0, 1.0, 0.0)},
    }
