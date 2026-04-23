from stable_worldmodel.envs.maniskill.table_top.base import TableTopManiSkillWrapper


class PlugChargerManiSkillWrapper(TableTopManiSkillWrapper):

    task_id = 'PlugCharger-v1'
    actors = {
        'charger': {'attrs': ('charger',), 'rgb': (0.859, 0.710, 0.224)},
        'receptacle': {'attrs': ('receptacle',), 'rgb': (1.0, 1.0, 1.0)},
    }
