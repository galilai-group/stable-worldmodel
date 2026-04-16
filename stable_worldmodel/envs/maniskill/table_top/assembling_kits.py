from stable_worldmodel.envs.maniskill.table_top.base import TableTopManiSkillWrapper


class AssemblingKitsManiSkillWrapper(TableTopManiSkillWrapper):

    task_id = 'AssemblingKits-v1'
    actors = {
        'obj': {'attrs': ('obj',), 'rgb': (0.85, 0.45, 0.20)},
    }
