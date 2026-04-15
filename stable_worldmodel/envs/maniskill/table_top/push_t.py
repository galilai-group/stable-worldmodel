from stable_worldmodel.envs.maniskill.table_top.base import TableTopManiSkillWrapper


class PushTManiSkillWrapper(TableTopManiSkillWrapper):

    task_id = 'PushT-v1'
    actors = {
        'tee':      {'attrs': ('tee',), 'rgb': (0.502, 0.502, 0.502), 'physics': True,
                     'mass': (0.05, 0.5, 0.15)},
        'goal_tee': {'attrs': ('goal_tee',), 'rgb': (0.761, 0.075, 0.086)},
    }
