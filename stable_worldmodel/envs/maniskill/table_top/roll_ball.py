from stable_worldmodel.envs.maniskill.table_top.base import TableTopManiSkillWrapper


class RollBallManiSkillWrapper(TableTopManiSkillWrapper):

    task_id = 'RollBall-v1'
    actors = {
        'ball': {'attrs': ('ball',), 'rgb': (0.0, 0.2, 0.8), 'physics': True,
                 'mass': (0.02, 0.3, 0.05)},
        'goal': {'attrs': ('goal_region',), 'rgb': (0.85, 0.1, 0.1)},
    }
