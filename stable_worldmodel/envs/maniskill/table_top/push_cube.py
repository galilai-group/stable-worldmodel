from stable_worldmodel.envs.maniskill.table_top.base import TableTopManiSkillWrapper


class PushCubeManiSkillWrapper(TableTopManiSkillWrapper):

    task_id = 'PushCube-v1'
    actors = {
        'cube': {'attrs': ('obj',), 'rgb': (0.047, 0.165, 0.627)},
        'goal': {'attrs': ('goal_region',), 'rgb': (0.85, 0.1, 0.1)},
    }
