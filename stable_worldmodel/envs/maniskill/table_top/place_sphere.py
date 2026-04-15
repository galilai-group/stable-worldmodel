from stable_worldmodel.envs.maniskill.table_top.base import TableTopManiSkillWrapper


class PlaceSphereManiSkillWrapper(TableTopManiSkillWrapper):

    task_id = 'PlaceSphere-v1'
    actors = {
        'sphere': {'attrs': ('obj',), 'rgb': (0.047, 0.165, 0.627), 'physics': True,
                   'mass': (0.02, 0.3, 0.05)},

        'bin':    {'attrs': ('bin',), 'rgb': (0.6, 0.4, 0.2)},
    }
