
from contextlib import contextmanager

from stable_worldmodel.envs.maniskill.table_top.base import TableTopManiSkillWrapper


@contextmanager
def _patched_cube_half_size(cube_half_size, robot_uids):
    if cube_half_size is None:
        yield
        return
    from mani_skill.envs.tasks.tabletop.pick_cube_cfgs import PICK_CUBE_CONFIGS

    cfg = PICK_CUBE_CONFIGS.get(robot_uids) or PICK_CUBE_CONFIGS['panda']
    old = cfg['cube_half_size']
    cfg['cube_half_size'] = float(cube_half_size)
    try:
        yield
    finally:
        cfg['cube_half_size'] = old


class PickCubeManiSkillWrapper(TableTopManiSkillWrapper):

    task_id = 'PickCube-v1'
    actors = {
        'cube': {'attrs': ('cube',), 'rgb': (1.0, 0.2, 0.2)},
        'goal': {'attrs': ('goal_site', 'goal'), 'rgb': (0.2, 1.0, 0.2)},
    }

    def __init__(
        self,
        control_mode: str = 'pd_ee_delta_pose',
        seed: int | None = None,
        cube_half_size: float | None = None,
        **make_kwargs,
    ):
        robot_uids = make_kwargs.get('robot_uids', 'panda')
        with _patched_cube_half_size(cube_half_size, robot_uids):
            super().__init__(control_mode=control_mode, seed=seed, **make_kwargs)
