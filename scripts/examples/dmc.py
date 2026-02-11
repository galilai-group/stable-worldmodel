import os

os.environ['MUJOCO_GL'] = 'egl'

import stable_worldmodel as swm
from stable_worldmodel.envs.dmcontrol import ExpertPolicy

world = swm.World(
    'swm/CheetahDMControl-v0',
    num_envs=3,
    image_shape=(224, 224),
    max_episode_steps=500,
)
world.set_policy(
    ExpertPolicy(
        ckpt_path='path/to/dmc/cheetah/expert_policy.zip',
        vec_normalize_path='path/to/dmc/cheetah/vec_normalize.pkl',
        device='cuda',
    )
)

world.record_video('./', max_steps=500)
