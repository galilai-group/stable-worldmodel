import os

os.environ['MUJOCO_GL'] = 'egl'

import stable_worldmodel as swm
from stable_worldmodel.envs.dmcontrol import ExpertPolicy


# record gif


ENVS = {
    'swm/CartpoleDMControl-v0': ('cartpole',),
    'swm/WalkerDMControl-v0': ('walker',),
    'swm/QuadrupedDMControl-v0': ('quadruped',),
    'swm/BallInCupDMControl-v0': ('ballincup',),
    # 'swm/AcrobotDMControl-v0': ('acrobot',),
    'swm/FingerDMControl-v0': ('finger',),
    'swm/HopperDMControl-v0': ('hopper',),
    # 'swm/HumanoidDMControl-v0': ('humanoid',),
    # 'swm/ManipulatorDMControl-v0': ('manipulator',),
    'swm/CheetahDMControl-v0': ('cheetah',),
    'swm/ReacherDMControl-v0': ('reacher',),
    'swm/PendulumDMControl-v0': ('pendulum',),
}

for env_name, (expert_name,) in ENVS.items():
    world = swm.World(
        env_name,
        num_envs=3,
        image_shape=(224, 224),
        max_episode_steps=500,
    )

    option_names = [
        name
        for name in world.single_variation_space.names()
        if 'color' in name
    ]

    world.set_policy(
        ExpertPolicy(
            ckpt_path=f'../stable-expert/models/sac_dmcontrol/{expert_name}/expert_policy.zip',
            vec_normalize_path=f'../stable-expert/models/sac_dmcontrol/{expert_name}/vec_normalize.pkl',
            noise_std=0.3,
            device='cuda',
        )
    )

    os.makedirs(f'./dmc/normal/{expert_name}_expert/', exist_ok=True)
    world.record_video(
        f'./dmc/normal/{expert_name}_expert/',
        max_steps=500,
        fps=24,
        extension='gif',
    )

    os.makedirs(f'./dmc/var/{expert_name}_expert/', exist_ok=True)
    world.record_video(
        f'./dmc/var/{expert_name}_expert/',
        max_steps=500,
        fps=24,
        extension='gif',
        options={'variation': option_names},
    )
