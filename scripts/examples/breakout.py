from pathlib import Path

import stable_worldmodel as swm
from stable_worldmodel.policy import RandomPolicy


VIDEO_DIR = Path(__file__).parent / 'videos' / 'breakout'

world = swm.World(
    'ALE/Breakout-v5',
    num_envs=1,
    image_shape=(224, 160),
    max_episode_steps=1000,
    goal_conditioned=False,
    render_mode='rgb_array',
)
world.set_policy(RandomPolicy(seed=0))

world.evaluate(episodes=1, seed=0, video=VIDEO_DIR)
