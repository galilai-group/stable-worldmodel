import stable_worldmodel as swm
from stable_worldmodel.envs.pusht import WeakPolicy


w = swm.World('swm/PushT-v1', num_envs=8, image_shape=(64, 64), render_mode='rgb_array')
w.set_policy(WeakPolicy(dist_constraint=100))
w.collect('data/pusht_genie.lance', episodes=2000, seed=0)
