import stable_worldmodel as swm

world = swm.World(env_name='swm/PushT-v1', num_envs=5, image_shape=(224, 224))
world.set_policy(swm.policy.RandomPolicy())

world.record_dataset(
    'pusht_debug',
    episodes=10,
    options={
        'variation': ('agent.color',),
    },
)
