from minari import DataCollector


#### make one collect with policy param

### policy.get_action(obs)


def random_action(env, num_episodes=100, seed=None):
    env = DataCollector(env, record_infos=True)

    for step in range(num_episodes):
        env.reset(seed=seed or step)
        while True:
            action = env.action_space.sample()
            obs, rew, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
    dataset = env.create_dataset(
        dataset_id=env.unwrapped.spec.id,
        algorithm_name="Random-Policy",
        # code_permalink="https://github.com/Farama-Foundation/Minari",
        # author="Farama",
        # author_email="contact@farama.org",
    )


def optimal_action(env, num_episodes=100, seed=None):
    env = DataCollector(env, record_infos=True)

    for step in range(num_episodes):
        env.reset(seed=seed or step)
        while True:
            action = env.unwrapped._get_optimal_action()
            obs, rew, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

    dataset = env.create_dataset(
        dataset_id=env.unwrapped.spec.id,
        algorithm_name="Expert-Policy",
        # code_permalink="https://github.com/Farama-Foundation/Minari",
        # author="Farama",
        # author_email="contact@farama.org",
    )
