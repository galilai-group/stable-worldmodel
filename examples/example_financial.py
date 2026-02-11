"""Example demonstrating financial trading environment with backtesting.

IMPORTANT: This example requires you to register a financial dataset BEFORE running.
Without a registered dataset, you will get an error.

Example registration:
    from stable_worldmodel.envs import register_financial_dataset
    import pandas as pd

    def my_data_loader(symbol, start_date, end_date, **kwargs):
        # Load your data from CSV, database, API, etc.
        # Must return DataFrame with columns: open, high, low, close
        df = pd.read_csv(f'data/{symbol}.csv', parse_dates=['timestamp'])
        df = df.set_index('timestamp')
        mask = (df.index >= start_date) & (df.index <= end_date)
        return df[mask]

    register_financial_dataset(my_data_loader)

See the financial environment documentation for more details on dataset registration.
"""

if __name__ == "__main__":
    import stable_worldmodel as swm

    ######################
    ##  World Creation  ##
    ######################

    world = swm.World(
        "swm/Financial-v0",
        num_envs=2,
        render_mode=None,
    )

    print("Available variations: ", world.single_variation_space.names())

    #######################
    ##  Random Trading   ##
    #######################

    # Set a random policy
    world.set_policy(swm.policy.RandomPolicy())

    # Run a few episodes manually
    world.reset(seed=42)

    for episode in range(2):
        world.reset(seed=42 + episode)

        for step in range(50):  # Run 50 steps per episode
            actions = world.policy.get_action(world.states)
            world.states, rewards, terminated, truncated, world.infos = world.envs.step(actions)

            if all(terminated) or all(truncated):
                break

    #############################
    ##  Backtest Analysis      ##
    #############################

    # Access the first environment from the world's vectorized environments
    env = world.envs.envs[0]
    env.reset(seed=42)

    # Run a short episode
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    # Get comprehensive backtest results
    backtest_results = env.unwrapped.get_backtest_results()

    print("\nBacktest Results:")
    print(backtest_results)
