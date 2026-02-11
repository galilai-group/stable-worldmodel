"""Example demonstrating financial trading environment with backtesting.

NOTE: This example requires a valid registered dataset.

You can set these in a .env file in the project root or export them in your shell.
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
