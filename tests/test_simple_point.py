def test_env():
    import xenoworlds

    world = xenoworlds.World(
        "xenoworlds/SimplePointMaze-v0",
        num_envs=4,
        image_shape=(224, 224),
        goal_shape=(224, 224),
        render_mode="rgb_array",
    )
    world.set_policy(xenoworlds.policy.RandomPolicy())
    world.policy.set_seed(42)
    world.record_video("./", seed=2347)

    # asdf
    #world_model = xenoworlds.wm.DummyWorldModel(image_shape=(3, 224, 224), action_dim=8)
    #solver = None
    #world.set_policy(xenoworlds.policy.WorldModelPolicy(world_model, solver))
    world.set_policy(xenoworlds.policy.RandomPolicy())
    world.policy.set_seed(42)
    world.record_dataset("./dataset", episodes=1, seed=2347)

    world.record_video_from_dataset(
        "./", "./dataset", episode_idx=0, fps=30, num_proc=1
    )

    # asdf
    # planner = CEMNevergrad(
    #     world_model, n_steps=100, action_space=world.action_space, planning_horizon=3
    # )
    # agent = xenoworlds.Agent(planner, world)
    # # 'FetchPush-v1'
    # agent.run(episodes=5)


if __name__ == "__main__":
    test_env()
