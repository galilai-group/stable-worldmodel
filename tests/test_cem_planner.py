def test_each_env():
    import gymnasium as gym
    import gymnasium_robotics
    import torch
    import xenoworlds
    from xenoworlds.planner import GD, CEMNevergrad

    # run with MUJOCO_GL=egl python example.py

    # gym.register_envs(gymnasium_robotics)
    envs = gym.envs.registry.keys()

    world_model = xenoworlds.DummyWorldModel(image_shape=(3, 224, 224), action_dim=8)

    # print(sorted(envs))
    wrappers = [
        # lambda x: RecordVideo(x, video_folder="./videos"),
        lambda x: xenoworlds.wrappers.AddRenderObservation(x, render_only=False),
        lambda x: xenoworlds.wrappers.TransformObservation(x),
    ]
    world = xenoworlds.World("AntMaze_Medium", num_envs=4, wrappers=wrappers)
    planner = CEMNevergrad(
        world_model, n_steps=100, action_space=world.action_space, planning_horizon=3
    )
    agent = xenoworlds.Agent(planner, world)
    # 'FetchPush-v1'
    agent.run(episodes=5)
