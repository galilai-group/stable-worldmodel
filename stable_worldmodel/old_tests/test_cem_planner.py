def test_each_env():
    import stable_worldmodel as swm
    from stable_worldmodel.planner import CEMNevergrad

    # run with MUJOCO_GL=egl python example.py

    # gym.register_envs(gymnasium_robotics)

    world_model = swm.DummyWorldModel(image_shape=(3, 224, 224), action_dim=8)

    # print(sorted(envs))
    wrappers = [
        # lambda x: RecordVideo(x, video_folder="./videos"),
        lambda x: swm.wrappers.AddRenderObservation(x, render_only=False),
        lambda x: swm.wrappers.TransformObservation(x),
    ]
    world = swm.World("AntMaze_Medium", num_envs=4, wrappers=wrappers)
    planner = CEMNevergrad(world_model, n_steps=100, action_space=world.action_space, planning_horizon=3)
    agent = swm.Agent(planner, world)
    # 'FetchPush-v1'
    agent.run(episodes=5)
