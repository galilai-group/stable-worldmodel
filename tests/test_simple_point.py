def test_env():
    import gymnasium as gym
    import gymnasium_robotics
    import torch
    import xenoworlds

    # # run with MUJOCO_GL=egl python example.py

    world = xenoworlds.World(
        "xenoworlds/SimplePointMaze-v0",
        num_envs=4,
        image_shape=(224, 224),
        render_mode="rgb_array",
    )
    world.record_video("./video.mp4")
    asdf
    world_model = xenoworlds.DummyWorldModel(image_shape=(3, 224, 224), action_dim=8)

    asdf
    planner = CEMNevergrad(
        world_model, n_steps=100, action_space=world.action_space, planning_horizon=3
    )
    agent = xenoworlds.Agent(planner, world)
    # 'FetchPush-v1'
    agent.run(episodes=5)


if __name__ == "__main__":
    test_env()
