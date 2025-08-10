import os
os.environ["MUJOCO_GL"] = "egl"
import xenoworlds


if __name__ == "__main__":
    # run with MUJOCO_GL=egl python example.py

    # gym.register_envs(gymnasium_robotics)
    # envs = gym.envs.registry.keys()
    # print(envs)
    # asdf
    wrappers = [
        # lambda x: RecordVideo(x, video_folder="./videos"),
        # lambda x: xenoworlds.wrappers.AddRenderObservation(x, render_only=False),
        # lambda x: xenoworlds.wrappers.TransformObservation(x),
    ]
    world = xenoworlds.World(
        "xenoworlds/PushT-v1", num_envs=4, wrappers=wrappers, max_episode_steps=100
    )

    world_model = xenoworlds.DummyWorldModel(
        image_shape=(3, 224, 224), action_dim=world.single_action_space.shape[0]
    )

    # -- create a planning policy with a gradient descent solver
    # solver = xenoworlds.solver.GDSolver(world_model, n_steps=100, action_space=world.action_space)
    # policy = xenoworlds.policy.PlanningPolicy(world, solver)
    # -- create a random policy
    policy = xenoworlds.policy.RandomPolicy(world)

    # -- run evaluation
    evaluator = xenoworlds.evaluator.Evaluator(world, policy)
    data = evaluator.run(episodes=5, video_episodes=5)
    # data will be a dict with all the collected metrics
    
    # visualize a rollout video (e.g. for debugging purposes)
    xenoworlds.utils.save_rollout_videos(data["frames_list"])
