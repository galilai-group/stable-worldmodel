import xenoworlds

if __name__ == "__main__":

    def noise_fn():
        import numpy as np

        return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

    wrappers = [
        lambda x: xenoworlds.BackgroundDeform(
            x,
            image="https://cs.brown.edu/media/filer_public/ba/c4/bac4b1d3-99b3-4b07-b755-8664f7ca7e85/img-20240706-wa0029.jpg",
            noise_fn=noise_fn,
        ),
        lambda x: xenoworlds.ColorDeform(
            x,
            target=["agent", "goal", "block"],
            every_k_steps=-1,
        ),
        lambda x: xenoworlds.ShapeDeform(x, target=["agent", "block"], randomize=True),
        lambda x: xenoworlds.wrappers.RecordVideo(x, video_folder="./videos"),
        lambda x: xenoworlds.wrappers.AddRenderObservation(x, render_only=False),
        lambda x: xenoworlds.wrappers.TransformObservation(x),
    ]

    goal_wrappers = [
        lambda x: xenoworlds.wrappers.AddRenderObservation(x, render_only=False),
        lambda x: xenoworlds.wrappers.TransformObservation(x),
    ]

    world = xenoworlds.World(
        "xenoworlds/PushT-v1",
        num_envs=1,
        wrappers=wrappers,
        max_episode_steps=100,
        goal_wrappers=goal_wrappers,
    )

    world_model = xenoworlds.wm.DummyWorldModel(
        image_shape=(3, 224, 224), action_dim=world.single_action_space.shape[0]
    )

    # -- create a gradient descent solver
    action_space = world.action_space
    # solver = xenoworlds.solver.GDSolver(
    #     world_model, n_steps=100, action_space=action_space
    # )
    # planning_policy = xenoworlds.policy.PlanningPolicy(world, solver)
    random_policy = xenoworlds.policy.RandomPolicy(world)

    # -- run evaluation
    evaluator = xenoworlds.Evaluator(world, random_policy)
    data = evaluator.run(episodes=1)
    # data will be a dict with all the collected metrics
