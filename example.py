if __name__ == "__main__":
    import stable_worldmodel as swm
    import torch

    # create world
    world = swm.World(
        "swm/SimplePointMaze-v0",
        num_envs=7,
        image_shape=(224, 224),
        render_mode="rgb_array",
    )

    print("Available variations: ", world.single_variation_space.names())

    # collect data for pre-training
    # world.set_policy(swm.policy.RandomPolicy())
    # world.record_dataset(
    #     "simple-pointmaze",
    #     episodes=10,
    #     seed=2347,
    #     options=dict(variation=("walls.number", "walls.shape", "walls.positions")),
    # )
    # world.record_video(
    #     "./",
    #     seed=2347,
    #     options=dict(variation=("walls.number", "walls.shape", "walls.positions")),
    # )

    # pre-train world model
    # swm.pretraining(
    #     "scripts/train/dummy.py",
    #     "++dump_object=True dataset_name=simple-pointmaze output_model_name=dummy_test",
    # )

    # evaluate world model
    # world.set_policy(swm.policy.AutoPolicy("output_model_name"))

    spt_module = torch.load(
        swm.utils.get_cache_dir() + "/dummy_test_object.ckpt", weights_only=False
    )

    model = spt_module.model
    config = swm.PlanConfig(horizon=10, receding_horizon=5)
    # solver = swm.solver.RandomSolver()
    solver = swm.solver.GDSolver(model, n_steps=10)
    policy = swm.policy.WorldModelPolicy(solver=solver, config=config)
    world.set_policy(policy)
    results = world.evaluate(episodes=2, seed=2347)  # , options={...})

    # what about eval on all type of env?
    # TODO: add leaderboard

    print(results)
