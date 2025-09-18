if __name__ == "__main__":
    import xenoworlds as swm
    #import stable_worldmodel as swm

    # create world
    world = swm.World(
        "xenoworlds/SimplePointMaze-v0",
        num_envs=7,
        image_shape=(224, 224),
        render_mode="rgb_array",
    )

    # collect data for pre-training
    # world.set_policy(swm.policy.RandomPolicy())
    # world.policy.set_seed(42)
    # world.record_dataset("./dataset", episodes=10, seed=2347)
    # world.record_video("./", seed=2347)
    
    # pre-train world model
    action_dim = world.envs.single_action_space.shape[0]
    world_model = swm.wm.DummyWorldModel((224, 224, 3), action_dim)
    solver = swm.solver.RandomSolver(...)
    policy = swm.policy.WorldModelPolicy(world_model, solver, horizon=10, action_block=5, receding_horizon=5)
    world.set_policy(swm.policy.WorldModelPolicy(world_model))

    swm.pretraining("scripts/tdmpc.py", "dataset_name", "output_model_name") # + save ckpt etc

    # evaluate world model
    world.set_policy(swm.AutoPolicy("output_model_name"))
    world.evaluate(episodes=10, seed=2347, options={...})

