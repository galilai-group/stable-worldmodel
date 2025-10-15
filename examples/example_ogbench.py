import os


os.environ["MUJOCO_GL"] = "egl"

if __name__ == "__main__":
    import stable_worldmodel as swm

    ######################
    ##  World Creation  ##
    ######################

    world = swm.World(
        "swm/OGBCube-v0",
        num_envs=1,
        image_shape=(224, 224),
        max_episode_steps=200,
        env_type="single",
        multiview=False,
        ob_type="pixels",
        width=224,
        height=224,
        visualize_info=False,
    )

    print("Available variations: ", world.single_variation_space.names())

    #######################
    ##  Data Collection  ##
    #######################

    world.set_policy(swm.policy.RandomPolicy())
    # world.record_dataset(
    #     "ogbench-cube-single",
    #     episodes=10,
    #     seed=2347,
    #     options={"variation": ("cube.color", "cube.size", "agent.color", "floor.color")},
    # )
    world.record_video(
        "./",
        seed=2347,
        options={
            "variation": (
                "cube.color",
                "cube.size",
                "agent.color",
                "floor.color",
                "camera.angle_delta",
                "light.intensity",
            )
        },
    )
    # exit()

    ################
    ##  Pretrain  ##
    ################

    swm.pretraining(
        "scripts/train/dinowm.py",
        dataset_name="ogbench-cube-single",
        output_model_name="dummy_ogcube",
        dump_object=True,
    )

    ################
    ##  Evaluate  ##
    ################

    # NOTE for user: make sure to match action_block with the one used during training!

    model = swm.policy.AutoCostModel("dummy_ogcube").to("cuda")
    config = swm.PlanConfig(horizon=5, receding_horizon=5, action_block=5)
    solver = swm.solver.CEMSolver(model, num_samples=300, var_scale=1.0, n_steps=30, topk=30, device="cuda")
    policy = swm.policy.WorldModelPolicy(solver=solver, config=config)

    world.set_policy(policy)
    results = world.evaluate(episodes=5, seed=2347)

    print(results)
