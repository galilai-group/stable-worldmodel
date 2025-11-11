if __name__ == "__main__":
    import datasets
    import stable_pretraining as spt
    import torch
    from sklearn import preprocessing
    from torchvision.transforms import v2 as transforms

    import stable_worldmodel as swm

    ######################
    ##  World Creation  ##
    ######################

    world = swm.World(
        "swm/PushT-v1",
        num_envs=1,
        image_shape=(224, 224),
        max_episode_steps=50,
        render_mode="rgb_array",
        n_stacks=3,
    )

    #########################
    ##  Transform/Process  ##
    #########################

    def img_transform():
        transform = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(**spt.data.dataset_stats.ImageNet),
                transforms.Resize(size=196),
                transforms.CenterCrop(size=196),
            ]
        )
        return transform

    transform = {
        "pixels": img_transform(),
        "goal": img_transform(),
    }

    dataset_path = swm.data.get_cache_dir() / "pusht_expert_train"
    dataset = datasets.load_from_disk(dataset_path).with_format("numpy")

    action_process = preprocessing.StandardScaler()
    action_process.fit(dataset["action"][:])

    proprio_process = preprocessing.StandardScaler()
    proprio_process.fit(dataset["proprio"][:])

    process = {
        "action": action_process,
        "proprio": proprio_process,
        "goal_proprio": proprio_process,
    }

    ################
    ##  Evaluate  ##
    ################

    model = swm.policy.AutoCostModel("pyro_test_epoch_40").to("cuda")
    model = model.eval()
    model.requires_grad_(False)
    config = swm.PlanConfig(horizon=5, receding_horizon=5, action_block=5)
    solver = swm.solver.CEMSolver(model, num_samples=300, var_scale=1.0, n_steps=30, topk=30, device="cuda")
    policy = swm.policy.WorldModelPolicy(solver=solver, config=config, process=process, transform=transform)

    world.set_policy(policy)

    # results = world.evaluate(episodes=20, seed=42, dump_every=10)
    results = world.evaluate_from_dataset(
        "pusht_expert_val",
        start_steps=[25],
        episodes_idx=[3],
        goal_offset_steps=25,
        eval_budget=25,
        callables={"_set_state": "state", "_set_goal_state": "goal_state"},
    )

    print("Evaluation results: ", results)
