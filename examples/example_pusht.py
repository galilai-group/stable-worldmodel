if __name__ == "__main__":
    import datasets
    import torch
    from sklearn import preprocessing
    from torchvision.transforms import v2 as transforms

    import stable_worldmodel as swm

    ######################
    ##  World Creation  ##
    ######################

    world = swm.World(
        "swm/PushT-v1",
        num_envs=10,
        image_shape=(224, 224),
        max_episode_steps=200,
        render_mode="rgb_array",
    )

    # print("Available variations: ", world.single_variation_space.names())

    # #######################
    # ##  Data Collection  ##
    # #######################

    # world.set_policy(swm.policy.RandomPolicy())
    # world.record_dataset(
    #     "example-pusht",
    #     episodes=10,
    #     seed=2347,
    #     options=None,
    # )

    # world.record_video_from_dataset(
    #     "./",
    #     "example-pusht",
    #     episode_idx=[0, 1],
    # )

    ################
    ##  Pretrain  ##
    ################

    # swm.pretraining(
    #     "scripts/train/dinowm.py",
    #     dataset_name="example-pusht",
    #     output_model_name="dummy_pusht",
    #     dump_object=True,
    # )

    #########################
    ##  Transform/Process  ##
    #########################

    def img_transform():
        return transforms.Compose(
            [
                transforms.ToImage(),
                transforms.Lambda(lambda img: torch.tensor(img)),
                transforms.Lambda(lambda img: img / 255.0),
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    transform = {
        "pixels": img_transform(),
        "goal": img_transform(),
    }

    import numpy as np

    dataset_path = swm.data.utils.get_cache_dir() / "pusht_expert_train"
    dataset = datasets.load_from_disk(dataset_path).with_format("numpy")

    action_process = preprocessing.StandardScaler()
    # action_process.fit(dataset["action"][:])

    print(vars(action_process))

    action_process.mean_ = np.array([-0.0087, 0.0068])
    action_process.scale_ = np.array([0.2019, 0.2002])

    proprio_process = preprocessing.StandardScaler()
    # proprio_process.fit(dataset["proprio"][:])

    proprio_process.mean_ = np.array([236.6155, 264.5674, -2.93032027, 2.54307914])
    proprio_process.scale_ = np.array([101.1202, 87.0112, 74.84556075, 74.14009094])

    process = {
        "action": action_process,
        "proprio": proprio_process,
        "goal_proprio": proprio_process,
    }

    ################
    ##  Evaluate  ##
    ################

    model = swm.policy.AutoCostModel("dinowm_pusht").to("cuda")
    config = swm.PlanConfig(horizon=5, receding_horizon=5, action_block=5)
    solver = swm.solver.CEMSolver(model, num_samples=300, var_scale=1.0, n_steps=30, topk=30, device="cuda")
    policy = swm.policy.WorldModelPolicy(solver=solver, config=config, process=process, transform=transform)

    torch.hub._validate_not_a_forked_repo = lambda a, b, c: True

    class DinoV2Encoder(torch.nn.Module):
        def __init__(self, name, feature_key):
            super().__init__()
            self.name = name
            self.base_model = torch.hub.load("facebookresearch/dinov2", name)
            self.feature_key = feature_key
            self.emb_dim = self.base_model.num_features
            if feature_key == "x_norm_patchtokens":
                self.latent_ndim = 2
            elif feature_key == "x_norm_clstoken":
                self.latent_ndim = 1
            else:
                raise ValueError(f"Invalid feature key: {feature_key}")

            self.patch_size = self.base_model.patch_size

        def forward(self, x):
            emb = self.base_model.forward_features(x)[self.feature_key]
            if self.latent_ndim == 1:
                emb = emb.unsqueeze(1)  # dummy patch dim
            return emb

    model.backbone = DinoV2Encoder("dinov2_vits14", feature_key="x_norm_patchtokens").to("cuda")
    ckpt = torch.load(swm.data.utils.get_cache_dir() / "dinowm_pusht_weights.ckpt")

    model.predictor.load_state_dict(ckpt["predictor"], strict=False)
    model.action_encoder.load_state_dict(ckpt["action_encoder"])
    model.proprio_encoder.load_state_dict(ckpt["proprio_encoder"])

    model = model.to("cuda")
    model = model.eval()

    print(ckpt.keys())

    world.set_policy(policy)
    # sample 50 episodes idx

    # fix random seed
    np.random.seed(42)
    episode_idx = np.random.choice(10000, size=10, replace=False).tolist()
    start_steps = np.random.randint(0, 60, size=10).tolist()

    print("Evaluating episodes: ", episode_idx)
    print("Starting steps: ", start_steps)

    dataset = swm.data.FrameDataset("pusht_expert_train")
    results = world.evaluate_from_dataset(
        dataset,
        start_steps=start_steps,
        episodes_idx=episode_idx,
        goal_offset_steps=25,
        eval_budget=25,
        callables={"_set_state": "state", "_set_goal_state": "goal_state"},
    )

    print("Evaluation results: ", results)
