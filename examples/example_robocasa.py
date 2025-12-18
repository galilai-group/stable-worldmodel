if __name__ == "__main__":
    import os

    import imageio
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from scipy.spatial.transform import Rotation as R

    import stable_worldmodel as swm

    ######################
    ##  World Creation  ##
    ######################

    # Create a RoboCasa world with parallel environments
    # The RoboCasa-v0 environment creates a default PnPCounterToSink task
    print("Creating RoboCasa environment...")

    cfg = {
        "task_specification": {
            "env": {
                "custom_task": False,
            }
        }
    }
    cfg = OmegaConf.create(cfg)

    NUM_EVAL_ENVS = 1
    world = swm.World(
        "swm/RoboCasa-v0",  # Base RoboCasa environment
        num_envs=NUM_EVAL_ENVS,  # Number of parallel environments (start with fewer for testing)
        image_shape=(224, 224),  # Image resolution
        max_episode_steps=75,  # Maximum steps per episode
        render_mode="rgb_array",
        cfg=cfg,
        goal_conditioned=False,  # RoboCasa doesn't provide goals in info dict
    )

    print("RoboCasa environment created successfully!\n")

    #######################################
    ##  Test Goal Sampling               ##
    #######################################

    print("=" * 50)
    print("Testing Goal Sampling")
    print("=" * 50)

    # Reset with goal sampling enabled
    print("\nResetting environment with goal sampling enabled...")
    world.reset(
        seed=42,
        options={
            "sample_goal": True,  # Enable goal sampling
            "goal_cube_size": 0.15,  # Half-size of sampling cube (meters)
            "goal_max_attempts": 10,  # Max IK attempts
            "goal_tolerance": 0.02,  # Position tolerance (meters)
        },
    )

    # Check if goal sampling was successful
    goal_image = world.infos["goal"][0]
    goal_eef_pos = world.infos.get("goal_eef_pos", [None])[0]
    goal_success = world.infos.get("goal_sampling_success", [None])[0]

    print("\nGoal sampling results:")
    print(f"  - Goal image shape: {goal_image.shape}")
    print(f"  - Goal EEF position: {goal_eef_pos}")
    print(f"  - Sampling success: {goal_success}")

    # Save the goal image for visualization
    goal_output_dir = os.path.join(os.path.dirname(__file__), "..", "local_scripts", "plots")
    os.makedirs(goal_output_dir, exist_ok=True)

    # Save current observation and goal side by side
    current_pixels = np.squeeze(world.infos["pixels"][0])
    goal_pixels = np.squeeze(goal_image)

    # Create side-by-side comparison
    comparison = np.concatenate([current_pixels, goal_pixels], axis=1)
    comparison_path = os.path.join(goal_output_dir, "robocasa_goal_comparison.png")
    imageio.imwrite(comparison_path, comparison)
    print(f"\nSaved goal comparison image to: {os.path.abspath(comparison_path)}")
    print("  Left: Current state | Right: Sampled goal state")

    # Also save them separately
    current_path = os.path.join(goal_output_dir, "robocasa_current_state.png")
    goal_path = os.path.join(goal_output_dir, "robocasa_goal_state.png")
    imageio.imwrite(current_path, current_pixels)
    imageio.imwrite(goal_path, goal_pixels)
    print(f"  Current state: {os.path.abspath(current_path)}")
    print(f"  Goal state: {os.path.abspath(goal_path)}")

    print("\n" + "=" * 50)
    print("Goal Sampling Test Complete")
    print("=" * 50 + "\n")

    #######################################
    ##  Dataset Transforms               ##
    #######################################

    def robocasa_dataset_transform(sample):
        """Transform RoboCasa dataset samples to match environment format.

        Applies:
        1. Proprio: 9-dim (raw) → 7-dim (environment format)
        2. Pixels: Resize 128→224 and ensure CHW format for world.py evaluate_from_dataset
        """
        from torchvision.transforms.functional import resize

        # Transform proprio from 9-dim to 7-dim
        if "proprio" in sample:
            proprio = sample["proprio"]

            # Handle both single samples and batches
            if isinstance(proprio, torch.Tensor):
                proprio = proprio.numpy()

            # Check if this is 9-dim (raw format) or already 7-dim
            if proprio.shape[-1] == 9:
                # Extract components
                eef_pos = proprio[..., :3]  # (T, 3)
                eef_quat = proprio[..., 3:7]  # (T, 4) in [w, x, y, z] format
                gripper_qpos = proprio[..., 7:9]  # (T, 2)

                # Convert quaternion to Euler angles
                orig_shape = eef_quat.shape[:-1]
                eef_quat_flat = eef_quat.reshape(-1, 4)

                # Convert from [w, x, y, z] to [x, y, z, w] for scipy
                eef_quat_xyzw = eef_quat_flat[:, [1, 2, 3, 0]]
                eef_euler_flat = R.from_quat(eef_quat_xyzw).as_euler("xyz", degrees=False)
                eef_euler = eef_euler_flat.reshape(*orig_shape, 3).astype(np.float32)

                # Convert gripper 2D to 1D
                gripper_1d = (gripper_qpos[..., 0:1] - gripper_qpos[..., 1:2]).astype(np.float32)

                # Concatenate to 7-dim format
                proprio_7d = np.concatenate([eef_pos, eef_euler, gripper_1d], axis=-1)
                sample["proprio"] = torch.tensor(proprio_7d)

        # Process pixels:
        # 1. Resize from 128x128 to 224x224 (dataset uses im128 files)
        # 2. Convert from HWC to CHW format for world.py evaluate_from_dataset
        # world.py expects (T, C, H, W) and will permute to (T, H, W, C)
        if "pixels" in sample:
            pixels = sample["pixels"]

            if isinstance(pixels, list):
                # List of tensors - resize and convert each from HWC to CHW
                converted_frames = []
                for frame in pixels:
                    if isinstance(frame, torch.Tensor):
                        # Frame is HWC (H, W, C), convert to CHW for resize
                        if frame.shape[-1] == 3:
                            frame = frame.permute(2, 0, 1)  # HWC -> CHW
                        # Resize if needed (128 -> 224)
                        if frame.shape[-1] == 128 or frame.shape[-2] == 128:
                            frame = resize(frame, [224, 224], antialias=True)
                        # Keep in CHW format for world.py
                    converted_frames.append(frame)
                sample["pixels"] = converted_frames

            elif isinstance(pixels, torch.Tensor):
                # Stacked tensor - check format and resize
                # Input could be (T, H, W, C) or (T, C, H, W)
                if pixels.shape[-1] == 3:
                    # HWC format (T, H, W, C) -> convert to CHW (T, C, H, W)
                    pixels = pixels.permute(0, 3, 1, 2)

                # Now in CHW format (T, C, H, W), resize if needed
                if pixels.shape[-1] == 128 or pixels.shape[-2] == 128:
                    # Resize each frame: (T, C, H, W)
                    pixels = resize(pixels, [224, 224], antialias=True)

                sample["pixels"] = pixels

        return sample

    #######################################
    ##  Load Converted RoboCasa Dataset  ##
    #######################################

    # Path to the converted dataset
    DATASET_PATH = os.path.join(os.environ.get("STABLEWM_HOME"), "robocasa/robocasa_pnp_subset")

    print(f"Loading converted RoboCasa dataset from: {DATASET_PATH}...")

    try:
        # Load the converted dataset using VideoDataset with transforms
        dataset = swm.data.VideoDataset(
            name="",  # Not used when cache_dir is the full path
            cache_dir=DATASET_PATH,
            frameskip=1,
            num_steps=1,
            decode_columns=["pixels"],
            transform=robocasa_dataset_transform,  # Apply proprio + image transforms at load time
        )
        print(f"Loaded dataset with {len(dataset.episodes)} episodes")
        print(f"Total indexable samples: {len(dataset)}")
        DATASET_AVAILABLE = True
    except Exception as e:
        print(f"Could not load dataset: {e}")
        print("\nTo create the dataset, run the conversion script first:")
        print("  python scripts/convert_robocasa_hdf5.py \\")
        print("      --data_path /checkpoint/amaia/video/basileterv/data/robocasa/v0.1 \\")
        print(f"      --output_dir {DATASET_PATH} \\")
        print("      --task_names PnPCounterToCab \\")
        print("      --filter_first_episodes 5 \\")
        print("      --mode video")
        print("\nSkipping dataset-based evaluation.")
        DATASET_AVAILABLE = False
        dataset = None

    ######################
    ##  Random Policy  ##
    ######################

    # Run random policy evaluation first to test goal sampling logic
    print("\n\nTesting RoboCasa environment with random policy (no dataset)...")

    world.set_policy(swm.policy.RandomPolicy())

    # Run a few episodes
    results = world.evaluate(episodes=2, seed=42)

    print("\n=== Random Policy Results ===")
    print(f"Success rate: {results.get('success_rate', 0):.2f}%")
    print(f"Episode successes: {results.get('episode_successes', [])}")

    ################
    ##  Evaluate  ##
    ################

    if DATASET_AVAILABLE and dataset is not None:
        # For testing, use random policy
        policy = swm.policy.RandomPolicy()
        world.set_policy(policy)

        # Select episodes to evaluate
        np.random.seed(42)
        n_episodes = min(NUM_EVAL_ENVS, len(dataset.episodes))
        episode_idx = np.random.choice(len(dataset.episodes), size=n_episodes, replace=False).tolist()

        # Determine valid start steps (ensuring enough room for goal_offset)
        goal_offset_steps = 25
        eval_budget = 50
        start_steps = []
        for ep in episode_idx:
            ep_len = len(dataset.episode_indices[ep])
            max_start = max(0, ep_len - goal_offset_steps - 1)
            start_step = np.random.randint(0, max(1, max_start))
            start_steps.append(start_step)

        print(f"\nEvaluating {n_episodes} episodes from dataset:")
        print(f"  Episode indices: {episode_idx}")
        print(f"  Start steps: {start_steps}")
        print(f"  Goal offset: {goal_offset_steps} steps")
        print(f"  Eval budget: {eval_budget} steps")

        # Evaluate from dataset
        results = world.evaluate_from_dataset(
            dataset,
            start_steps=start_steps,
            episodes_idx=episode_idx,
            goal_offset_steps=goal_offset_steps,
            eval_budget=eval_budget,
            callables={
                "prepare": "state",
                "_set_goal_state": "goal_state",
            },
        )

        print("\n=== Dataset Evaluation Results ===")
        print(f"Success rate: {results['success_rate']:.2f}%")
        print(f"Episode successes: {results['episode_successes']}")

    ######################
    ##  Save GIF Video  ##
    ######################

    print("\n\nRecording evaluation episode for GIF...")

    # Reset the world to start a new episode
    world.reset(seed=123)

    # Collect frames during evaluation
    frames = []
    max_steps = 75

    for step in range(max_steps):
        # Capture the current frame from the environment infos
        # Squeeze out extra dimensions to get (H, W, C) format
        frame = np.squeeze(world.infos["pixels"][0])
        frames.append(frame)

        # Take a step
        world.step()

        # Check if episode is done
        if world.terminateds[0] or world.truncateds[0]:
            # Capture final frame
            final_frame = np.squeeze(world.infos["pixels"][0])
            frames.append(final_frame)
            print(f"Episode ended at step {step + 1}")
            break

    # Save frames as GIF
    gif_output_dir = os.path.join(os.path.dirname(__file__), "..", "local_scripts", "plots")
    os.makedirs(gif_output_dir, exist_ok=True)
    gif_path = os.path.join(gif_output_dir, "robocasa_evaluation.gif")

    print(f"Saving {len(frames)} frames to GIF: {gif_path}")
    imageio.mimsave(gif_path, frames, fps=10, loop=0)

    print(f"GIF saved successfully to: {os.path.abspath(gif_path)}")
    print("\nRoboCasa environment test completed!")
