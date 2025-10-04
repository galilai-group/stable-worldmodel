def test_env():
    import stable_worldmodel as swm

    world = swm.World(
        "swm/PushT-v1",
        num_envs=5,
        image_shape=(224, 224),
        render_mode="rgb_array",
    )
    world.set_policy(swm.policy.RandomPolicy())
    world.policy.set_seed(42)
    world.record_dataset("debug-pusht", episodes=10, seed=2547, options={"variation": ["all"]})

    world.policy.set_seed(42)
    world.record_video(
        "./",
        seed=2547,
        options={"variation": ["all"]},
    )

    world.record_video_from_dataset("./", "debug-pusht", episode_idx=[0, 1, 2, 3, 4, 5], fps=30, num_proc=1)


if __name__ == "__main__":
    test_env()
