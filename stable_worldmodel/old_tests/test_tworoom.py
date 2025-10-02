"""Integration tests for checkpointing functionality."""


def test_env():
    import stable_worldmodel as swm

    world = swm.World(
        "swm/TwoRoom-v0",
        num_envs=25,
        image_shape=(224, 224),
        render_mode="rgb_array",
    )
    world.set_policy(swm.policy.RandomPolicy())
    world.policy.set_seed(42)
    world.record_video(
        "./",
        seed=2547,
        options=dict(),
    )


if __name__ == "__main__":
    test_env()
