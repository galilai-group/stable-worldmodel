"""Integration tests for checkpointing functionality."""


def test_env():
    import stable_worldmodel as swm

    world = swm.World(
        "swm/VoidRun-v0",
        num_envs=5,
        image_shape=(224, 224),
        render_mode="rgb_array",
    )

    world.set_policy(swm.policy.RandomPolicy())
    world.policy.set_seed(3495)
    world.record_video("./", seed=324, options=None)  # dict(variation=("all",)))


if __name__ == "__main__":
    test_env()
