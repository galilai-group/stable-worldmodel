"""Integration tests for checkpointing functionality."""


def test_env():
    import stable_worldmodel as swm

    world = swm.World(
        "swm/SimplePointMaze-v0",
        num_envs=5,
        image_shape=(224, 224),
        render_mode="rgb_array",
    )
    world.set_policy(swm.policy.RandomPolicy())
    world.policy.set_seed(42)
    world.record_dataset(
        "debug",
        episodes=10,
        seed=2547,
        options=dict(variation=("walls.number", "walls.shape", "walls.positions")),
    )

    world.policy.set_seed(42)
    world.record_video(
        "./",
        seed=2547,
        options=dict(variation=("walls.number", "walls.shape", "walls.positions")),
    )

    # asdf
    # world_model = swm.wm.DummyWorldModel(image_shape=(3, 224, 224), action_dim=8)
    # solver = None
    # world.set_policy(swm.policy.WorldModelPolicy(world_model, solver))
    # world.set_policy(swm.policy.RandomPolicy())
    # world.policy.set_seed(42)
    # world.record_dataset("./dataset", episodes=1, seed=2347)

    world.record_video_from_dataset(
        "./", "debug", episode_idx=[0, 1, 2, 3, 4, 5], fps=30, num_proc=1
    )

    # asdf
    # planner = CEMNevergrad(
    #     world_model, n_steps=100, action_space=world.action_space, planning_horizon=3
    # )
    # agent = swm.Agent(planner, world)
    # # 'FetchPush-v1'
    # agent.run(episodes=5)


if __name__ == "__main__":
    test_env()
