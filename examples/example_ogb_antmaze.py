import os


os.environ["MUJOCO_GL"] = "egl"

if __name__ == "__main__":
    import stable_worldmodel as swm

    ######################
    ##  World Creation  ##
    ######################

    world = swm.World(
        "swm/OGBAntMaze-Inside-v0",
        num_envs=5,
        image_shape=(224, 224),
        max_episode_steps=200,
        width=224,
        height=224,
    )

    # print("Available variations: ", world.single_variation_space.names())

    #######################
    ##  Data Collection  ##
    #######################

    world.set_policy(swm.policy.RandomPolicy())
    # world.record_dataset(
    #     "ogbench-cube-single",
    #     episodes=10,
    #     seed=2347,
    # )

    world.record_video("./", seed=2347, options={"variation": ("camera",)})

    # world.record_video_from_dataset(
    #     "./",
    #     "ogbench-cube-single",
    #     episode_idx=[0, 1, 2, 3, 4],
    # )
