if __name__ == "__main__":
    import stable_worldmodel as swm
    from stable_worldmodel.envs.simple_nav import ExpertPolicy

    ######################
    ##  World Creation  ##
    ######################

    world = swm.World(
        "swm/SimpleNavigation-v0",
        num_envs=10,
        image_shape=(224, 224),
        max_episode_steps=200,
        render_mode="rgb_array",
        size=9,
    )

    print("Available variations: ", world.single_variation_space.names())

    # #######################
    # ##  Data Collection  ##
    # #######################

    world.set_policy(ExpertPolicy())
    world.record_dataset(
        "example-nav",
        episodes=10,
        seed=2347,
        options=None,
    )

    dataset = swm.data.FrameDataset("example-nav")
    world.record_video_from_dataset(
        "./",
        dataset,
        episode_idx=[0, 1],
    )
