if __name__ == "__main__":
    import stable_worldmodel as swm

    ######################
    ##  World Creation  ##
    ######################

    world = swm.World(
        "swm/PFRocketLanding-v0",
        num_envs=2,
        image_shape=(224, 224),
        max_episode_steps=1000,
        render_mode="rgb_array",
    )

    print("Available variations: ", world.single_variation_space.names())

    # #######################
    # ##  Data Collection  ##
    # #######################

    world.set_policy(swm.policy.RandomPolicy())
    world.record_dataset(
        "example-pfrl",
        episodes=2,
        seed=2347,
        options={"variation": ("all",)},
    )

    world.record_video_from_dataset(
        "./",
        "example-pfrl",
        episode_idx=[0, 1],
    )

    # ################
    # ##  Evaluate  ##
    # ################

    # world.set_policy(swm.policy.RandomPolicy())
    # results = world.evaluate(episodes=3, seed=2347)

    # print(results)
