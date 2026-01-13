if __name__ == "__main__":
    import stable_worldmodel as swm

    ######################
    ##  World Creation  ##
    ######################

    world = swm.World(
        "swm/PFRocketLanding-v0",
        num_envs=1,
        image_shape=(224, 224),
        max_episode_steps=5000,
        render_mode="rgb_array",
    )

    print("Available variations: ", world.single_variation_space.names())

    # #######################
    # ##  Data Collection  ##
    # #######################
    from stable_worldmodel.envs.rocket_landing import ExpertPolicy

    world.set_policy(ExpertPolicy())
    world.record_dataset("example-pfrl", episodes=2, seed=2347, options={"variation": ("all",)})
    world.record_video_from_dataset("./", "example-pfrl", episode_idx=[0, 1])

    # ################
    # ##  Evaluate  ##
    # ################

    # world.set_policy(swm.policy.RandomPolicy())
    # results = world.evaluate(episodes=3, seed=2347)

    # print(results)
