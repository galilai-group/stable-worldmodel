import os


os.environ["MUJOCO_GL"] = "egl"

if __name__ == "__main__":
    import stable_worldmodel as swm
    from stable_worldmodel.envs.pusht.collection_policy import PushTCollectionPolicy

    ######################
    ##  World Creation  ##
    ######################

    world = swm.World(
        "swm/PushT-v1",
        num_envs=10,
        image_shape=(224, 224),
        max_episode_steps=400,
        render_mode="rgb_array",
    )

    print("Available variations: ", world.single_variation_space.names())

    #######################
    ##  Data Collection  ##
    #######################

    world.set_policy(PushTCollectionPolicy(dist_constraint=100))

    world.record_video(
        "./",
        seed=2347,
        options={"variation": ("all",)},
    )
