import os


os.environ["MUJOCO_GL"] = "egl"

if __name__ == "__main__":
    import stable_worldmodel as swm
    from stable_worldmodel.envs.ogbench_manip.collection_policy import OGBCollectionPolicy

    ######################
    ##  World Creation  ##
    ######################

    world = swm.World(
        "swm/OGBCube-v0",
        # "swm/OGBScene-v0",
        num_envs=5,
        image_shape=(224, 224),
        max_episode_steps=400,
        env_type="single",
        ob_type="pixels",
        multiview=True,
        width=224,
        height=224,
        visualize_info=False,
        terminate_at_goal=False,
        mode="data_collection",
    )

    print("Available variations: ", world.single_variation_space.names())

    #######################
    ##  Data Collection  ##
    #######################

    world.set_policy(OGBCollectionPolicy(policy_type="plan_oracle"))  # replace with your expert policy

    world.record_video(
        "./",
        seed=2347,
        options={"variation": ("all",)},
        viewname="pixels.front_pixels",
    )
