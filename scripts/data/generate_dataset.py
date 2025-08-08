if __name__ == "__main__":
    import gymnasium as gym
    import xenoworlds

    # images = [
    #     xenoworlds.utils.create_pil_image_from_url(
    #         "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQK5OnlnP3_GHXI2y1LoIHbMROdN8_DYyLEGg&s"
    #     ).resize((64, 64)),
    #     xenoworlds.utils.create_pil_image_from_url(
    #         "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQjrFGrhOLwgYP0cdjTIBEWMpy9MHBcya4c5Q&s"
    #     ).resize((32, 32)),
    # ]

    wrappers = [
        lambda x: xenoworlds.wrappers.AddRenderObservation(x, render_only=False),
        lambda x: xenoworlds.wrappers.TimeLimit(x, max_episode_steps=200),
        # lambda x: xenoworlds.wrappers.TransformObservation(x),
    ]

    env = gym.make(
        "xenoworlds/PushT-v1",
        render_mode="rgb_array",
        resolution=224,
        with_velocity=True,
    )

    wrapped_env = env
    for wrapper in wrappers:
        wrapped_env = wrapper(wrapped_env)

    # data = xenoworlds.collect.optimal_action(wrapped_env, num_episodes=10, seed=42)
    data = xenoworlds.collect.random_action(wrapped_env, num_episodes=10, seed=42)
    print(data)
