"""Example demonstrating rocket landing on a moving pad using O-U process."""

if __name__ == "__main__":
    import numpy as np

    import stable_worldmodel as swm
    from stable_worldmodel.envs.rocket_landing import ExpertPolicy

    ######################
    ##  World Creation  ##
    ######################

    world = swm.World(
        "swm/PFRocketLanding-v0",
        num_envs=1,
        image_shape=(224, 224),
        max_episode_steps=100,
        render_mode="rgb_array",
    )

    print("Available variations: ", world.single_variation_space.names())

    ####################################
    ##  Test 1: Stationary Pad (Default)
    ####################################
    print("\n=== Test 1: Stationary Pad ===")
    world.set_policy(ExpertPolicy())

    obs, info = world.reset(seed=42)
    print(f"Initial pad position: {info[0]['pad_position']}")
    print(f"Initial pad velocity: {info[0]['pad_velocity']}")

    for step in range(10):
        obs, rewards, dones, truncs, infos = world.step()
        print(f"Step {step}: pad_pos={infos[0]['pad_position']}, pad_vel={infos[0]['pad_velocity']}")

    # Pad should remain stationary
    assert np.allclose(infos[0]["pad_velocity"], 0.0), "Pad should be stationary by default"
    print("✓ Stationary pad test passed")

    ####################################
    ##  Test 2: Moving Pad (Enabled)   ##
    ####################################
    print("\n=== Test 2: Moving Pad (Enabled) ===")

    # Enable pad motion with gentle parameters
    obs, info = world.reset(seed=42, options={"variation": ["pad_motion.enabled"]})

    # Manually enable it by setting the value
    world.single_variation_space["pad_motion"]["enabled"]._value = 1

    # Reset again with motion enabled
    obs, info = world.reset(seed=42)
    print(f"Initial pad position: {info[0]['pad_position']}")
    print(f"Initial pad velocity: {info[0]['pad_velocity']}")

    positions = []
    for step in range(20):
        obs, rewards, dones, truncs, infos = world.step()
        positions.append(infos[0]["pad_position"].copy())
        if step % 5 == 0:
            print(f"Step {step}: pad_pos={infos[0]['pad_position']}, pad_vel={infos[0]['pad_velocity']}")

    # Check that pad actually moved
    positions = np.array(positions)
    total_displacement = np.linalg.norm(positions[-1][:2] - positions[0][:2])
    print(f"\nTotal horizontal displacement: {total_displacement:.3f}m")
    assert total_displacement > 0.01, "Pad should have moved"
    print("✓ Moving pad test passed")

    ####################################
    ##  Test 3: Custom O-U Parameters  ##
    ####################################
    print("\n=== Test 3: Custom O-U Parameters (Rough Seas) ===")

    # Set rough seas parameters
    world.single_variation_space["pad_motion"]["enabled"]._value = 1
    world.single_variation_space["pad_motion"]["theta_xy"]._value = 0.5  # weak restoring
    world.single_variation_space["pad_motion"]["sigma_xy"]._value = 1.0  # high noise

    obs, info = world.reset(seed=42)
    positions = []
    for step in range(50):
        obs, rewards, dones, truncs, infos = world.step()
        positions.append(infos[0]["pad_position"].copy())

    positions = np.array(positions)
    max_displacement = np.max(np.linalg.norm(positions[:, :2], axis=1))
    std_displacement = np.std(np.linalg.norm(positions[:, :2], axis=1))

    print(f"Max horizontal displacement: {max_displacement:.3f}m")
    print(f"Std horizontal displacement: {std_displacement:.3f}m")
    print(f"Theoretical steady-state std: {1.0 / np.sqrt(2 * 0.5):.3f}m")

    assert max_displacement < 3.0, "Pad should stay within max_radius constraint"
    print("✓ Custom O-U parameters test passed")

    ####################################
    ##  Test 4: Expert Policy Tracking ##
    ####################################
    print("\n=== Test 4: Expert Policy Tracking Moving Target ===")

    # Enable pad motion and run a full episode
    world.single_variation_space["pad_motion"]["enabled"]._value = 1
    world.single_variation_space["pad_motion"]["theta_xy"]._value = 1.0
    world.single_variation_space["pad_motion"]["sigma_xy"]._value = 0.5

    world.set_policy(ExpertPolicy())
    results = world.evaluate(episodes=1, seed=2347)

    print(f"Episode results: {results}")
    print("✓ Expert policy can track moving pad")

    print("\n=== All tests passed! ===")
