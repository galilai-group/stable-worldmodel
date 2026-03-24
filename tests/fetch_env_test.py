import time
import gymnasium as gym
import stable_worldmodel.envs

# 1. Create the environment (FetchPush has the physical block to manipulate)
env = gym.make('swm/FetchPush-v3', render_mode="human")

# 2. Loop through 100 episodes
for i in range(100):
    # Pass the variations you want to sample during this reset!
    obs, info = env.reset(options={
        "variation": [
            "table.color", "object.color", "background.color", "light.intensity",
            "agent.start_position", "block.start_position", "block.angle", "goal.start_position"
        ]
    })
    
    # Retrieve the sampled values from the variation space using Gymnasium's wrapper accessor
    bg_color = env.get_wrapper_attr("variation_space")["background"]["color"].value
    agent_pos = env.get_wrapper_attr("variation_space")["agent"]["start_position"].value
    block_pos = env.get_wrapper_attr("variation_space")["block"]["start_position"].value
    goal_pos = env.get_wrapper_attr("variation_space")["goal"]["start_position"].value
    
    # Format them cleanly for printing
    b_c = [round(float(c), 2) for c in bg_color]
    a_p = [round(float(p), 2) for p in agent_pos]
    b_p = [round(float(p), 2) for p in block_pos]
    g_p = [round(float(p), 2) for p in goal_pos]
    
    print(f"Ep.{i}: [bg: {b_c}, agent_xy: {a_p}, block_xy: {b_p}, goal_xyz: {g_p}]")
    
    # Run 50 random steps just to see the robot move
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        time.sleep(0.02)  # Slight pause to make it easier to watch
