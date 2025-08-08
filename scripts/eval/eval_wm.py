import os

os.environ["MUJOCO_GL"] = "egl"
import gymnasium as gym
import gymnasium_robotics
import torch
import xenoworlds
from xenoworlds.planner import GD, CEMNevergrad


if __name__ == "__main__":
    # run with MUJOCO_GL=egl python example.py

    # gym.register_envs(gymnasium_robotics)
    # envs = gym.envs.registry.keys()
    # print(envs)
    # asdf
    wrappers = [
        # lambda x: RecordVideo(x, video_folder="./videos"),
        lambda x: xenoworlds.wrappers.AddRenderObservation(x, render_only=False),
        lambda x: xenoworlds.wrappers.TransformObservation(x),
    ]
    world = xenoworlds.World(
        "InvertedPendulum-v5", num_envs=4, wrappers=wrappers, max_episode_steps=10
    )

    world_model = xenoworlds.DummyWorldModel(
        image_shape=(3, 224, 224), action_dim=world.single_action_space.shape[0]
    )

    # print(sorted(envs))
    planner = GD(world_model, n_steps=100, action_space=world.action_space)
    # planner = CEMNevergrad(
    #     world_model, n_steps=100, action_space=world.action_space, planning_horizon=3
    # )
    agent = xenoworlds.Agent(planner, world)
    # 'FetchPush-v1'
    agent.run(episodes=5)
