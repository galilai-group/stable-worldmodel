import minari
from scripts.data.pusht_dataset import PushTDataset


class PushTDatasetCallback(minari.StepDataCallback):
    def __init__(self):
        super().__init__()

        self.train_dataset = PushTDataset(data_path="pusht_noise/train/")
        self.val_dataset = PushTDataset(data_path="pusht_noise/val/")

        self._episode = 0
        self._counter = 0

    def __call__(self, env, **kwargs):
        step_data = super().__call__(env, **kwargs)

        is_val = self._episode >= len(self.train_dataset)

        if not is_val:
            idx = self._episode
            dataset = self.train_dataset
        else:
            idx = self._episode - len(self.train_dataset)
            dataset = self.val_dataset

        obs, act, state, shape = dataset[idx]

        step_data = {}

        step_data: minari.StepData = {
            "action": act[self._counter].numpy(),
            "observation": {
                "proprio": obs["proprio"][self._counter].numpy(),
                "state": state[self._counter].numpy(),
                "pixels": obs["visual"][self._counter]
                .permute(1, 2, 0)
                .numpy()
                .astype(np.uint8),
            },
            "reward": 0.0,
            "terminated": False,
            "truncated": False,
            "info": {"shape": shape["shape"]},
        }

        self._counter += 1

        if self._counter == dataset.get_seq_length(idx).item():
            self._episode += 1
            self._counter = 0

        return step_data


if __name__ == "__main__":
    import xenoworlds
    import gymnasium as gym
    import numpy as np
    from minari import DataCollector

    train_dataset = PushTDataset(data_path="pusht_noise/train/")
    val_dataset = PushTDataset(data_path="pusht_noise/val/")

    MAX_STEPS = 109
    N_EPISODES = len(train_dataset)
    seed = None

    env = gym.make(
        "xenoworlds/PushT",
        render_mode="rgb_array",
        resolution=224,
    )

    env = gym.wrappers.AddRenderObservation(env, render_only=False)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=MAX_STEPS)

    # Override the observation space for pixels
    env.observation_space.spaces["pixels"] = gym.spaces.Box(
        low=0, high=255, shape=(224, 224, 3), dtype=np.uint8
    )

    env.observation_space.spaces["proprio"] = gym.spaces.Box(
        low=0, high=512, shape=(4,), dtype=np.float32
    )

    env = DataCollector(env, record_infos=True, step_data_callback=PushTDatasetCallback)

    # env.reset(seed=42)
    # action = env.action_space.sample()
    # obs, rew, terminated, truncated, info = env.step(action)

    # print(f"Observation state shape: {obs['state'].shape}")
    # print(f"Observation pixels shape: {obs['pixels'].shape}")

    # print(f"Reward: {rew}")
    # print(f"Terminated: {terminated}, Truncated: {truncated}")
    # print(f"Info: {info}")

    for ep_idx in range(len(train_dataset)):
        print(f"Episode {ep_idx + 1}/{len(train_dataset)}")
        env.reset(seed=seed or ep_idx)
        for _ in range(train_dataset.get_seq_length(ep_idx).item()):
            action = env.action_space.sample()
            obs, rew, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

    for ep_idx in range(len(val_dataset)):
        print(f"Episode {ep_idx + 1}/{len(val_dataset)}")
        env.reset(seed=seed or ep_idx)
        for _ in range(val_dataset.get_seq_length(ep_idx).item()):
            action = env.action_space.sample()
            obs, rew, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

    dataset = env.create_dataset(
        dataset_id="dinowm/pusht_noise-v0",
        algorithm_name="Export-Policy",
        code_permalink="https://github.com/gaoyuezhou/dino_wm",
        author="Gaoyue Zhou",
        # author_email="contact@farama.org",
    )
