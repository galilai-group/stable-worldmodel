import minari
import torch
from .utils import set_state
from tqdm.rich import tqdm

import stable_pretraining as spt
import numpy as np

from torch.utils.data import default_collate

class StepsDataset(spt.data.HFDataset):

    def __init__(self,*args,
                num_steps=2,
                frameskip=1,
                torch_exclude_column={"pixels",
                                       "goal",},
                **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_steps = num_steps
        self.frameskip = frameskip

        assert "episode_idx" in self.dataset.column_names, "Dataset must have 'episode_idx' column"
        assert "step_idx" in self.dataset.column_names, "Dataset must have 'step_idx' column"


        self.episodes = np.unique(self.dataset["episode_idx"])
        self.slices = {e:self.num_slice(e) for e in self.episodes}
        self.cum_slices = np.cumsum([0] + [self.slices[e] for e in self.episodes])

        self.torch_exclude_column = torch_exclude_column

        # TODO: add assert for basic column name
    
        cols = [c for c in self.dataset.column_names if c not in self.torch_exclude_column]
        self.dataset = self.dataset.with_format("torch", columns=cols, output_all_columns=True)


    def num_slice(self, episode_idx):
        """Return number of possible slices for a given episode index"""
        idx = np.nonzero(self.dataset["episode_idx"] == episode_idx)[0][0].item()
        episode_len = self.dataset["episode_len"][idx]
        num_slices = 1 + (episode_len - self.num_steps*self.frameskip)

        assert num_slices > 0, f"Episode {episode_idx} is too short for {self.num_steps} steps with {self.frameskip} frameskip (len={episode_len})"

        return num_slices


    def process_sample(self, sample):
        if self._trainer is not None:
            if "global_step" in sample:
                raise ValueError("Can't use that keywords")
            if "current_epoch" in sample:
                raise ValueError("Can't use that keywords")
            sample["global_step"] = self._trainer.global_step
            sample["current_epoch"] = self._trainer.current_epoch
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return int(self.cum_slices[-1])

    def __getitem__(self, idx):

        # find which episode this idx belongs to
        ep_idx = np.searchsorted(self.cum_slices, idx, side="right") - 1
        episode = self.episodes[ep_idx]

        # find local slice within episode
        local_idx = idx - self.cum_slices[ep_idx]

        # get dataset indices for this slice
        ep_mask = torch.nonzero(self.dataset["episode_idx"] == episode).squeeze().tolist()

        # starting step index inside this episode
        start_step = local_idx
        
        # slice steps with frameskip
        slice_idx = [
            ep_mask[start_step + i] for i in range(self.num_steps*self.frameskip)
        ]

        # transform the data
        raw = [self.transform(self.dataset[i]) for i in slice_idx]
        raw_steps = default_collate(raw)

        # add the frameskip
        steps = {}
        for k, v in raw_steps.items():
            steps[k] = v[::self.frameskip]

        # process actions
        actions = raw_steps["action"]
        if self.frameskip > 1:
            actions = actions.reshape(self.num_steps, -1)

        steps["action"] = actions

        # # align actions and observations (skip first action)
        # if len(actions) > 1:
        #     actions = actions[1:]

        # group actions between frameskip
        # if self.frameskip > 1:
        #    # drop last chunk of action (useless to keep)
        #    actions = actions[:-(self.frameskip-1)]
        #    actions = actions.reshape(self.num_steps-1, -1)
        #    assert len(actions) == self.num_steps-1, f"Actions length {len(actions)} should be {self.num_steps-1}"
        
        return steps


# def cache_minari_with_pixels(name, env):
#     dataset = minari.load_dataset(name)
#     N = len(dataset)
#     for i, episode_data in enumerate(dataset.iterate_episodes()):
#         env.reset()
#         observations = episode_data.observations
#         actions = episode_data.actions
#         rewards = episode_data.rewards
#         terminations = episode_data.terminations
#         truncations = episode_data.truncations
#         infos = episode_data.infos
#         pixels = []
#         assert "observation" in observations
#         for action, observation in zip(actions, observations["observation"]):
#             set_state(env, observation)
#             env.step(action)
#             pixels.append(env.render())


if __name__ == "__main__":
    import gymnasium as gym
    import gymnasium_robotics
    import xenoworlds
    from gymnasium.wrappers import RecordVideo

    env = gym.make("AntMaze_Large-v1", render_mode="rgb_array")
    env = RecordVideo(
        env, video_folder="test_videos_data_caching", episode_trigger=lambda x: True
    )
    data = xenoworlds.data.cache_minari_with_pixels("D4RL/antmaze/large-play-v1", env)
