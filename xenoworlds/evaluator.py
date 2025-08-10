import numpy as np
import torch
from .policy import BasePolicy
from .world import World


## -- Evaluator / Collector
### Evaluator(env, policy)
class Evaluator:
    # the role of evaluator is to determine perf of the policy in the env
    def __init__(self, world: World, policy: BasePolicy):
        self.world = world
        self.policy = policy

    def run(self, episodes=1, video_episodes=1, max_steps=100):
        # TODO return interested logging data
        # NOTE Dan: we should have a convention for metrics to log and then get them from the infos
        data = {
            # "success": [],
        }
        frames_list = []

        for episode in range(episodes):
            frames = []
            for states, infos in self.world:
                # get action                
                # NOTE Dan: It might be preferable that the environment return the raw pixel observation and the policy would take care of the transforms
                obs = torch.from_numpy(states)
                goal = torch.from_numpy(self.world.cur_goals)  # NOTE: would be more efficient if we get the goal once per episode
                actions = self.policy.get_action(obs, goal)
                # actions = actions.squeeze(0) if actions.ndim == 2 else actions
                # apply actions in the env
                # for a in actions.unbind(0):
                #     self.world.step(a.numpy())

                # save logging data
                for key in data.keys():
                    data[key].append(infos[key])
                # save frames for video visualization
                if episode < video_episodes:
                    frames.append(np.concatenate([self.world.cur_goal_images[0], infos["image"][0]], axis=0))

                # todo assert action has the right shape
                self.world.step(actions)

            if episode < video_episodes:
                frames_list.append(np.array(frames))
            
            print(f"Episode {episode + 1} finished ")

        self.world.close()
        data["frames_list"] = frames_list

        return data

### DataSetUpload (download using stable_ssl)
