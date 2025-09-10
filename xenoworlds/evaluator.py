import numpy as np
import torch
from .policy import BasePolicy
from .world import World
import numpy as np


## -- Evaluator / Collector
### Evaluator(env, policy)
class Evaluator:
    # the role of evaluator is to determine perf of the policy in the env
    def __init__(self, world: World, policy: BasePolicy):
        self.world = world
        self.policy = policy

    def run(self, episodes=1):
        # todo return interested logging data
        data = {}

        for episode in range(episodes):
            # sample goals and get their representations
            goals = None  # self.goal_dist.sample()
            # goals = self.solver.world_model.encode(goals).unsqueeze(0)

            for states, rewards in self.world:
                # get action from the policy wih optional goal specification
                pixels = torch.from_numpy(states["pixels"])  # get the state
                actions = self.policy.get_action(pixels, goals=goals)
                self.world.step(actions)

            print(f"Episode {episode + 1} finished ")
            self.world.close()

        return data