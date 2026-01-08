import numpy as np

from stable_worldmodel.policy import BasePolicy


class WeakPolicy(BasePolicy):
    """Collection policy for PushT environment."""

    def __init__(
        self,
        dist_constraint=100,
        **kwargs,
    ):
        """
        Args:
            dist_constraint (int, optional): pixels square constraint around the block for sampling actions.
            **kwargs: Arbitrary keyword arguments passed to parent BasePolicy.
        """
        super().__init__(**kwargs)

        self.dist_constraint = dist_constraint
        self.discrete = False
        assert self.dist_constraint > 0, "dist_constraint must be positive."

    def set_env(self, env):
        self.env = env
        self.discrete = "Discrete" in self.env.spec.id
        assert "swm/PushT" in self.env.spec.id, "PushTCollectionPolicy can only be used with the PushT environment."

    def get_action(self, info_dict, **kwargs):
        assert hasattr(self, "env"), "Environment not set for the policy"

        # Handle vectorized envs (VecEnv-style) and single envs gracefully
        base_env = self.env.unwrapped
        if hasattr(base_env, "envs"):
            envs = [e.unwrapped for e in base_env.envs]
        else:
            envs = [base_env]

        act_shape = self.env.action_space.shape
        actions = np.zeros(act_shape, dtype=np.float32)

        for i, env in enumerate(envs):
            # sample a random action
            action = np.random.uniform(-1, 1, size=env.action_space.shape)
            # scale action to environment position
            action = action * env.action_scale
            action = env.agent.position + action
            # constrain agent to be near the block to increase probability of interaction
            block_pos = np.array((env.block.position.x, env.block.position.y))
            action = np.clip(action, block_pos - self.dist_constraint, block_pos + self.dist_constraint)
            # rescale action back to action space
            action = (action - env.agent.position) / env.action_scale
            action = np.clip(action, -1, 1)

            if self.discrete:
                action = env.quantizer.quantize(action)

            # set action for this env
            actions[i] = action

        # VecEnv expects (n_envs, action_dim)
        return actions
