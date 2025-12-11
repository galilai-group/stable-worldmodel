import numpy as np
from ogbench.manipspace.oracles.markov.button_markov import ButtonMarkovOracle
from ogbench.manipspace.oracles.markov.cube_markov import CubeMarkovOracle
from ogbench.manipspace.oracles.markov.drawer_markov import DrawerMarkovOracle
from ogbench.manipspace.oracles.markov.window_markov import WindowMarkovOracle
from ogbench.manipspace.oracles.plan.button_plan import ButtonPlanOracle
from ogbench.manipspace.oracles.plan.cube_plan import CubePlanOracle
from ogbench.manipspace.oracles.plan.drawer_plan import DrawerPlanOracle
from ogbench.manipspace.oracles.plan.window_plan import WindowPlanOracle

from stable_worldmodel.policy import BasePolicy


class OGBCollectionPolicy(BasePolicy):
    """Collection Policy for OGBench Manipulation Environments."""

    def __init__(
        self,
        policy_type="markov_oracle",
        action_noise=0.1,
        p_random_action=0.0,
        noise_smoothing=0.5,
        min_norm=0.4,
        **kwargs,
    ):
        """
        Args:
            policy_type (str, optional): Collection policy type. Can one of 2 options:
              - "plan_oracle": a non-Markovian oracle that follows a pre-computed plan.
              - "markov_oracle": a Markovian, closed-loop oracle with Gaussian action noise.
            action_noise (float, optional): Action noise level. Applies to both policy types.
            p_random_action (float, optional): Probability of selecting a random action. Applies to both policy types.
            noise_smoothing (float, optional): Action noise smoothing level for "plan_oracle" policy type.
            min_norm (float, optional): Minimum action norm for "markov_oracle" policy type.
            **kwargs: Arbitrary keyword arguments passed to parent BasePolicy.

            Note: see original OGBench params per environment and policy type in
            https://github.com/seohongpark/ogbench/blob/master/data_gen_scripts/commands.sh
        """
        super().__init__(**kwargs)
        self.type = policy_type
        assert self.type in ["plan_oracle", "markov_oracle"], (
            "Invalid policy_type. Must be one of ['plan_oracle', 'markov_oracle']."
        )

        self.action_noise = action_noise
        self.p_random_action = p_random_action
        self.noise_smoothing = noise_smoothing
        self.min_norm = min_norm

    def set_env(self, env):
        self.env = env
        assert self.env.spec.id in [
            "swm/OGBCube-v0",
            "swm/OGBScene-v0",
        ], "OGBCollectionPolicy can only be used with OGBench Manipulation environments."

        self._set_oracle_agents()

        # to be set at each env reset:
        self._p_stack = np.zeros(self.env.num_envs)
        self._xi = np.zeros(self.env.num_envs)  # action noise level for Markov oracle
        self._agents = [None] * self.env.num_envs

    def _set_oracle_agents(self):
        single_env = self.env.unwrapped.envs[1]
        if "Cube" in single_env.spec.id:
            if self.type == "markov_oracle":
                # create one independent oracle instance per environment
                self._oracle_agents = {
                    "cube": [
                        CubeMarkovOracle(env=single_env, min_norm=self.min_norm) for _ in range(self.env.num_envs)
                    ],
                }
            else:
                self._oracle_agents = {
                    "cube": [
                        CubePlanOracle(env=single_env, noise=self.action_noise, noise_smoothing=self.noise_smoothing)
                        for _ in range(self.env.num_envs)
                    ],
                }
        elif "Scene" in single_env.spec.id:
            if self.type == "markov_oracle":
                # create one independent oracle instance per env for each task type
                self._oracle_agents = {
                    "cube": [
                        CubeMarkovOracle(env=single_env, min_norm=self.min_norm, max_step=100)
                        for _ in range(self.env.num_envs)
                    ],
                    "button": [
                        ButtonMarkovOracle(env=single_env, min_norm=self.min_norm) for _ in range(self.env.num_envs)
                    ],
                    "drawer": [
                        DrawerMarkovOracle(env=single_env, min_norm=self.min_norm) for _ in range(self.env.num_envs)
                    ],
                    "window": [
                        WindowMarkovOracle(env=single_env, min_norm=self.min_norm) for _ in range(self.env.num_envs)
                    ],
                }
            else:
                self._oracle_agents = {
                    "cube": [
                        CubePlanOracle(env=single_env, noise=self.action_noise, noise_smoothing=self.noise_smoothing)
                        for _ in range(self.env.num_envs)
                    ],
                    "button": [
                        ButtonPlanOracle(env=single_env, noise=self.action_noise, noise_smoothing=self.noise_smoothing)
                        for _ in range(self.env.num_envs)
                    ],
                    "drawer": [
                        DrawerPlanOracle(env=single_env, noise=self.action_noise, noise_smoothing=self.noise_smoothing)
                        for _ in range(self.env.num_envs)
                    ],
                    "window": [
                        WindowPlanOracle(env=single_env, noise=self.action_noise, noise_smoothing=self.noise_smoothing)
                        for _ in range(self.env.num_envs)
                    ],
                }
        elif "puzzle" in single_env.spec.id:
            if self.type == "markov_oracle":
                self._oracle_agents = {
                    "button": [
                        ButtonMarkovOracle(env=single_env, min_norm=self.min_norm, gripper_always_closed=True)
                        for _ in range(self.env.num_envs)
                    ],
                }
            else:
                self._oracle_agents = {
                    "button": [
                        ButtonPlanOracle(
                            env=single_env,
                            noise=self.action_noise,
                            noise_smoothing=self.noise_smoothing,
                            gripper_always_closed=True,
                        )
                        for _ in range(self.env.num_envs)
                    ],
                }

    def _get_cube_stack_prob(self):
        base_env = self.env.unwrapped
        if hasattr(base_env, "envs"):
            envs = [e.unwrapped for e in base_env.envs]
        else:
            envs = [base_env]

        env_type = envs[0]._env_type  # assuming all envs have the same type

        if env_type == "single":
            p_stack = 0.0
        elif env_type == "double":
            p_stack = np.random.uniform(0.0, 0.25)
        elif env_type == "triple":
            p_stack = np.random.uniform(0.05, 0.35)
        elif env_type == "quadruple":
            p_stack = np.random.uniform(0.1, 0.5)
        elif env_type == "octuple":
            p_stack = np.random.uniform(0.0, 0.35)
        else:
            p_stack = 0.5

        return p_stack

    def get_action(self, info_dict, **kwargs):
        assert hasattr(self, "env"), "Environment not set for the policy"
        assert "privileged/target_task" in info_dict, "'privileged/target_task' must be provided in info_dict"

        # Handle vectorized envs (VecEnv-style) and single envs gracefully
        base_env = self.env.unwrapped
        if hasattr(base_env, "envs"):
            envs = [e.unwrapped for e in base_env.envs]
        else:
            envs = [base_env]

        act_shape = self.env.action_space.shape
        actions = np.zeros(act_shape, dtype=np.float32)

        for i, env in enumerate(envs):
            # extract env-relevant info
            info = {k: v[i][0] for k, v in info_dict.items() if not k.startswith("_")}

            # new episode resets
            if info["step_idx"] == 0:  # new episode
                self._p_stack[i] = self._get_cube_stack_prob()
                if self.type == "markov_oracle":
                    # Set the action noise level for this episode.
                    self._xi[i] = np.random.uniform(0, self.action_noise)

                self._agents[i] = self._oracle_agents[info["privileged/target_task"]][i]
                self._agents[i].reset(None, info)

            # action logic
            if np.random.rand() < self.p_random_action:
                # Sample a random action.
                action = env.action_space.sample()
            else:
                # Get an action from the oracle.
                action = self._agents[i].select_action(None, info)
                action = np.array(action)
                if self.type == "markov_oracle":
                    # Add Gaussian noise to the action.
                    xi = self._xi[i]
                    action = action + np.random.normal(0, [xi, xi, xi, xi * 3, xi * 10], action.shape)
            action = np.clip(action, -1, 1)
            actions[i] = action

            # Set a new task when the current subtask is done.
            if self._agents[i].done:
                agent_ob, agent_info = env.unwrapped.set_new_target(p_stack=self._p_stack[i])
                self._agents[i] = self._oracle_agents[agent_info["privileged/target_task"]][i]
                self._agents[i].reset(agent_ob, agent_info)

        # NOTE: there is a health check in the Scene env that discards episodes where the cube is not visible, think if we want to deal with this somehow
        # NOTE: modify Scene tasks with closed drawer to open drawer

        # VecEnv expects (n_envs, action_dim)
        return actions
