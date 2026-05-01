"""Tests for new World — self-contained with CounterEnv and mock policy."""

from collections import deque

import gymnasium as gym
import numpy as np
import pytest

from stable_worldmodel.world.env_pool import EnvPool
from stable_worldmodel.world.world import World


class CounterEnv(gym.Env):
    """Env that terminates after max_steps. Puts terminated in info like MegaWrapper."""

    def __init__(self, max_steps: int = 3):
        super().__init__()
        self.observation_space = gym.spaces.Box(0, 1, shape=(4,))
        self.action_space = gym.spaces.Box(-1, 1, shape=(2,))
        self._max_steps = max_steps
        self._step_count = 0
        self._seed_val = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        if seed is not None:
            self._seed_val = seed
        obs = np.zeros(4, dtype=np.float32)
        return obs, self._make_info(terminated=False)

    def step(self, action):
        self._step_count += 1
        obs = np.full(4, self._step_count, dtype=np.float32)
        terminated = self._step_count >= self._max_steps
        return obs, 1.0, terminated, False, self._make_info(terminated)

    @property
    def unwrapped(self):
        return self

    def _make_info(self, terminated):
        return {
            'pixels': np.full((1, 3, 3, 3), self._step_count, dtype=np.uint8),
            'goal': np.zeros((1, 3, 3, 3), dtype=np.uint8),
            'state': np.array([self._step_count], dtype=np.float32),
            'terminated': terminated,
        }


class RecordingPolicy:
    """Mock policy that records calls and tracks per-env action buffers.

    Mimics WorldModelPolicy's _needs_flush and terminated handling.
    """

    def __init__(self):
        self.env = None
        self.call_count = 0
        self.last_infos = None
        self._action_buffer = None
        self._flush_log = []
        self._dead_log = []

    def set_env(self, env):
        self.env = env
        n = env.num_envs
        self._action_buffer = [deque(maxlen=3) for _ in range(n)]
        for buf in self._action_buffer:
            buf.extend([np.zeros(env.single_action_space.shape)] * 3)

    def get_action(self, info_dict, **kwargs):
        self.call_count += 1
        self.last_infos = {k: v for k, v in info_dict.items()}
        n = self.env.num_envs

        needs_flush = info_dict.pop('_needs_flush', None)
        if needs_flush is not None:
            for i in range(n):
                if needs_flush[i]:
                    self._flush_log.append(i)
                    self._action_buffer[i].clear()
                    self._action_buffer[i].extend(
                        [np.zeros(self.env.single_action_space.shape)] * 3
                    )

        terminated = info_dict.get('terminated')
        actions = np.zeros((n, *self.env.single_action_space.shape))
        if terminated is not None:
            for i in range(n):
                if terminated[i]:
                    self._dead_log.append(i)
                    actions[i] = np.nan
                    continue
        return actions


def _make_world(num_envs=2, max_steps=3):
    """Build a World with CounterEnv directly, bypassing gym.make."""
    pool = EnvPool(
        [lambda ms=max_steps: CounterEnv(ms) for _ in range(num_envs)]
    )
    world = object.__new__(World)
    world.envs = pool
    world.policy = None
    world.infos = {}
    world.rewards = None
    world.terminateds = None
    world.truncateds = None
    return world


class TestRunAutoMode:
    def test_basic_auto_episodes(self):
        world = _make_world(num_envs=2, max_steps=3)
        policy = RecordingPolicy()
        world.policy = policy
        policy.set_env(world.envs)

        world._run(episodes=4, seed=0, mode='auto')

        assert policy.call_count > 0

    def test_auto_resets_envs(self):
        world = _make_world(num_envs=2, max_steps=2)
        policy = RecordingPolicy()
        world.policy = policy
        policy.set_env(world.envs)

        ep_done = []

        def on_done(env_idx, ep_idx, w):
            ep_done.append((env_idx, ep_idx))

        world._run(episodes=4, seed=0, mode='auto', on_done=on_done)

        assert len(ep_done) == 4

    def test_auto_sets_needs_flush(self):
        world = _make_world(num_envs=2, max_steps=2)
        policy = RecordingPolicy()
        world.policy = policy
        policy.set_env(world.envs)

        world._run(episodes=4, seed=0, mode='auto')

        # after first termination, _needs_flush should have been set
        # and the policy should have flushed those envs
        assert len(policy._flush_log) > 0

    def test_auto_unique_seeds(self):
        world = _make_world(num_envs=2, max_steps=2)
        policy = RecordingPolicy()
        world.policy = policy
        policy.set_env(world.envs)

        seeds_seen = []

        def on_done(env_idx, ep_idx, w):
            seeds_seen.append(w.envs.envs[env_idx]._seed_val)

        world._run(episodes=6, seed=0, mode='auto', on_done=on_done)

        assert len(seeds_seen) == 6
        assert len(set(seeds_seen)) == len(seeds_seen), (
            f'Duplicate seeds: {seeds_seen}'
        )

    def test_auto_infos_updated_after_reset(self):
        world = _make_world(num_envs=1, max_steps=2)
        policy = RecordingPolicy()
        world.policy = policy
        policy.set_env(world.envs)

        infos_after_reset = []

        def on_done(env_idx, ep_idx, w):
            pass

        def on_step(w):
            infos_after_reset.append(w.infos['state'][0].copy())

        world._run(
            episodes=2, seed=0, mode='auto', on_step=on_step, on_done=on_done
        )

        # after reset, state should go back to low values
        # episode 1: state goes 1, 2 (terminates)
        # episode 2: state goes 1, 2 (terminates)
        states = [s[0] for s in infos_after_reset]
        assert states[0] == 1.0
        assert states[1] == 2.0  # terminates
        assert states[2] == 1.0  # reset happened, fresh env
        assert states[3] == 2.0


class TestRunWaitMode:
    def test_wait_stops_dead_envs(self):
        world = _make_world(num_envs=2, max_steps=3)
        policy = RecordingPolicy()
        world.policy = policy
        policy.set_env(world.envs)

        world._run(max_steps=10, mode='wait', seed=0)

        # both envs should terminate after 3 steps, run should stop
        assert policy.call_count == 3

    def test_wait_dead_envs_get_nan(self):
        """When one env dies before the other, dead env gets NaN actions."""
        world = _make_world(num_envs=2, max_steps=5)
        # make env 0 die faster
        world.envs.envs[0]._max_steps = 2
        world.envs.envs[1]._max_steps = 4

        policy = RecordingPolicy()
        world.policy = policy
        policy.set_env(world.envs)

        world._run(max_steps=10, mode='wait', seed=0)

        # env 0 dies at step 2, env 1 at step 4
        # after env 0 dies, policy should see terminated=True for env 0
        assert 0 in policy._dead_log

    def test_wait_skips_stepping_dead_envs(self):
        world = _make_world(num_envs=2, max_steps=5)
        world.envs.envs[0]._max_steps = 2
        world.envs.envs[1]._max_steps = 5

        policy = RecordingPolicy()
        world.policy = policy
        policy.set_env(world.envs)

        world._run(max_steps=10, mode='wait', seed=0)

        # env 0 dies at step 2, env 1 continues
        # env 0 should stay at step_count=2 (not stepped further)
        assert world.envs.envs[0]._step_count == 2
        assert world.envs.envs[1]._step_count == 5

    def test_wait_no_needs_flush(self):
        world = _make_world(num_envs=2, max_steps=3)
        policy = RecordingPolicy()
        world.policy = policy
        policy.set_env(world.envs)

        world._run(max_steps=10, mode='wait', seed=0)

        # in wait mode, no resets happen, so no flush
        assert len(policy._flush_log) == 0


class TestRunCallbacks:
    def test_on_step_called_every_step(self):
        world = _make_world(num_envs=1, max_steps=3)
        policy = RecordingPolicy()
        world.policy = policy
        policy.set_env(world.envs)

        step_count = [0]

        def on_step(w):
            step_count[0] += 1

        world._run(max_steps=3, mode='wait', seed=0, on_step=on_step)

        assert step_count[0] == 3

    def test_on_done_receives_correct_ep_idx(self):
        world = _make_world(num_envs=2, max_steps=2)
        policy = RecordingPolicy()
        world.policy = policy
        policy.set_env(world.envs)

        ep_indices = []

        def on_done(env_idx, ep_idx, w):
            ep_indices.append(ep_idx)

        world._run(episodes=4, seed=0, mode='auto', on_done=on_done)

        assert ep_indices == [0, 1, 2, 3]

    def test_on_done_stops_at_episode_limit(self):
        world = _make_world(num_envs=2, max_steps=2)
        policy = RecordingPolicy()
        world.policy = policy
        policy.set_env(world.envs)

        done_count = [0]

        def on_done(env_idx, ep_idx, w):
            done_count[0] += 1

        world._run(episodes=3, seed=0, mode='auto', on_done=on_done)

        assert done_count[0] == 3


class TestRunEdgeCases:
    def test_no_policy_raises(self):
        world = _make_world(num_envs=1)
        with pytest.raises(RuntimeError, match='No policy set'):
            world._run(episodes=1, seed=0)

    def test_no_episodes_or_max_steps_raises(self):
        world = _make_world(num_envs=1)
        policy = RecordingPolicy()
        world.policy = policy
        policy.set_env(world.envs)
        with pytest.raises(ValueError):
            world._run(seed=0)

    def test_invalid_mode_raises(self):
        world = _make_world(num_envs=1)
        policy = RecordingPolicy()
        world.policy = policy
        policy.set_env(world.envs)
        with pytest.raises(AssertionError):
            world._run(episodes=1, seed=0, mode='invalid')

    def test_max_steps_stops_even_without_termination(self):
        world = _make_world(num_envs=1, max_steps=100)
        policy = RecordingPolicy()
        world.policy = policy
        policy.set_env(world.envs)

        world._run(max_steps=5, seed=0, mode='auto')

        assert policy.call_count == 5

    def test_both_episodes_and_max_steps(self):
        world = _make_world(num_envs=1, max_steps=2)
        policy = RecordingPolicy()
        world.policy = policy
        policy.set_env(world.envs)

        done_count = [0]

        def on_done(env_idx, ep_idx, w):
            done_count[0] += 1

        # episodes=1 should stop before max_steps=100
        world._run(
            episodes=1, max_steps=100, seed=0, mode='auto', on_done=on_done
        )

        assert done_count[0] == 1


class TestNeedsFlush:
    def test_flush_clears_buffer_on_auto_reset(self):
        world = _make_world(num_envs=2, max_steps=2)
        policy = RecordingPolicy()
        world.policy = policy
        policy.set_env(world.envs)

        # fill buffers with identifiable data
        for buf in policy._action_buffer:
            buf.clear()
            buf.extend([np.ones(2) * 99] * 3)

        world._run(episodes=4, seed=0, mode='auto')

        # both envs should have been flushed at least once
        assert 0 in policy._flush_log
        assert 1 in policy._flush_log

    def test_flush_only_for_done_envs(self):
        world = _make_world(num_envs=2, max_steps=10)
        # only env 0 terminates quickly
        world.envs.envs[0]._max_steps = 2
        world.envs.envs[1]._max_steps = 100

        policy = RecordingPolicy()
        world.policy = policy
        policy.set_env(world.envs)

        # need episodes > 1 so the run continues after env 0's first termination
        # (otherwise _run returns before the reset/flush happens)
        world._run(episodes=2, seed=0, mode='auto')

        # only env 0 should be flushed
        assert 0 in policy._flush_log
        assert 1 not in policy._flush_log

    def test_needs_flush_not_present_initially(self):
        world = _make_world(num_envs=2, max_steps=100)
        policy = RecordingPolicy()
        world.policy = policy
        policy.set_env(world.envs)

        world._run(max_steps=1, seed=0, mode='auto')

        # no termination happened, no flush
        assert len(policy._flush_log) == 0


class TestEvaluate:
    def test_evaluate_returns_results(self):
        world = _make_world(num_envs=2, max_steps=2)
        policy = RecordingPolicy()
        world.policy = policy
        policy.set_env(world.envs)

        results = world.evaluate(episodes=4, seed=0)

        assert 'success_rate' in results
        assert 'episode_successes' in results
        assert 'seeds' in results
        assert len(results['episode_successes']) == 4

    def test_evaluate_default_mode_is_auto(self):
        world = _make_world(num_envs=1, max_steps=2)
        policy = RecordingPolicy()
        world.policy = policy
        policy.set_env(world.envs)

        # should complete 2 episodes (auto-resets)
        results = world.evaluate(episodes=2, seed=0)

        assert len(results['episode_successes']) == 2

    def test_evaluate_wait_mode(self):
        world = _make_world(num_envs=2, max_steps=3)
        policy = RecordingPolicy()
        world.policy = policy
        policy.set_env(world.envs)

        results = world.evaluate(episodes=2, seed=0, reset_mode='wait')

        assert len(results['episode_successes']) == 2


class TestSetPolicy:
    def test_set_policy(self):
        world = _make_world(num_envs=2)
        policy = RecordingPolicy()

        world.set_policy(policy)

        assert world.policy is policy
        assert policy.env is world.envs

    def test_set_policy_seeds_policy(self):
        class SeededPolicy(RecordingPolicy):
            def __init__(self):
                super().__init__()
                self.seed = 1234
                self.seed_calls = []

            def set_seed(self, seed):
                self.seed_calls.append(seed)

        world = _make_world(num_envs=2)
        policy = SeededPolicy()
        world.set_policy(policy)
        assert policy.seed_calls == [1234]

    def test_reset_initializes_state(self):
        world = _make_world(num_envs=2)
        world.reset(seed=42)

        assert world.terminateds is not None
        assert world.truncateds is not None
        assert len(world.infos) > 0
        np.testing.assert_array_equal(world.terminateds, [False, False])

    def test_reset_per_env_options_passed_through(self):
        seen = {'opts': None}

        class OptionEnv(CounterEnv):
            def reset(self, *, seed=None, options=None):
                seen['opts'] = options
                return super().reset(seed=seed, options=options)

        pool = EnvPool([lambda: OptionEnv(3) for _ in range(2)])
        world = object.__new__(World)
        world.envs = pool
        world.policy = None
        world.infos = {}
        world.rewards = None
        world.terminateds = None
        world.truncateds = None

        per_env = [{'variation': ['a']}, {'variation': ['b']}]
        world.reset(options=per_env)
        # The second env's reset is called last, so `seen` holds its options.
        assert seen['opts'] == {'variation': ['b']}


class TestWorldMisc:
    def test_num_envs_matches_pool(self):
        world = _make_world(num_envs=3)
        assert world.num_envs == 3

    def test_close_calls_every_env(self):
        close_calls = []

        class CloseEnv(CounterEnv):
            def close(self):
                close_calls.append(id(self))

        pool = EnvPool([lambda: CloseEnv(3) for _ in range(3)])
        world = object.__new__(World)
        world.envs = pool
        world.close()
        assert len(close_calls) == 3
