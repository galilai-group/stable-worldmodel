"""Tests for new World — self-contained with CounterEnv and mock policy."""

from collections import deque

import gymnasium as gym
import numpy as np
import pytest
import torch

from stable_worldmodel.world.env_pool import EnvPool
from stable_worldmodel.world.world import World, _extract_init_goal


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

        def on_step(w, mask):
            infos_after_reset.append(w.infos['state'][0].copy())

        world._run(
            episodes=2, seed=0, mode='auto', on_step=on_step, on_done=on_done
        )

        # on_step now also fires right after each reset (initial + auto),
        # so the reset frame (state=0) is captured before the step frames.
        # episode 1: reset (0), step 1, step 2 (terminates)
        # episode 2: reset (0), step 1, step 2 (terminates)
        states = [s[0] for s in infos_after_reset]
        assert states[0] == 0.0  # initial reset
        assert states[1] == 1.0
        assert states[2] == 2.0  # terminates
        assert states[3] == 0.0  # reset happened, fresh env
        assert states[4] == 1.0
        assert states[5] == 2.0


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

        def on_step(w, mask):
            step_count[0] += 1

        world._run(max_steps=3, mode='wait', seed=0, on_step=on_step)

        # +1 for the initial reset (seed is passed) firing on_step once
        # before the 3 step-loop iterations.
        assert step_count[0] == 4

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

            def _set_seed(self, seed):
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


class EpisodeDataEnv(CounterEnv):
    """CounterEnv exposing episode-scoped data snapshotted at reset."""

    def __init__(self, max_steps: int = 3, tag: str = 'env'):
        super().__init__(max_steps)
        self._tag = tag
        self._reset_count = 0
        self._episode_data = None

    def reset(self, *, seed=None, options=None):
        out = super().reset(seed=seed, options=options)
        self._reset_count += 1
        self._episode_data = {
            'model_xml': f'<{self._tag} reset="{self._reset_count}"/>',
            'reset_idx': self._reset_count,
        }
        return out

    def get_episode_data(self):
        return self._episode_data


class FlatPixelsEpisodeDataEnv(EpisodeDataEnv):
    """Emits HWC image infos (no leading time dim), like real wrapped envs —
    the shape the lance JPEG encoder expects per step."""

    def _make_info(self, terminated):
        info = super()._make_info(terminated)
        info['pixels'] = np.full((3, 3, 3), self._step_count, dtype=np.uint8)
        info['goal'] = np.zeros((3, 3, 3), dtype=np.uint8)
        return info


class DatasetResetEnv(CounterEnv):
    """CounterEnv implementing the dataset-restore API; records the rows."""

    def __init__(self, max_steps: int = 3):
        super().__init__(max_steps)
        self.received = []

    def reset_options_from_dataset(self, init_row, goal_row):
        self.received.append((init_row, goal_row))
        return {}


class FakeEvalDataset:
    """Two-episode dataset with per-episode-varying ``states`` dims plus
    episode-scoped blobs, mimicking the robocasa lance layout."""

    column_names = ['pixels', 'proprio', 'states', 'seed']
    episode_column_names = ['model_xml', 'ep_meta']

    def __init__(self):
        self.lengths = np.array([6, 6])
        self._state_dims = {0: 4, 1: 7}

    def load_chunk(self, episodes_idx, start, end):
        chunks = []
        for ep, s, e in zip(episodes_idx, start, end):
            ep, n = int(ep), int(e - s)
            chunks.append(
                {
                    'pixels': torch.full(
                        (n, 3, 3, 3), ep + 1, dtype=torch.uint8
                    ),
                    'proprio': (
                        torch.arange(n * 2, dtype=torch.float32).reshape(n, 2)
                        + 10.0 * ep
                    ),
                    'states': torch.full(
                        (n, self._state_dims[ep]),
                        float(ep),
                        dtype=torch.float32,
                    ),
                    'seed': torch.full((n, 1), 100.0 + ep),
                }
            )
        return chunks

    def get_episode_data(self, episodes_idx=None):
        idxs = (
            [int(i) for i in episodes_idx]
            if episodes_idx is not None
            else [0, 1]
        )
        return {
            'model_xml': [f'<scene {i}/>' for i in idxs],
            'ep_meta': [f'meta-{i}'.encode() for i in idxs],
        }


def _make_world_with(env_fns):
    pool = EnvPool(env_fns)
    world = object.__new__(World)
    world.envs = pool
    world.policy = None
    world.infos = {}
    world.rewards = None
    world.terminateds = None
    world.truncateds = None
    policy = RecordingPolicy()
    world.policy = policy
    policy.set_env(world.envs)
    return world


class TestExtractInitGoal:
    def test_rows_content(self):
        ds = FakeEvalDataset()
        init_rows, goal_rows, videos = _extract_init_goal(
            ds, [0, 1], [0, 0], 3
        )

        assert len(init_rows) == len(goal_rows) == 2
        # Per-episode `states` dims stay per-row — no stacking heuristics.
        assert init_rows[0]['states'].shape == (4,)
        assert init_rows[1]['states'].shape == (7,)
        # Episode-scoped columns land in the init rows only.
        assert init_rows[0]['model_xml'] == '<scene 0/>'
        assert init_rows[1]['ep_meta'] == b'meta-1'
        assert set(goal_rows[0]) == {
            'goal',
            'goal_proprio',
            'goal_states',
            'goal_seed',
        }
        # init = first chunk step, goal = last (start + goal_offset).
        np.testing.assert_array_equal(init_rows[0]['proprio'], [0.0, 1.0])
        np.testing.assert_array_equal(goal_rows[0]['goal_proprio'], [6.0, 7.0])
        np.testing.assert_array_equal(init_rows[1]['proprio'], [10.0, 11.0])
        # pixels permuted to HWC; one panel video per episode.
        assert init_rows[0]['pixels'].shape == (3, 3, 3)
        assert len(videos) == 2
        assert videos[0].shape == (4, 3, 3, 3)


class TestEvaluateFromDataset:
    def test_method_path_receives_rows(self):
        world = _make_world_with(
            [lambda: DatasetResetEnv(3) for _ in range(2)]
        )
        results = world.evaluate(
            dataset=FakeEvalDataset(),
            episodes_idx=[0, 1],
            start_steps=[0, 0],
            goal_offset=3,
            eval_budget=5,
        )

        assert results['seeds'] == [100, 101]
        assert results['success_rate'] == 100.0
        for i in range(2):
            env = world.envs.envs[i]
            assert len(env.received) == 1
            init_row, goal_row = env.received[0]
            assert init_row['model_xml'] == f'<scene {i}/>'
            assert init_row['ep_meta'] == f'meta-{i}'.encode()
            assert init_row['states'].shape == ((4,) if i == 0 else (7,))
            assert goal_row['goal_states'].shape == ((4,) if i == 0 else (7,))

    def test_infos_broadcast_rules(self):
        world = _make_world_with(
            [lambda: DatasetResetEnv(3) for _ in range(2)]
        )
        world.evaluate(
            dataset=FakeEvalDataset(),
            episodes_idx=[0, 1],
            start_steps=[0, 0],
            goal_offset=3,
            eval_budget=5,
        )

        # Uniform goal columns are injected and re-applied every step.
        assert world.infos['goal'].shape == (2, 1, 3, 3, 3)
        assert int(world.infos['goal'][0].max()) == 1
        assert int(world.infos['goal'][1].max()) == 2
        assert world.infos['goal_proprio'].shape == (2, 1, 2)
        # Ragged and episode-scoped columns never reach the infos.
        assert 'goal_states' not in world.infos
        assert 'states' not in world.infos
        assert 'model_xml' not in world.infos
        assert 'ep_meta' not in world.infos

    def test_callables_path_gets_merged_row(self):
        received = []

        class CallableEnv(CounterEnv):
            def _set_state(self, state=None, goal_proprio=None):
                received.append((np.asarray(state), np.asarray(goal_proprio)))

        world = _make_world_with([lambda: CallableEnv(3) for _ in range(2)])
        callables = [
            {
                'method': '_set_state',
                'args': {
                    'state': {'in_dataset': True, 'value': 'proprio'},
                    'goal_proprio': {
                        'in_dataset': True,
                        'value': 'goal_proprio',
                    },
                },
            }
        ]
        world.evaluate(
            dataset=FakeEvalDataset(),
            episodes_idx=[0, 1],
            start_steps=[0, 0],
            goal_offset=3,
            eval_budget=5,
            callables=callables,
        )

        assert len(received) == 2
        np.testing.assert_array_equal(received[0][0], [0.0, 1.0])
        np.testing.assert_array_equal(received[0][1], [6.0, 7.0])
        np.testing.assert_array_equal(received[1][0], [10.0, 11.0])
        np.testing.assert_array_equal(received[1][1], [16.0, 17.0])


class TestCollectEpisodeData:
    def _world(self, tags, max_steps=2):
        return _make_world_with(
            [lambda t=t: EpisodeDataEnv(max_steps, tag=t) for t in tags]
        )

    def test_capture_is_pre_auto_reset(self):
        from stable_worldmodel.data import EPISODE_DATA_KEY, ReplayBuffer

        world = self._world(['env0', 'env1'])
        buf = ReplayBuffer(max_steps=64)
        world.collect(writer=buf, episodes=4, seed=0, progress=False)

        assert buf.num_episodes == 4
        data = buf.get_episode_data()
        # Both envs finish on the same step; each snapshot must belong to
        # the episode that just finished (resets 1 then 2), not to the
        # auto-reset that immediately follows (which would read 2,2,3,3).
        assert data['reset_idx'] == [1, 1, 2, 2]
        assert data['model_xml'] == [
            '<env0 reset="1"/>',
            '<env1 reset="1"/>',
            '<env0 reset="2"/>',
            '<env1 reset="2"/>',
        ]
        # The snapshot rides the reserved key, never the per-step columns.
        assert 'model_xml' not in buf.column_names
        assert EPISODE_DATA_KEY not in buf.column_names

    def test_collect_to_lance_and_evaluate(self, tmp_path):
        from stable_worldmodel.data import LanceDataset

        world = _make_world_with(
            [
                lambda t=t: FlatPixelsEpisodeDataEnv(2, tag=t)
                for t in ('env0', 'env1')
            ]
        )
        out = tmp_path / 'demo.lance'
        world.collect(path=str(out), episodes=2, seed=0, progress=False)

        ds = LanceDataset(path=str(out))
        assert ds.episode_column_names == ['model_xml', 'reset_idx']
        assert ds.get_episode_data()['reset_idx'] == [1, 1]

        eval_world = _make_world_with(
            [lambda: DatasetResetEnv(2) for _ in range(2)]
        )
        results = eval_world.evaluate(
            dataset=ds,
            episodes_idx=[0, 1],
            start_steps=[0, 0],
            goal_offset=1,
            eval_budget=4,
        )

        assert results['success_rate'] == 100.0
        for i in range(2):
            init_row, _ = eval_world.envs.envs[i].received[0]
            assert init_row['model_xml'] == f'<env{i} reset="1"/>'
            assert int(init_row['reset_idx']) == 1
