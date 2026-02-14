"""World environment manager for vectorized Gymnasium environments."""

import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Callable, Sequence
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any

import gymnasium as gym
import h5py
import hdf5plugin
import numpy as np
import torch
from gymnasium.vector import VectorEnv
from loguru import logger as logging
from rich import print
from tqdm import tqdm

from stable_worldmodel.data.utils import get_cache_dir
from stable_worldmodel.policy import Policy

from .wrapper import MegaWrapper, SyncWorld, VariationWrapper


def _make_env(env_name, max_episode_steps, wrappers, **kwargs):
    """Create a gymnasium environment with specified wrappers.

    Factory function for creating environments within a vectorized setup.
    Creates the base environment and applies wrappers in order.

    Args:
        env_name: Name of the gymnasium environment to create.
        max_episode_steps: Maximum steps per episode before truncation.
        wrappers: List of wrapper functions/classes to apply. Each wrapper
            should accept an environment and return a wrapped environment.
        **kwargs: Additional keyword arguments passed to gym.make.

    Returns:
        The wrapped gymnasium environment.

    Example:
        >>> from functools import partial
        >>> wrappers = [partial(MegaWrapper, image_shape=(64, 64))]
        >>> env = _make_env("CartPole-v1", max_episode_steps=500, wrappers=wrappers)
    """
    env = gym.make(env_name, max_episode_steps=max_episode_steps, **kwargs)
    for wrapper in wrappers:
        env = wrapper(env)
    return env


class World:
    """High-level manager for vectorized Gymnasium environments.

    Manages a set of synchronized vectorized environments with automatic
    preprocessing (resizing, frame stacking, goal conditioning).

    Args:
        env_name: Name of the Gymnasium environment to create.
        num_envs: Number of parallel environments.
        image_shape: Target shape for image observations (H, W).
        goal_transform: Optional callable to transform goal observations.
        image_transform: Optional callable to transform image observations.
        seed: Random seed for reproducibility.
        history_size: Number of frames to stack.
        frame_skip: Number of frames to skip per step.
        max_episode_steps: Maximum steps per episode before truncation.
        verbose: Verbosity level (0: silent, >0: info).
        extra_wrappers: List of additional wrappers to apply to each env.
        goal_conditioned: Whether to separate goal from observation.
        **kwargs: Additional keyword arguments passed to `gym.make_vec`.
    """

    def __init__(
        self,
        env_name: str,
        num_envs: int,
        image_shape: tuple[int, int],
        goal_transform: Callable[[Any], Any] | None = None,
        image_transform: Callable[[Any], Any] | None = None,
        seed: int = 2349867,
        history_size: int = 1,
        frame_skip: int = 1,
        max_episode_steps: int = 100,
        verbose: int = 1,
        extra_wrappers: list[Callable] | None = None,
        goal_conditioned: bool = True,
        **kwargs: Any,
    ) -> None:
        wrappers = [
            partial(
                MegaWrapper,
                image_shape=image_shape,
                pixels_transform=image_transform,
                goal_transform=goal_transform,
                history_size=history_size,
                frame_skip=frame_skip,
                separate_goal=goal_conditioned,
            ),
            *(extra_wrappers or []),
        ]

        env_fn = partial(
            _make_env, env_name, max_episode_steps, wrappers, **kwargs
        )
        env_fns = [env_fn for _ in range(num_envs)]
        self.envs: VectorEnv = VariationWrapper(SyncWorld(env_fns))
        self.envs.unwrapped.autoreset_mode = gym.vector.AutoresetMode.DISABLED

        self._history_size = history_size
        self.policy: Policy | None = None
        self.states: dict | None = None
        self.infos: dict = {}
        self.rewards: np.ndarray | None = None
        self.terminateds: np.ndarray | None = None
        self.truncateds: np.ndarray | None = None

        if verbose > 0:
            logging.info(f'🌍🌍🌍 World {env_name} initialized 🌍🌍🌍')

            logging.info('🕹️ 🕹️ 🕹️ Action space 🕹️ 🕹️ 🕹️')
            logging.info(f'{self.envs.action_space}')

            logging.info('👁️ 👁️ 👁️ Observation space 👁️ 👁️ 👁️')
            logging.info(f'{str(self.envs.observation_space)}')

            if self.envs.variation_space is not None:
                logging.info('⚗️ ⚗️ ⚗️ Variation space ⚗️ ⚗️ ⚗️')
                print(self.single_variation_space.to_str())
            else:
                logging.warning('No variation space provided!')

        self.seed = seed

    @property
    def num_envs(self) -> int:
        """Number of parallel environment instances."""
        return self.envs.num_envs

    @property
    def observation_space(self) -> gym.Space:
        """Batched observation space for all environments."""
        return self.envs.observation_space

    @property
    def action_space(self) -> gym.Space:
        """Batched action space for all environments."""
        return self.envs.action_space

    @property
    def variation_space(self) -> gym.Space | None:
        """Batched variation space for domain randomization."""
        return self.envs.variation_space

    @property
    def single_variation_space(self) -> gym.Space | None:
        """Variation space for a single environment instance."""
        return self.envs.single_variation_space

    @property
    def single_action_space(self) -> gym.Space:
        """Action space for a single environment instance."""
        return self.envs.single_action_space

    @property
    def single_observation_space(self) -> gym.Space:
        """Observation space for a single environment instance."""
        return self.envs.single_observation_space

    def close(self, **kwargs: Any) -> None:
        """Close all environments and clean up resources."""
        return self.envs.close(**kwargs)

    def step(self) -> None:
        """Advance all environments by one step using the current policy."""
        # note: reset happens before because of auto-reset, should fix that
        if self.policy is None:
            raise RuntimeError('No policy set. Call set_policy() first.')

        actions = self.policy.get_action(self.infos)
        (
            self.states,
            self.rewards,
            self.terminateds,
            self.truncateds,
            self.infos,
        ) = self.envs.step(actions)

    def reset(
        self,
        seed: int | list[int] | None = None,
        options: dict | None = None,
    ) -> None:
        """Reset all environments to initial states.

        Args:
            seed: Random seed(s) for the environments.
            options: Additional options passed to the environment reset.
        """
        self.states, self.infos = self.envs.reset(seed=seed, options=options)

    def set_policy(self, policy: Policy) -> None:
        """Attach a policy to the world.

        Args:
            policy: The policy instance to use for determining actions.
        """
        self.policy = policy
        self.policy.set_env(self.envs)

        if hasattr(self.policy, 'seed') and self.policy.seed is not None:
            self.policy.set_seed(self.policy.seed)

    def record_video(
        self,
        video_path: str | Path,
        max_steps: int = 500,
        fps: int = 30,
        viewname: str | list[str] = 'pixels',
        seed: int | None = None,
        extension: str = 'mp4',
        options: dict | None = None,
    ) -> None:
        """Record rollout videos for each environment under the current policy.

        Args:
            video_path: Directory path to save the videos.
            max_steps: Maximum steps to record per environment.
            fps: Frames per second for the output video.
            viewname: Key(s) in `infos` containing image data to render.
            seed: Random seed for reset.
            extension: Video file format ('mp4' or 'gif').
            options: Options for reset.
        """

        assert extension in ['mp4', 'gif'], (
            'Unsupported video format. Use "mp4" or "gif".'
        )

        import imageio

        viewname = [viewname] if isinstance(viewname, str) else viewname
        out = [
            imageio.get_writer(
                Path(video_path) / f'env_{i}.{extension}',
                fps=fps,
                codec='libx264',
            )
            for i in range(self.num_envs)
        ]

        self.reset(seed, options)

        for i, o in enumerate(out):
            frames_to_stack = []
            for v_name in viewname:
                frame_data = self.infos[v_name][i]
                # if frame_data has a history dimension, take the last frame
                if frame_data.ndim > 3:
                    frame_data = frame_data[-1]
                frames_to_stack.append(frame_data)
            frame = np.vstack(frames_to_stack)

            if 'goal' in self.infos:
                goal_data = self.infos['goal'][i]
                if goal_data.ndim > 3:
                    goal_data = goal_data[-1]
                frame = np.vstack([frame, goal_data])
            o.append_data(frame)

        for _ in range(max_steps):
            self.step()

            if np.any(self.terminateds) or np.any(self.truncateds):
                break

            for i, o in enumerate(out):
                frames_to_stack = []
                for v_name in viewname:
                    frame_data = self.infos[v_name][i]
                    # if frame_data has a history dimension, take the last frame
                    if frame_data.ndim > 3:
                        frame_data = frame_data[-1]
                    frames_to_stack.append(frame_data)
                frame = np.vstack(frames_to_stack)

                if 'goal' in self.infos:
                    goal_data = self.infos['goal'][i]
                    if goal_data.ndim > 3:
                        goal_data = goal_data[-1]
                    frame = np.vstack([frame, goal_data])
                o.append_data(frame)
        for o in out:
            o.close()
        print(f'Video saved to {video_path}')

    def record_dataset(
        self,
        dataset_name: str,
        episodes: int = 10,
        seed: int | None = None,
        cache_dir: os.PathLike | str | None = None,
        options: dict | None = None,
    ) -> None:
        """Records episodes from the environment into an HDF5 dataset.

        Args:
            dataset_name: Name of the dataset file (without extension).
            episodes: Total number of episodes to record.
            seed: Initial random seed.
            cache_dir: Directory to save the dataset. Defaults to standard cache.
            options: Reset options passed to environments.

        Raises:
            NotImplementedError: If history_size > 1.
        """
        if self._history_size > 1:
            raise NotImplementedError(
                'Frame history > 1 not supported for dataset recording.'
            )

        path = Path(cache_dir or get_cache_dir()) / f'{dataset_name}.h5'
        path.parent.mkdir(parents=True, exist_ok=True)

        self.terminateds = np.zeros(self.num_envs, dtype=bool)
        self.truncateds = np.zeros(self.num_envs, dtype=bool)

        episode_buffers = [defaultdict(list) for _ in range(self.num_envs)]

        h5_kwargs = {
            'name': str(path),
            'mode': 'a' if path.exists() else 'w',
            'libver': 'latest',
        }

        if not path.exists():  # creation only args
            h5_kwargs.update(
                {'fs_strategy': 'page', 'fs_page_size': 4 * 1024 * 1024}
            )

        with h5py.File(**h5_kwargs) as f:
            f.swmr_mode = True  # avoid issue when killed

            if 'ep_len' in f:
                n_ep_recorded = f['ep_len'].shape[0]
                global_step_ptr = (
                    f['ep_offset'][-1] + f['ep_len'][-1]
                    if n_ep_recorded > 0
                    else 0
                )
                initialized = True
                seed = None if seed is None else (seed + n_ep_recorded)
                logging.info(
                    f'Resuming: {n_ep_recorded} episodes already on disk.'
                )
            else:
                n_ep_recorded = 0
                global_step_ptr = 0
                initialized = False

            self.reset(seed, options=options)
            seed = None if seed is None else (seed + self.num_envs)
            self._dump_step_data(episode_buffers)  # record initial state

            with tqdm(
                total=episodes, initial=n_ep_recorded, desc='Recording'
            ) as pbar:
                while n_ep_recorded < episodes:
                    self.step()
                    self._dump_step_data(episode_buffers)

                    for i in range(self.num_envs):
                        if self.terminateds[i] or self.truncateds[i]:
                            finished_ep = self._handle_done_ep(
                                episode_buffers, i, n_ep_recorded
                            )

                            # lazy dataset initialization
                            if not initialized:
                                self._init_h5_datasets(f, finished_ep)
                                initialized = True

                            # contiguous writing
                            steps_written = self._write_episode(
                                f, finished_ep, global_step_ptr
                            )
                            global_step_ptr += steps_written
                            n_ep_recorded += 1
                            pbar.update(1)

                            f.flush()  # flush metadata to avoid corruption

                            if n_ep_recorded >= episodes:
                                break

                            # reset terminated env and record initial state
                            n_seed = (
                                None
                                if seed is None
                                else (seed + n_ep_recorded)
                            )
                            self._reset_single_env(i, n_seed, options)
                            self._dump_step_data(episode_buffers, env_idx=i)

        logging.info(f'Recording complete. Total frames: {global_step_ptr}')

    def _init_h5_datasets(
        self, f: h5py.File, sample_episode: dict[str, list[Any]]
    ) -> None:
        """Initialize resizable HDF5 datasets based on the first episode.

        Args:
            f: The open HDF5 file handle.
            sample_episode: A dictionary containing data from a single episode,
                used to determine shapes and dtypes.
        """
        for key, data_list in sample_episode.items():
            if key in ['ep_len', 'ep_idx', 'policy']:
                continue

            # determine array shape and dtype from sample data
            sample_data = np.array(data_list[0])
            shape = (0,) + sample_data.shape
            maxshape = (None,) + sample_data.shape

            # determine chunk size and compression
            if sample_data.ndim >= 2:
                chunks = (100,) + sample_data.shape
                compression = hdf5plugin.Blosc(
                    cname='lz4', clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE
                )

            else:
                chunks = (1000,) + sample_data.shape
                compression = None

            dtype = sample_data.dtype
            if np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.bytes_):
                dtype = h5py.string_dtype()

            f.create_dataset(
                key,
                shape=shape,
                maxshape=maxshape,
                dtype=dtype,
                chunks=chunks,
                compression=compression,
            )

        # index metadata
        f.create_dataset(
            'ep_offset', shape=(0,), maxshape=(None,), dtype=np.int64
        )
        f.create_dataset(
            'ep_len', shape=(0,), maxshape=(None,), dtype=np.int32
        )

        # per-step episode index
        f.create_dataset(
            'ep_idx',
            shape=(0,),
            maxshape=(None,),
            dtype=np.int32,
            chunks=(1000,),
        )

    def _reset_single_env(
        self,
        env_idx: int,
        seed: int | None = None,
        options: dict | None = None,
    ) -> None:
        """Reset a single environment and update infos dict.

        Args:
            env_idx: Index of the environment to reset.
            seed: Random seed for this specific environment.
            options: Reset options.
        """
        self.envs.unwrapped._autoreset_envs = np.zeros(self.num_envs)
        _, infos = self.envs.envs[env_idx].reset(seed=seed, options=options)

        for k, v in infos.items():
            if k in self.infos:
                self.infos[k][env_idx] = v

    def _handle_done_ep(
        self,
        tmp_buffer: list[dict[str, list[Any]]],
        env_idx: int,
        n_ep_recorded: int,
    ) -> dict[str, list[Any]]:
        """Prepare the episode buffer for writing.

        Args:
            tmp_buffer: List of dictionaries accumulating step data per env.
            env_idx: Index of the environment that finished an episode.
            n_ep_recorded: Number of episodes recorded so far.

        Returns:
            A dictionary containing the complete episode data.
        """
        ep_buffer = tmp_buffer[env_idx]

        # left-shift actions to align with observations i.e. (o_t, a_t)
        if 'action' in ep_buffer:
            actions = ep_buffer['action']
            nan = actions.pop(0)
            actions.append(nan)

        # Extract a copy and clear the temporary buffer
        out = {k: list(v) for k, v in ep_buffer.items()}
        ep_buffer.clear()
        self.terminateds[env_idx] = False
        self.truncateds[env_idx] = False

        # Add episode index to all steps
        ep_len = len(out['step_idx'])
        out['ep_idx'] = [n_ep_recorded] * ep_len

        return out

    def _write_episode(
        self, f: h5py.File, ep_data: dict[str, list[Any]], global_ptr: int
    ) -> int:
        """Write a single contiguous episode to the HDF5 file.

        Args:
            f: The open HDF5 file handle.
            ep_data: The episode data dictionary.
            global_ptr: The global step index where this episode starts.

        Returns:
            The length of the episode written.
        """
        ep_len = len(ep_data['step_idx'])

        # append data to each dataset
        for key in ep_data:
            if key in ['ep_len', 'policy']:
                continue

            ds = f[key]
            curr_size = ds.shape[0]
            ds.resize(curr_size + ep_len, axis=0)
            ds[curr_size:] = np.array(ep_data[key])

        # update metadata
        meta_idx = f['ep_offset'].shape[0]
        f['ep_offset'].resize(meta_idx + 1, axis=0)
        f['ep_len'].resize(meta_idx + 1, axis=0)

        f['ep_offset'][meta_idx] = global_ptr
        f['ep_len'][meta_idx] = ep_len

        return ep_len

    def _dump_step_data(
        self,
        tmp_buffer: list[dict[str, list[Any]]],
        env_idx: int | None = None,
    ) -> None:
        """Append current step data to temporary episode buffers.

        Args:
            tmp_buffer: List of dictionaries accumulating step data.
            env_idx: Optional index to dump data for a single environment.
                If None, dumps for all environments.
        """
        env_indices = range(self.num_envs) if env_idx is None else [env_idx]

        for col, data in self.infos.items():
            if col.startswith('_'):
                continue

            # normalize data shape and type
            if isinstance(data, np.ndarray):
                data = (
                    np.squeeze(data, axis=1)
                    if data.ndim > 1 and data.shape[1] == 1
                    else data
                )
                if data.dtype == object:
                    data = np.concatenate(data).tolist()

            # append to buffers
            for i in env_indices:
                env_data = (
                    data[i].copy()
                    if isinstance(data[i], np.ndarray)
                    else data[i]
                )
                tmp_buffer[i][col].append(env_data)

    def evaluate(
        self,
        episodes: int = 10,
        eval_keys: list[str] | None = None,
        seed: int | None = None,
        options: dict | None = None,
        dump_every: int = -1,
    ) -> dict:
        """Evaluate the current policy over multiple episodes.

        Args:
            episodes: Number of episodes to evaluate.
            eval_keys: List of keys in `infos` to collect and return.
            seed: Random seed for evaluation.
            options: Reset options.
            dump_every: Interval to save intermediate results (for long evals).

        Returns:
            Dictionary containing success rates, seeds, and collected keys.
        """
        options = options or {}

        results: dict = {
            'episode_count': 0,
            'success_rate': 0.0,
            'episode_successes': np.zeros(episodes),
            'seeds': np.zeros(episodes, dtype=np.int32),
        }

        if eval_keys:
            for key in eval_keys:
                results[key] = np.zeros(episodes)

        self.terminateds = np.zeros(self.num_envs)
        self.truncateds = np.zeros(self.num_envs)

        episode_idx = np.arange(self.num_envs)
        self.reset(seed=seed, options=options)
        root_seed = seed + self.num_envs if seed is not None else None

        eval_done = False

        # determine "unique" hash for this eval run
        config = {
            'episodes': episodes,
            'eval_keys': tuple(sorted(eval_keys)) if eval_keys else None,
            'seed': seed,
            'options': tuple(sorted(options.items())) if options else None,
            'dump_every': dump_every,
        }

        config_str = json.dumps(config, sort_keys=True)
        run_hash = hashlib.sha256(config_str.encode()).hexdigest()[:8]
        run_tmp_path = Path(f'eval_tmp_{run_hash}.npy')

        # load back intermediate results if file exists
        if run_tmp_path.exists():
            tmp_results = np.load(run_tmp_path, allow_pickle=True).item()
            results.update(tmp_results)

            ep_count = results['episode_count']
            episode_idx = np.arange(ep_count, ep_count + self.num_envs)

            # reset seed where we left off
            last_seed = seed + ep_count if seed is not None else None
            self.reset(seed=last_seed, options=options)

            logging.success(
                f'Found existing eval tmp file {run_tmp_path}, resuming from episode {ep_count}/{episodes}'
            )

        while True:
            self.step()

            # start new episode for done envs
            for i in range(self.num_envs):
                if self.terminateds[i] or self.truncateds[i]:
                    # record eval info
                    ep_idx = episode_idx[i]
                    results['episode_successes'][ep_idx] = self.terminateds[i]
                    results['seeds'][ep_idx] = self.envs.envs[
                        i
                    ].unwrapped.np_random_seed

                    if eval_keys:
                        for key in eval_keys:
                            assert key in self.infos, (
                                f'key {key} not found in infos'
                            )
                            results[key][ep_idx] = self.infos[key][i]

                    # determine new episode idx
                    # re-reset env with seed and options (no supported by auto-reset)
                    new_seed = (
                        root_seed + results['episode_count']
                        if seed is not None
                        else None
                    )
                    next_ep_idx = episode_idx.max() + 1
                    episode_idx[i] = next_ep_idx
                    results['episode_count'] += 1

                    # break if enough episodes evaluated
                    if results['episode_count'] >= episodes:
                        eval_done = True
                        if run_tmp_path.exists():
                            logging.info(
                                f'Eval done, deleting tmp file {run_tmp_path}'
                            )
                            os.remove(run_tmp_path)
                        break

                    # dump temporary results in a file
                    if dump_every > 0 and (
                        results['episode_count'] % dump_every == 0
                    ):
                        np.save(run_tmp_path, results)
                        logging.success(
                            f'Dumped intermediate eval results to {run_tmp_path} ({results["episode_count"]}/{episodes})'
                        )
                    self.envs.unwrapped._autoreset_envs = np.zeros(
                        (self.num_envs,)
                    )
                    _, infos = self.envs.envs[i].reset(
                        seed=new_seed, options=options
                    )

                    for k, v in infos.items():
                        if k not in self.infos:
                            continue
                        # Convert to array and extract scalar to preserve dtype
                        self.infos[k][i] = np.asarray(v)

            if eval_done:
                break

        # compute success rate
        results['success_rate'] = (
            float(np.sum(results['episode_successes'])) / episodes * 100.0
        )

        assert results['episode_count'] == episodes, (
            f'episode_count {results["episode_count"]} != episodes {episodes}'
        )

        assert np.unique(results['seeds']).shape[0] == episodes, (
            'Some episode seeds are identical!'
        )

        return results

    def evaluate_from_dataset(
        self,
        dataset: Any,
        episodes_idx: Sequence[int],
        start_steps: Sequence[int],
        goal_offset_steps: int,
        eval_budget: int,
        callables: list[dict] | None = None,
        save_video: bool = True,
        video_path: str | Path = './',
    ) -> dict:
        """Evaluate the policy starting from states sampled from a dataset.

        Args:
            dataset: The source dataset to sample initial states/goals from.
            episodes_idx: Indices of episodes to sample from.
            start_steps: Step indices within those episodes to start from.
            goal_offset_steps: Number of steps ahead to look for the goal.
            eval_budget: Maximum steps allowed for the agent to reach the goal.
            callables: Optional list of method calls to setup the env.
            save_video: Whether to save rollout videos.
            video_path: Path to save videos.

        Returns:
            Dictionary containing success rates and other metrics.

        Raises:
            ValueError: If input sequence lengths mismatch or don't match num_envs.
        """
        assert (
            self.envs.envs[0].spec.max_episode_steps is None
            or self.envs.envs[0].spec.max_episode_steps >= goal_offset_steps
        ), 'env max_episode_steps must be greater than eval_budget'

        ep_idx_arr = np.array(episodes_idx)
        start_steps_arr = np.array(start_steps)
        end_steps = start_steps_arr + goal_offset_steps

        if not (len(ep_idx_arr) == len(start_steps_arr)):
            raise ValueError(
                'episodes_idx and start_steps must have the same length'
            )

        if len(ep_idx_arr) != self.num_envs:
            raise ValueError(
                'Number of episodes to evaluate must match number of envs'
            )

        data = dataset.load_chunk(ep_idx_arr, start_steps_arr, end_steps)
        columns = dataset.column_names

        # keep relevant part of the chunk
        init_step_per_env: dict[str, list[Any]] = defaultdict(list)
        goal_step_per_env: dict[str, list[Any]] = defaultdict(list)

        for i, ep in enumerate(data):
            for col in columns:
                if col.startswith('goal'):
                    continue
                if col.startswith('pixels'):
                    # permute channel to be last
                    ep[col] = ep[col].permute(0, 2, 3, 1)

                if not isinstance(ep[col], (torch.Tensor | np.ndarray)):
                    continue

                init_data = ep[col][0]
                goal_data = ep[col][-1]

                # TODO handle that better
                if not isinstance(init_data, (np.ndarray | torch.Tensor)):
                    logging.warning(
                        f'Data type {type(init_data)} for column {col} not supported, yet skipping conversion'
                    )
                    continue

                init_data = (
                    init_data.numpy()
                    if isinstance(init_data, torch.Tensor)
                    else init_data
                )
                goal_data = (
                    goal_data.numpy()
                    if isinstance(goal_data, torch.Tensor)
                    else goal_data
                )

                init_step_per_env[col].append(init_data)
                goal_step_per_env[col].append(goal_data)

        init_step = {
            k: np.stack(v) for k, v in deepcopy(init_step_per_env).items()
        }

        goal_step = {}
        for key, value in goal_step_per_env.items():
            key = 'goal' if key == 'pixels' else f'goal_{key}'
            goal_step[key] = np.stack(value)

        # get dataset info
        seeds = init_step.get('seed')
        # get dataset variation
        vkey = 'variation.'
        variations_dict = {
            k.removeprefix(vkey): v
            for k, v in init_step.items()
            if k.startswith(vkey)
        }

        options = [{} for _ in range(self.num_envs)]

        if len(variations_dict) > 0:
            for i in range(self.num_envs):
                options[i]['variation'] = list(variations_dict.keys())
                options[i]['variation_values'] = {
                    k: v[i] for k, v in variations_dict.items()
                }

        init_step.update(deepcopy(goal_step))
        self.reset(seed=seeds, options=options)  # set seeds for all envs

        # apply callable list (e.g used for set initial position if not access to seed)
        callables = callables or []
        for i, env in enumerate(self.envs.unwrapped.envs):
            env_unwrapped = env.unwrapped

            for spec in callables:
                method_name = spec['method']
                if not hasattr(env_unwrapped, method_name):
                    logging.warning(
                        f'Env {env_unwrapped} has no method {method_name}, skipping callable'
                    )
                    continue

                method = getattr(env_unwrapped, method_name)
                args = spec.get('args', spec)

                # prepare args
                prepared_args = {}
                for args_name, args_data in args.items():
                    value = args_data.get('value', None)
                    is_in_datset = args_data.get('in_dataset', True)

                    if is_in_datset:
                        if value not in init_step:
                            logging.warning(
                                f'Col {value} not found in dataset, skipping callable for env {env_unwrapped}'
                            )
                            continue
                        prepared_args[args_name] = deepcopy(
                            init_step[value][i]
                        )
                    else:
                        prepared_args[args_name] = args_data.get('value')

                # call method with prepared args
                method(**prepared_args)

        for i, env in enumerate(self.envs.unwrapped.envs):
            env_unwrapped = env.unwrapped

            # TODO remove this
            if 'goal_state' in init_step and 'goal_state' in goal_step:
                assert np.array_equal(
                    init_step['goal_state'][i], goal_step['goal_state'][i]
                ), 'Goal state info does not match at reset'

        results: dict = {
            'success_rate': 0.0,
            'episode_successes': np.zeros(len(episodes_idx)),
            'seeds': seeds,
        }

        # expend all data to the right shape (x, y, (original_shape))
        shape_prefix = next(iter(self.infos.values())).shape[:2]

        # TODO get the data from the previous step in the dataset for history
        init_step = {
            k: np.broadcast_to(v[:, None, ...], shape_prefix + v.shape[1:])
            for k, v in init_step.items()
        }
        goal_step = {
            k: np.broadcast_to(v[:, None, ...], shape_prefix + v.shape[1:])
            for k, v in goal_step.items()
        }

        # update the reset with our new init and goal infos
        self.infos.update(deepcopy(init_step))
        self.infos.update(deepcopy(goal_step))

        if 'goal' in goal_step and 'goal' in self.infos:
            assert np.allclose(self.infos['goal'], goal_step['goal']), (
                'Goal info does not match'
            )

        target_frames = torch.stack([ep['pixels'] for ep in data]).numpy()
        video_frames = np.empty(
            (self.num_envs, eval_budget, *self.infos['pixels'].shape[-3:]),
            dtype=np.uint8,
        )

        # run normal evaluation for eval_budget and record video
        for i in range(eval_budget):
            video_frames[:, i] = self.infos['pixels'][:, -1]
            self.infos.update(deepcopy(goal_step))
            self.step()
            results['episode_successes'] = np.logical_or(
                results['episode_successes'], self.terminateds
            )
            # for auto-reset
            self.envs.unwrapped._autoreset_envs = np.zeros((self.num_envs,))

        video_frames[:, -1] = self.infos['pixels'][:, -1]

        n_episodes = len(episodes_idx)

        # compute success rate
        results['success_rate'] = (
            float(np.sum(results['episode_successes'])) / n_episodes * 100.0
        )

        # save video if required
        if save_video:
            import imageio

            target_len = target_frames.shape[1]
            video_path_obj = Path(video_path)
            video_path_obj.mkdir(parents=True, exist_ok=True)
            for i in range(self.num_envs):
                out = imageio.get_writer(
                    video_path_obj / f'rollout_{i}.mp4',
                    fps=15,
                    codec='libx264',
                )
                goals = np.vstack([target_frames[i, -1], target_frames[i, -1]])
                for t in range(eval_budget):
                    stacked_frame = np.vstack(
                        [video_frames[i, t], target_frames[i, t % target_len]]
                    )
                    frame = np.hstack([stacked_frame, goals])
                    out.append_data(frame)
                out.close()
            print(f'Video saved to {video_path_obj}')

        if results['seeds'] is not None:
            assert np.unique(results['seeds']).shape[0] == n_episodes, (
                'Some episode seeds are identical!'
            )

        return results
