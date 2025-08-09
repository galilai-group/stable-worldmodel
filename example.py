import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np

from PIL import Image, ImageEnhance
from moviepy import ImageSequenceClip

import xenoworlds


def reshape_video(v, n_cols=5):
		"""Helper function to reshape videos."""
		if v.ndim == 4:
			v = v[None,]

		_, t, h, w, c = v.shape

		if n_cols is None:
			# Set n_cols to the square root of the number of videos.
			n_cols = np.ceil(np.sqrt(v.shape[0])).astype(int)
		if v.shape[0] % n_cols != 0:
			len_addition = n_cols - v.shape[0] % n_cols
			v = np.concatenate((v, np.zeros(shape=(len_addition, t, h, w, c))), axis=0)
		n_rows = v.shape[0] // n_cols

		v = np.reshape(v, newshape=(n_rows, n_cols, t, h, w, c))
		v = np.transpose(v, axes=(2, 5, 0, 3, 1, 4))
		v = np.reshape(v, newshape=(t, c, n_rows * h, n_cols * w))

		return v


def frames_list_to_array(frames_list=None, n_cols=5):
		"""

		It takes a list of videos and reshapes them into a single video with the specified number of columns.

		Args:
			frames_list: List of videos. Each video should be a numpy array of shape (t, h, w, c).
			n_cols: Number of columns for the reshaped video. If None, it is set to the square root of the number of videos.
		"""
		# Pad videos to the same length.
		max_length = max([len(frames) for frames in frames_list])
		for i, frames in enumerate(frames_list):
			assert frames.dtype == np.uint8

			# Decrease brightness of the padded frames.
			final_frame = frames[-1]
			final_image = Image.fromarray(final_frame)
			enhancer = ImageEnhance.Brightness(final_image)
			final_image = enhancer.enhance(0.5)
			final_frame = np.array(final_image)

			pad = np.repeat(final_frame[np.newaxis, ...], max_length - len(frames), axis=0)
			frames_list[i] = np.concatenate([frames, pad], axis=0)

			# Add borders.
			frames_list[i] = np.pad(frames_list[i], ((0, 0), (1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
		frames_array = np.array(frames_list)  # (n, t, h, w, c)

		frames_array = reshape_video(frames_array, n_cols)  # (t, c, nr * h, nc * w)

		return frames_array


if __name__ == "__main__":
    # run with MUJOCO_GL=egl python example.py

    # gym.register_envs(gymnasium_robotics)
    # envs = gym.envs.registry.keys()
    # print(envs)
    # asdf
    wrappers = [
        # lambda x: RecordVideo(x, video_folder="./videos"),
        # lambda x: xenoworlds.wrappers.AddRenderObservation(x, render_only=False),
        # lambda x: xenoworlds.wrappers.TransformObservation(x),
    ]
    world = xenoworlds.World(
        "xenoworlds/PushT-v1", num_envs=4, wrappers=wrappers, max_episode_steps=100
    )

    world_model = xenoworlds.DummyWorldModel(
        image_shape=(3, 224, 224), action_dim=world.single_action_space.shape[0]
    )

    # -- create a planning policy with a gradient descent solver
    # solver = xenoworlds.solver.GDSolver(world_model, n_steps=100, action_space=world.action_space)
    # policy = xenoworlds.policy.PlanningPolicy(world, solver)
    # -- create a random policy
    policy = xenoworlds.policy.RandomPolicy(world)

    # -- run evaluation
    evaluator = xenoworlds.evaluator.Evaluator(world, policy)
    data = evaluator.run(episodes=5, video_episodes=5)
    # data will be a dict with all the collected metrics
    
    # visualize a rollout video (e.g. for debugging purposes)
    frames_list = data["frames_list"]
    frames_array = frames_list_to_array(frames_list, n_cols=len(frames_list))
    frames_array = np.transpose(frames_array, (0, 2, 3, 1))
    clip = ImageSequenceClip([frames_array[i] for i in range(len(frames_array))], fps=15)
    clip.write_videofile(os.path.join("test_videos_data_caching", f'eval_video.mp4'))
