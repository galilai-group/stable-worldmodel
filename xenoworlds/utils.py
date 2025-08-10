from .world import World

import os
import requests
from io import BytesIO
from PIL import Image, ImageEnhance
from moviepy import ImageSequenceClip

import numpy as np


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


def save_rollout_videos(frames_list, logdir="test_videos_data_caching"):
    frames_array = frames_list_to_array(frames_list, n_cols=len(frames_list))
    frames_array = np.transpose(frames_array, (0, 2, 3, 1))
    clip = ImageSequenceClip([frames_array[i] for i in range(len(frames_array))], fps=15)
    clip.write_videofile(os.path.join(logdir, f'eval_video.mp4'))


def create_pil_image_from_url(image_url):
    """
    Creates a PIL Image object from a given image URL.

    Args:
        image_url (str): The URL of the image.

    Returns:
        PIL.Image.Image: The PIL Image object, or None if an error occurs.
    """
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        image_data = BytesIO(response.content)
        img = Image.open(image_data)
        return img
    except requests.exceptions.RequestException as e:
        print(image_url)
        print(f"Error downloading image from URL: {e}")
        return None
    except IOError as e:
        print(image_url)
        print(f"Error opening image data with PIL: {e}")
        return None


def set_state(world: World, state):
    if isinstance(world, World):
        envs = world.envs.envs
    else:
        envs = [world.env]
        state = [state]
    for i, env in enumerate(envs):
        if hasattr(env.unwrapped, "state"):
            env.unwrapped.state = state[i]
        elif hasattr(env.unwrapped, "s"):
            env.unwrapped.s = state[i]
        elif hasattr(env.unwrapped, "sim") and hasattr(
            env.unwrapped.sim, "set_state_from_flattened"
        ):
            env.unwrapped.sim.set_state_from_flattened(state[i].view(-1))


if __name__ == "__main__":
    import gymnasium as gym
    import gymnasium_robotics
    import numpy as np
    import xenoworlds
    import matplotlib.pyplot as plt

    wrappers = [
        # lambda x: RecordVideo(x, video_folder="./videos"),
        lambda x: xenoworlds.wrappers.AddRenderObservation(x, render_only=False),
        lambda x: xenoworlds.wrappers.TransformObservation(x),
    ]
    # print(ogbench.make_env_and_datasets(env, env_only=True))
    world = xenoworlds.World(
        "FrozenLake-v1", num_envs=2, wrappers=wrappers, max_episode_steps=2
    )

    world.envs.reset()
    img = world.envs.envs[0].render()
    plt.subplot(2, 4, 1)
    plt.imshow(img)
    plt.subplot(2, 4, 5)
    img = world.envs.envs[1].render()
    plt.imshow(img)

    xenoworlds.set_state(world, np.arange(2))
    img = world.envs.envs[0].render()
    plt.subplot(2, 4, 2)
    plt.imshow(img)
    plt.subplot(2, 4, 6)
    img = world.envs.envs[1].render()
    plt.imshow(img)

    world.envs.reset()
    img = world.envs.envs[0].render()
    plt.subplot(2, 4, 3)
    plt.imshow(img)
    plt.subplot(2, 4, 7)
    img = world.envs.envs[1].render()
    plt.imshow(img)

    xenoworlds.set_state(world, np.asarray([0, 3]))
    img = world.envs.envs[0].render()
    plt.subplot(2, 4, 4)
    plt.imshow(img)
    plt.subplot(2, 4, 8)
    img = world.envs.envs[1].render()
    plt.imshow(img)
    plt.savefig("test_delete.png")
    world.close()
