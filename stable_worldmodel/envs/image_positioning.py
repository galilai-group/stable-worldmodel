from typing import Optional
import numpy as np
import gymnasium as gym
import pygame
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Get the default color cycle from Matplotlib's rcParams
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors_hex = prop_cycle.by_key()["color"]

# Convert hex colors to RGBA tuples
COLORS = np.asarray([mcolors.to_rgba(hex_color) for hex_color in colors_hex])
COLORS = (COLORS * 255).astype(int)
COLORS = [tuple(u) for u in COLORS]


class ImagePositioning(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        resolution: int,
        images: list[Image],
        render_mode: str = None,
        background_power_decay: float = 1,
    ):
        self.resolution = resolution
        self.background_power_decay = background_power_decay

        # Define what the agent can observe
        # Dict space gives us structured, human-readable observations
        self.observation_space = gym.spaces.Dict({
            "current_background": gym.spaces.Box(0, 0.9, shape=(2, 1), dtype=float),
            "current_locations": gym.spaces.Box(
                0, 0.9, shape=(len(images), 2), dtype=float
            ),
            "current_rotations": gym.spaces.Box(
                0, 1, shape=(len(images), 1), dtype=float
            ),
            "target_background": gym.spaces.Box(0, 1, shape=(2, 1), dtype=float),
            "target_locations": gym.spaces.Box(
                0, 0.9, shape=(len(images), 2), dtype=float
            ),
            "target_rotations": gym.spaces.Box(
                0, 1, shape=(len(images), 1), dtype=float
            ),
        })

        # Initialize positions - will be set randomly in reset()
        # Using -1,-1 as "uninitialized" state
        self._current_locations = np.empty(
            self.observation_space["current_locations"].shape, dtype=float
        )
        self._target_locations = np.array(
            self.observation_space["target_locations"].shape, dtype=float
        )
        self._current_rotations = np.empty(
            self.observation_space["current_rotations"].shape, dtype=float
        )
        self._target_rotations = np.array(
            self.observation_space["target_rotations"].shape, dtype=float
        )
        self._current_background = np.empty(
            self.observation_space["current_background"].shape, dtype=float
        )
        self._target_background = np.array(
            self.observation_space["target_background"].shape, dtype=float
        )

        # Define what actions are available (4 directions)
        self.action_space = gym.spaces.Dict({
            "delta_background": gym.spaces.Box(low=-0.1, high=0.1, shape=(2, 1)),
            "delta_locations": gym.spaces.Box(
                low=-0.1, high=0.1, shape=(len(images), 2)
            ),
            "delta_rotations": gym.spaces.Box(
                low=-0.1, high=0.1, shape=(len(images), 1)
            ),
        })
        self.images = [
            ImageOps.expand(img, border=5, fill=c).convert("RGBA")
            for img, c in zip(images, COLORS)
        ]

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agent and target positions
        """
        return {
            "current_background": self._current_background,
            "current_locations": self._current_locations,
            "current_rotations": self._current_rotations,
            "target_background": self._target_background,
            "target_locations": self._target_locations,
            "target_rotations": self._target_rotations,
        }

    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with distance between agent and target
        """
        return {
            "location_distance": np.linalg.norm(
                self._current_locations - self._target_locations, ord=1
            ),
            "rotation_distance": np.linalg.norm(
                self._current_rotations - self._target_rotations, ord=1
            ),
            "background_distance": np.linalg.norm(
                self._current_background - self._target_background, ord=1
            ),
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        # Randomly place the agent anywhere on the grid
        self._current_background = self.np_random.random(size=(2, 1), dtype=float)
        self._current_locations = self.np_random.random(
            size=(len(self.images), 2), dtype=float
        )
        self._current_rotations = self.np_random.random(
            size=(len(self.images), 2), dtype=float
        )
        self._target_background = self.np_random.random(size=(2, 1), dtype=float)
        self._target_locations = self.np_random.random(
            size=(len(self.images), 2), dtype=float
        )
        self._target_rotations = self.np_random.random(
            size=(len(self.images), 1), dtype=float
        )

        white_noise = np.random.randn(self.resolution * 2, self.resolution * 2)
        rows, cols = white_noise.shape
        fft_white_noise = np.fft.fft2(white_noise)
        # Create frequency coordinates
        fy = np.fft.fftfreq(rows)
        fx = np.fft.fftfreq(cols)
        # Create 2D frequency grid
        fx_grid, fy_grid = np.meshgrid(fx, fy)
        # Calculate radial frequency magnitude
        f_magnitude = np.sqrt(fx_grid**2 + fy_grid**2)

        # Avoid division by zero at the DC component (f=0)
        f_magnitude[0, 0] = 1  # Or a small epsilon to prevent singularity

        # Apply the 1/f filter to the frequency magnitudes
        # For power spectral density 1/f, amplitude is 1/sqrt(f)
        pink_filter = (1 / f_magnitude) ** self.background_power_decay
        fft_pink_noise = fft_white_noise * pink_filter
        pink_noise = np.fft.ifft2(fft_pink_noise).real
        pink_noise -= pink_noise.min()
        pink_noise /= pink_noise.max()
        pink_noise = (pink_noise * 255).astype(np.uint8)
        self.pink_noise = np.tile(np.expand_dims(pink_noise, 2), (1, 1, 3))

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0-3 for directions)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        action["delta_background"] = np.clip(
            action["delta_background"],
            self.action_space["delta_background"].low,
            self.action_space["delta_background"].high,
        )
        action["delta_locations"] = np.clip(
            action["delta_locations"],
            self.action_space["delta_locations"].low,
            self.action_space["delta_locations"].high,
        )
        action["delta_rotations"] = np.clip(
            action["delta_rotations"],
            self.action_space["delta_rotations"].low,
            self.action_space["delta_rotations"].high,
        )
        self._current_background = np.clip(
            self._current_background + action["delta_background"],
            self.observation_space["current_background"].low,
            self.observation_space["current_background"].high,
        )
        self._current_locations = np.clip(
            self._current_locations + action["delta_locations"],
            self.observation_space["current_locations"].low,
            self.observation_space["current_locations"].high,
        )
        self._current_rotations = np.clip(
            self._current_rotations + action["delta_rotations"],
            self.observation_space["current_rotations"].low,
            self.observation_space["current_rotations"].high,
        )

        observation = self._get_obs()
        info = self._get_info()
        # Check if agent reached the target
        terminated = (
            info["location_distance"] < 1e-2
            and info["rotation_distance"] < 1e-2
            and info["background_distance"] < 1e-2
        )

        # We don't use truncation in this simple environment
        # (could add a step limit here if desired)
        truncated = False

        # Simple reward structure: +1 for reaching target, 0 otherwise
        # Alternative: could give small negative rewards for each step to encourage efficiency
        reward = 1 if terminated else 0

        return observation, reward, terminated, truncated, info

    def _get_optimal_action(self):
        rotations = self._current_rotations - self._target_rotations
        locations = self._current_locations - self._target_locations
        background = self._current_background - self._target_background
        return dict(
            delta_background=-background,
            delta_locations=-locations,
            delta_rotations=-rotations,
        )

    def render(self, mode="current"):
        if self.render_mode == "rgb_array":
            return self._render_frame(mode=mode)

    def _render_frame(self, mode):
        if self.window is None and self.render_mode in ["human", "rgb_array"]:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.resolution, self.resolution))
        if self.clock is None and self.render_mode in ["human", "rgb_array"]:
            self.clock = pygame.time.Clock()

        # canvas = pygame.Surface((self.resolution, self.resolution))
        # canvas.fill((255, 255, 255))

        # get the image
        if mode == "current":
            x = int(self.resolution * self._current_background[0, 0])
            y = int(self.resolution * self._current_background[1, 0])
            new_background = Image.fromarray(
                self.pink_noise[x : x + self.resolution, y : y + self.resolution :]
            )
        else:
            x = int(self.resolution * self._target_background[0, 0])
            y = int(self.resolution * self._target_background[1, 0])
            new_background = Image.fromarray(
                self.pink_noise[x : x + self.resolution, y : y + self.resolution :]
            )
        for i, img in enumerate(self.images):
            if mode == "current":
                box = [
                    int(self._current_locations[i, 0] * self.resolution),
                    int(self._current_locations[i, 1] * self.resolution),
                    int(self._current_locations[i, 0] * self.resolution + img.height),
                    int(self._current_locations[i, 1] * self.resolution + img.width),
                ]
                new_background.paste(
                    img.rotate(self._current_rotations[i, 0] * 360), box
                )
            else:
                box = [
                    int(self._target_locations[i, 0] * self.resolution),
                    int(self._target_locations[i, 1] * self.resolution),
                    int(self._target_locations[i, 0] * self.resolution + img.height),
                    int(self._target_locations[i, 1] * self.resolution + img.width),
                ]
                new_background.paste(
                    img.rotate(self._target_rotations[i, 0] * 360), box
                )

        # get the surface
        # Get image data, size, and mode from PIL Image
        image_bytes = new_background.tobytes()
        image_size = new_background.size
        image_mode = new_background.mode

        # Create a Pygame Surface from the PIL image data
        pygame_surface = pygame.image.frombytes(image_bytes, image_size, image_mode)
        self.window.blit(pygame_surface, (0, 0))  # Blit at position (0,0)

        # Update the display
        pygame.display.flip()
        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            # self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(pygame_surface)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


if __name__ == "__main__":
    import stable_worldmodel as swm
    import gymnasium as gym
    import matplotlib.pyplot as plt
    import numpy as np
    from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

    # 1. Setup Environment
    # Create a CartPole environment with "rgb_array" render mode to get image data
    images = [
        swm.utils.create_pil_image_from_url(
            "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQK5OnlnP3_GHXI2y1LoIHbMROdN8_DYyLEGg&s"
        ).resize((64, 64)),
        swm.utils.create_pil_image_from_url(
            "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQjrFGrhOLwgYP0cdjTIBEWMpy9MHBcya4c5Q&s"
        ).resize((32, 32)),
    ]

    env = gym.make(
        "swm/ImagePositioning",
        render_mode="rgb_array",
        resolution=224,
        images=images,
        background_power_decay=2,
        max_episode_steps=20,
    )  #
    env = gym.wrappers.AddRenderObservation(env, render_only=False)
    swm.collect.random_action(env, num_episodes=1)
    env = RecordVideo(
        env,
        video_folder="cartpole-agent",  # Folder to save videos
        name_prefix="eval",  # Prefix for video filenames
        episode_trigger=lambda x: True,  # Record every episode
    )

    # 2. Reset the environment to get an initial observation
    observation, info = env.reset()  #
    print(observation)
    print(info)

    # 3. Render the environment to get the image array
    # The render method returns an RGB array when render_mode is "rgb_array"

    # 4. Save the figure
    # Use Matplotlib to display and save the image
    fig, axs = plt.subplots(1, 2)
    rgb_array = env.unwrapped.render()  #
    axs[0].imshow(rgb_array)
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].set_title("Init.")
    rgb_array = env.unwrapped.render(mode="target")
    axs[1].imshow(rgb_array)
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[1].set_title("Target")
    plt.savefig("cartpole_observation.png")
    plt.close()  # Close the plot to free up memory
    for i in range(5):
        action = env.unwrapped._get_optimal_action()
        env.step(action)

    print("Saved CartPole observation as cartpole_observation.png")

    # 5. Close the environment
    env.close()
