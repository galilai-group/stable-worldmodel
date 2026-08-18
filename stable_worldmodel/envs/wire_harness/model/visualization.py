"""
VISUALIZATION UTILITIES - functions from the original script
"""

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os
import numpy as np

# One color per mover (up to 5): RED, GREEN, YELLOW, PURPLE, ORANGE
_TARGET_COLORS = ['red', 'green', 'yellow', 'purple', 'orange']

# Workspace bounds matching the MuJoCo XML
_X_MIN, _X_MAX = 0.0, 6.72
_Y_MIN, _Y_MAX = 0.0, 3.84


_MOVER_MARKERS = [
    's',
    'D',
    '^',
    'P',
    '*',
]  # square, diamond, triangle, plus, star


def render_map_panel(
    mover_positions_or_x=None,
    targets_list_or_y=None,
    current_indices_or_targets=None,
    current_target_idx=None,
    width: int = 640,
    height: int = 352,
    *,
    # new-style keyword args (used by v0.1+)
    mover_positions=None,
    targets_list=None,
    current_indices=None,
    show_only_current: bool = False,
) -> np.ndarray:
    """
    Render a top-down 2-D map panel showing mover positions and target markers.

    Supports two call styles:

    Old (v0, single mover):
        render_map_panel(x, y, targets, current_target_idx, width=..., height=...)

    New (v0.1+, N movers):
        render_map_panel(
            mover_positions=[(x1,y1),(x2,y2),...],
            targets_list=[[t1_1,...],[t2_1,...],...],
            current_indices=[idx1, idx2, ...],
            width=..., height=...,
            show_only_current=True,  # show only the next target per mover
        )
    """
    # Resolve call style
    if mover_positions is not None:
        pass  # new-style kwargs already set
    elif isinstance(mover_positions_or_x, (int, float)):
        # Old positional style: (x, y, targets, current_idx)
        mover_positions = [(mover_positions_or_x, targets_list_or_y)]
        targets_list = [current_indices_or_targets]
        current_indices = [current_target_idx]
    else:
        # New positional style: (mover_positions, targets_list, current_indices)
        mover_positions = mover_positions_or_x
        targets_list = targets_list_or_y
        current_indices = current_indices_or_targets

    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes([0.13, 0.14, 0.82, 0.68])

    for mi, ((mx, my), tgts, cur_idx) in enumerate(
        zip(mover_positions, targets_list, current_indices)
    ):
        marker = _MOVER_MARKERS[mi % len(_MOVER_MARKERS)]
        mover_color = _TARGET_COLORS[mi % len(_TARGET_COLORS)]
        n = len(tgts)

        if show_only_current:
            # Show only the current (next) target for this mover
            if cur_idx < n:
                tx, ty = tgts[cur_idx]
                ax.plot(
                    tx,
                    ty,
                    marker=marker,
                    color=mover_color,
                    markersize=11,
                    alpha=1.0,
                    zorder=3,
                    markeredgecolor='white',
                    markeredgewidth=0.8,
                )
                ax.text(
                    tx,
                    ty + 0.14,
                    f'Config {cur_idx + 1}',
                    ha='center',
                    va='bottom',
                    fontsize=6.5,
                    color=mover_color,
                    alpha=1.0,
                    fontweight='bold',
                )
        else:
            # Target markers — all use the mover's own color, faded when reached
            for j, (tx, ty) in enumerate(tgts):
                reached = j < cur_idx
                alpha = 0.25 if reached else 1.0
                ax.plot(
                    tx,
                    ty,
                    marker=marker,
                    color=mover_color,
                    markersize=11,
                    alpha=alpha,
                    zorder=3,
                    markeredgecolor='white',
                    markeredgewidth=0.8,
                )
                ax.text(
                    tx,
                    ty + 0.14,
                    f'Config {j + 1}',
                    ha='center',
                    va='bottom',
                    fontsize=6.5,
                    color=mover_color,
                    alpha=alpha,
                    fontweight='bold',
                )

        # Mover position (white dot, edged in mover color)
        ax.plot(
            mx,
            my,
            'o',
            color='white',
            markersize=8,
            zorder=4,
            markeredgecolor=mover_color,
            markeredgewidth=1.5,
        )

    # Axes styling
    ax.set_xlim(_X_MIN, _X_MAX)
    ax.set_ylim(_Y_MIN, _Y_MAX)
    ax.set_xlabel('X [m]', fontsize=8, color='white')
    ax.set_ylabel('Y [m]', fontsize=8, color='white')
    ax.tick_params(colors='white', labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor('#555')
    ax.set_facecolor('#1a1a2e')
    ax.grid(True, color='#333', linewidth=0.5, alpha=0.6)
    fig.patch.set_facecolor('#0f0f1a')

    # Title — show current target configuration
    parts = [
        f'Target Config {current_indices[i] + 1}'
        for i in range(len(mover_positions))
    ]
    ax.set_title(' | '.join(parts), fontsize=8, color='white', pad=4)

    fig.canvas.draw()
    canvas_w, canvas_h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = buf.reshape(canvas_h, canvas_w, 3)
    plt.close(fig)

    if img.shape[:2] != (height, width):
        from PIL import Image as _PIL

        img = np.array(
            _PIL.fromarray(img).resize((width, height), _PIL.LANCZOS)
        )

    return img


def reward_plot(env, path):
    """
    ORIGINAL FUNCTION from the main script.
    Creates a plot of the rewards over time for all movers.

    What this function does:
    1. Creates a Matplotlib plot
    2. Plots each mover's reward history
    3. Uses the mover colors (red, green, yellow, purple, orange)
    4. Saves it as a PNG file

    Args:
        env: Environment object with the movers and their reward_list
        path: Output path for the PNG file
    """
    # X-axis: time steps (same length for all movers)
    x_values = range(len(env.movers[0].reward_list))

    # Create figure with a specific size
    fig1, ax1 = plt.subplots(figsize=(8, 5))

    # Plot the reward history for each mover with its color
    ax1.plot(
        x_values, env.movers[0].reward_list, label='Reward Red', color='red'
    )
    ax1.plot(
        x_values,
        env.movers[1].reward_list,
        label='Reward Green',
        color='green',
    )
    ax1.plot(
        x_values,
        env.movers[2].reward_list,
        label='Reward Yellow',
        color='yellow',
    )
    ax1.plot(
        x_values,
        env.movers[3].reward_list,
        label='Reward Purple',
        color='purple',
    )
    ax1.plot(
        x_values,
        env.movers[4].reward_list,
        label='Reward Orange',
        color='orange',
    )

    # Axis labels and title
    ax1.set_xlabel('Time / Episodes')
    ax1.set_ylabel('Reward')
    ax1.set_title('Reward History')

    # Show legend
    ax1.legend()

    # Save to file
    fig1.savefig(path)

    # Close figure to free memory
    plt.close(fig1)


class VideoRecorder:
    """
    Class for video recording of the simulation.
    Based on the video functions from the original Environment.
    """

    def __init__(self, width=640, height=352):
        """
        Initializes the video recorder.

        Args:
            width: Video width in pixels (default: 640)
            height: Video height in pixels (default: 352)
        """
        self.video_w = width
        self.video_h = height
        self.video_writer = None
        self._video_path = None
        self._frame_count = 0

    def start_video(self, path, fps=30):
        """
        ORIGINAL FUNCTION from Environment.
        Starts video recording of the simulation.

        What this function does:
        1. Converts the path to an absolute path
        2. Creates directories if needed
        3. Initializes the imageio writer with H.264 codec
        4. Resets the frame counter

        Args:
            path: Output path for the video (e.g. "videos/simulation.mp4")
            fps: Frames per second (default: 30)
        """
        # Use an absolute path for reliability
        path = os.path.abspath(path)

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save path and reset frame counter
        self._video_path = path
        self._frame_count = 0

        try:
            # Initialize video writer with H.264 codec
            # macro_block_size=1 for better quality on small details
            self.video_writer = imageio.get_writer(
                path,
                fps=fps,
                codec='libx264',  # H.264 codec for MP4
                macro_block_size=1,
            )
            print(
                f'[Video] Recording started → {path} ({self.video_w}x{self.video_h} @ {fps}fps)'
            )
        except Exception as e:
            # If the video writer can't be created
            self.video_writer = None
            print(f'[Video] Could not open MP4 writer: {e}')

    def capture_frame(self, renderer, data, cam, opt):
        """
        ORIGINAL FUNCTION from Environment.
        Captures a frame for the video.

        What this function does:
        1. Checks whether the video writer is active
        2. Updates the MuJoCo scene
        3. Renders the frame
        4. Adds the frame to the video
        5. Increments the frame counter

        Args:
            renderer: MuJoCo renderer object
            data: MuJoCo simulation data
            cam: MuJoCo camera object
            opt: MuJoCo visualization options
        """
        # Only capture if the writer is active
        if self.video_writer is None:
            return

        # Update the MuJoCo scene with current data
        renderer.update_scene(data, camera=cam, scene_option=opt)

        # Render the frame (returns a numpy array)
        frame = renderer.render()

        # Add the frame to the video
        self.video_writer.append_data(frame)

        # Increment frame counter for statistics
        self._frame_count += 1

    def finish_video(self):
        """
        ORIGINAL FUNCTION from Environment.
        Finishes video recording and saves the file.

        What this function does:
        1. Closes the video writer (saves the file)
        2. Waits briefly for the file to be fully written
        3. Checks whether the file was created successfully
        4. Prints a success or error message
        """
        if self.video_writer is not None:
            try:
                # Close the video writer - this saves the file
                self.video_writer.close()
            finally:
                # Reset writer to None for a clean state
                self.video_writer = None

            # Wait briefly so the OS finishes writing the file
            import time

            time.sleep(0.2)

            # Check whether the file exists and has size > 0
            ok = (
                os.path.exists(self._video_path)
                and os.path.getsize(self._video_path) > 0
            )

            # Print status message
            print(
                f'[Video] Recording finished ({self._frame_count} frames) → {self._video_path} '
                f'{"(OK)" if ok else "(ERROR: file missing/empty)"}'
            )


def integrate_video_recording(env):
    """
    Helper function to integrate video recording into the Environment.

    What this function does:
    - Adds the video methods to the Environment
    - Initializes required variables

    Args:
        env: Environment object
    """
    # Initialize video variables
    env.video_w = 640
    env.video_h = 352
    env.video_writer = None
    env._video_path = None
    env._frame_count = 0

    # Add methods as instance methods
    recorder = VideoRecorder(env.video_w, env.video_h)

    # Bind methods
    env.start_video = recorder.start_video
    env.capture_frame = lambda: recorder.capture_frame(
        env.renderer, env.data, env.cam, env.opt
    )
    env.finish_video = recorder.finish_video
