#!/usr/bin/env python3
"""Convert RoboCasa HDF5 dataset to HuggingFace format for FrameDataset/VideoDataset.

This script converts raw RoboCasa HDF5 demonstrations into a format compatible with
stable_worldmodel's VideoDataset/FrameDataset loaders.

Environment Variables:
    STABLEWM_HOME: Base directory for stable-worldmodel data (default: ~/.stable-worldmodel)
    ROBOCASA_DATA: Path to raw RoboCasa HDF5 data (default: ~/robocasa/datasets)

Usage:
    # Basic conversion (uses environment variables for paths)
    python scripts/convert_robocasa_hdf5.py \\
        --task_names PnPCounterToCab \\
        --mode video

    # With explicit paths
    python scripts/convert_robocasa_hdf5.py \\
        --data_path /path/to/robocasa/datasets \\
        --output_dir /path/to/output \\
        --task_names PnPCounterToCab PnPCounterToSink \\
        --mode video

    # Convert subset for testing
    python scripts/convert_robocasa_hdf5.py \\
        --task_names PnPCounterToCab \\
        --filter_first_episodes 5 \\
        --mode video
"""

import argparse
import json
import logging
import os
from pathlib import Path

import h5py
import imageio.v3 as iio
import numpy as np
from datasets import Dataset, Features, Sequence, Value
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Default paths using STABLEWM_HOME environment variable
STABLEWM_HOME = os.environ.get("STABLEWM_HOME", os.path.expanduser("~/.stable-worldmodel"))
DEFAULT_DATA_PATH = os.environ.get("ROBOCASA_DATA", os.path.expanduser("~/robocasa/datasets"))
DEFAULT_OUTPUT_DIR = os.path.join(STABLEWM_HOME, "robocasa")


def eef_quat_to_euler(eef_quat):
    """Convert quaternion (wxyz) to Euler angles (xyz).

    Args:
        eef_quat: Array of shape (T, 4) with quaternion in [w, x, y, z] format

    Returns:
        Array of shape (T, 3) with Euler angles in xyz order (radians)
    """
    # Convert from [w, x, y, z] to [x, y, z, w] for scipy
    eef_quat_xyzw = eef_quat[:, [1, 2, 3, 0]]
    eef_euler = R.from_quat(eef_quat_xyzw).as_euler("xyz", degrees=False)
    return eef_euler


def gripper_2d_to_1d(gripper_qpos):
    """Convert 2D gripper position to 1D representation.

    Args:
        gripper_qpos: Array of shape (T, 2) for gripper position

    Returns:
        Array of shape (T, 1) for gripper closure state
    """
    return gripper_qpos[:, 0:1] - gripper_qpos[:, 1:2]


def discover_hdf5_files(data_path: str, task_names: list[str], use_human: bool, use_mg: bool) -> list[str]:
    """Find all HDF5 files for specified tasks by searching the directory tree."""
    file_paths = []
    data_path = Path(data_path)

    for task_name in task_names:
        # Search for task directory anywhere in the tree
        task_dirs = list(data_path.rglob(task_name))
        if not task_dirs:
            logger.warning(f"Task {task_name} not found in {data_path}")
            continue

        for task_dir in task_dirs:
            if not task_dir.is_dir():
                continue

            for hdf5_file in task_dir.rglob("*.hdf5"):
                file_str = str(hdf5_file)
                is_mg = "/mg/" in file_str

                # Filter based on human/mg preference
                if is_mg and not use_mg:
                    continue
                if not is_mg and not use_human:
                    continue

                # Only include im128 files
                if "im128" in hdf5_file.name:
                    file_paths.append(file_str)

    return sorted(set(file_paths))


def convert_robocasa_to_huggingface(
    data_path: str,
    output_dir: str,
    task_names: list[str],
    camera_names: list[str] | None = None,
    use_human: bool = True,
    use_mg: bool = True,
    filter_first_episodes: int | None = None,
    manip_only: bool = True,
    mode: str = "video",
):
    """Convert RoboCasa HDF5 to HuggingFace format."""
    camera_names = camera_names or ["robot0_agentview_left"]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if (output_path / "state.json").exists():
        logger.warning(f"Dataset exists at {output_path}. Aborting.")
        return

    file_paths = discover_hdf5_files(data_path, task_names, use_human, use_mg)
    if not file_paths:
        raise ValueError(f"No HDF5 files found for tasks: {task_names}")
    logger.info(f"Found {len(file_paths)} HDF5 files")

    media_dir = output_path / ("videos" if mode == "video" else "img")
    media_dir.mkdir(exist_ok=True)

    records = {
        k: []
        for k in [
            "episode_idx",
            "step_idx",
            "episode_len",
            "action",
            "proprio",
            "state",
            "reward",
            "pixels",
            "task_name",
            "is_mg",
            "model_xml",
            "ep_meta",
        ]
    }
    episode_idx = 0

    for file_path in tqdm(file_paths, desc="Processing files"):
        with h5py.File(file_path, "r") as f:
            env_args = json.loads(f["data"].attrs.get("env_args", "{}"))
            task_name = env_args.get("env_name", "PnPCounterTop")
            demos = sorted([k for k in f["data"].keys() if k.startswith("demo_")], key=lambda x: int(x.split("_")[1]))
            if filter_first_episodes:
                demos = demos[:filter_first_episodes]
            is_mg = "/mg/" in file_path

            for demo_key in tqdm(demos, desc=f"{Path(file_path).name}", leave=False):
                demo = f["data"][demo_key]
                T = demo["actions"].shape[0]
                actions = np.array(demo["actions"][:, :7] if manip_only else demo["actions"][:], dtype=np.float32)
                states = np.array(demo["states"][:], dtype=np.float32) if "states" in demo else np.zeros((T, 1))
                rewards = np.array(demo["rewards"][:], dtype=np.float32) if "rewards" in demo else np.zeros(T)

                # Proprio: Apply same transformation as RoboCasa environment
                # eef_pos(3) + eef_quat(4)->euler(3) + gripper(2)->1d(1) = 7 dims
                obs = demo.get("obs", {})
                if "robot0_eef_pos" in obs and "robot0_eef_quat" in obs and "robot0_gripper_qpos" in obs:
                    eef_pos = np.array(obs["robot0_eef_pos"][:T], dtype=np.float32)  # (T, 3)
                    eef_quat = np.array(obs["robot0_eef_quat"][:T], dtype=np.float32)  # (T, 4)
                    gripper_qpos = np.array(obs["robot0_gripper_qpos"][:T], dtype=np.float32)  # (T, 2)

                    # Apply transformations to match environment proprio format
                    eef_euler = eef_quat_to_euler(eef_quat)  # (T, 4) -> (T, 3)
                    gripper_1d = gripper_2d_to_1d(gripper_qpos)  # (T, 2) -> (T, 1)

                    proprio = np.concatenate([eef_pos, eef_euler, gripper_1d], axis=1)  # (T, 7)
                else:
                    proprio = np.zeros((T, 7), dtype=np.float32)

                # Metadata
                model_xml = demo.attrs.get("model_file", b"")
                model_xml = model_xml.decode() if isinstance(model_xml, bytes) else (model_xml or "")
                ep_meta = demo.attrs.get("ep_meta", b"")
                ep_meta = ep_meta.decode() if isinstance(ep_meta, bytes) else (ep_meta or "")

                # Find and save images
                cam_key = next(
                    (f"{c}_image" for c in camera_names if f"{c}_image" in obs),
                    next((k for k in obs if k.endswith("_image")), None),
                )
                if not cam_key:
                    logger.warning(f"No images in {file_path}/{demo_key}")
                    continue

                images = np.array(obs[cam_key][:], dtype=np.uint8)
                if mode == "video":
                    video_path = media_dir / f"{episode_idx}_pixels.mp4"
                    iio.imwrite(video_path, images)
                    rel_path = f"videos/{episode_idx}_pixels.mp4"
                    for t in range(T):
                        records["pixels"].append(rel_path)
                else:
                    ep_dir = media_dir / str(episode_idx)
                    ep_dir.mkdir(exist_ok=True)
                    for t in range(T):
                        img_path = ep_dir / f"{t}_pixels.jpeg"
                        iio.imwrite(img_path, images[t])
                        records["pixels"].append(f"img/{episode_idx}/{t}_pixels.jpeg")

                for t in range(T):
                    records["episode_idx"].append(episode_idx)
                    records["step_idx"].append(t)
                    records["episode_len"].append(T)
                    records["action"].append(actions[t].tolist())
                    records["proprio"].append(proprio[t].tolist())
                    records["state"].append(states[t].tolist())
                    records["reward"].append(float(rewards[t]))
                    records["task_name"].append(task_name)
                    records["is_mg"].append(is_mg)
                    records["model_xml"].append(model_xml)
                    records["ep_meta"].append(ep_meta)

                episode_idx += 1

    if not episode_idx:
        raise ValueError("No episodes converted")

    # Create dataset with proper features
    action_dim = len(records["action"][0])
    proprio_dim = len(records["proprio"][0])

    features = Features(
        {
            "episode_idx": Value("int32"),
            "step_idx": Value("int32"),
            "episode_len": Value("int32"),
            "pixels": Value("string"),
            "reward": Value("float32"),
            "action": Sequence(Value("float32"), length=action_dim),
            "proprio": Sequence(Value("float32"), length=proprio_dim),
            "state": Sequence(Value("float32")),  # Variable length
            "task_name": Value("string"),
            "is_mg": Value("bool"),
            "model_xml": Value("string"),
            "ep_meta": Value("string"),
        }
    )

    ds = Dataset.from_dict(records, features=features)
    ds.save_to_disk(output_path, num_shards=max(1, episode_idx // 50))
    logger.info(f"Saved {episode_idx} episodes to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert RoboCasa HDF5 to HuggingFace format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables:
  STABLEWM_HOME    Base directory for stable-worldmodel data (default: ~/.stable-worldmodel)
  ROBOCASA_DATA    Path to raw RoboCasa HDF5 data (default: ~/robocasa/datasets)

Examples:
  # Basic conversion using default paths
  python scripts/convert_robocasa_hdf5.py --task_names PnPCounterToCab --mode video

  # Convert multiple tasks with explicit paths
  python scripts/convert_robocasa_hdf5.py \\
      --data_path /path/to/robocasa/datasets \\
      --output_dir /path/to/output \\
      --task_names PnPCounterToCab PnPCounterToSink

  # Convert small subset for testing
  python scripts/convert_robocasa_hdf5.py \\
      --task_names PnPCounterToCab \\
      --filter_first_episodes 5
        """,
    )
    parser.add_argument(
        "--data_path", default=DEFAULT_DATA_PATH, help=f"RoboCasa dataset root path (default: {DEFAULT_DATA_PATH})"
    )
    parser.add_argument(
        "--output_dir", default=DEFAULT_OUTPUT_DIR, help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument("--task_names", nargs="+", required=True, help="Task names to convert")
    parser.add_argument("--camera_names", nargs="+", default=["robot0_agentview_left"])
    parser.add_argument("--no_human", action="store_true", help="Exclude human demos")
    parser.add_argument("--no_mg", action="store_true", help="Exclude MimicGen data")
    parser.add_argument("--filter_first_episodes", type=int, help="Limit episodes per file")
    parser.add_argument("--full_actions", action="store_true", help="Include navigation actions")
    parser.add_argument("--mode", choices=["frame", "video"], default="video", help="Output format")
    args = parser.parse_args()

    convert_robocasa_to_huggingface(
        data_path=args.data_path,
        output_dir=args.output_dir,
        task_names=args.task_names,
        camera_names=args.camera_names,
        use_human=not args.no_human,
        use_mg=not args.no_mg,
        filter_first_episodes=args.filter_first_episodes,
        manip_only=not args.full_actions,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
