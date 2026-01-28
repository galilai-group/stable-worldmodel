"""Stress tests for ImageDataset performance."""

import time
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from stable_worldmodel.data import ImageDataset, HDF5Dataset


def create_large_image_dataset(path: Path, num_episodes: int, steps_per_episode: int):
    """Create a large synthetic ImageDataset for stress testing."""
    path.mkdir(parents=True, exist_ok=True)

    total_steps = num_episodes * steps_per_episode
    ep_lengths = np.array([steps_per_episode] * num_episodes)
    ep_offsets = np.array([i * steps_per_episode for i in range(num_episodes)])

    # Save metadata
    np.savez(path / "ep_len.npz", ep_lengths)
    np.savez(path / "ep_offset.npz", ep_offsets)

    # Save action data
    np.savez(path / "action.npz", np.random.randn(total_steps, 2).astype(np.float32))

    # Save observation data
    np.savez(path / "observation.npz", np.random.randn(total_steps, 10).astype(np.float32))

    # Create pixels folder with images
    pixels_path = path / "pixels"
    pixels_path.mkdir(exist_ok=True)

    print(f"Creating {num_episodes} episodes × {steps_per_episode} steps = {total_steps} images...")

    for ep_idx in range(num_episodes):
        for step_idx in range(steps_per_episode):
            img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(pixels_path / f"ep_{ep_idx}_step_{step_idx}.jpeg", quality=85)

    return total_steps


class TestImageDatasetStress:
    """Stress tests for ImageDataset."""

    @pytest.fixture(scope="class")
    def large_dataset_path(self, tmp_path_factory):
        """Create a moderately large dataset for testing."""
        path = tmp_path_factory.mktemp("stress_test")
        dataset_path = path / "large_dataset"

        # 50 episodes × 100 steps = 5000 images
        num_episodes = 50
        steps_per_episode = 100
        create_large_image_dataset(dataset_path, num_episodes, steps_per_episode)

        return path, "large_dataset", num_episodes, steps_per_episode

    def test_dataset_creation_time(self, large_dataset_path):
        """Test how fast the dataset can be initialized."""
        cache_dir, name, num_episodes, steps_per_episode = large_dataset_path

        start = time.perf_counter()
        dataset = ImageDataset(name, cache_dir=str(cache_dir))
        init_time = time.perf_counter() - start

        print(f"\n[INIT] Dataset initialization: {init_time:.3f}s")
        print(f"[INIT] Total samples: {len(dataset)}")

        assert init_time < 5.0, f"Dataset init too slow: {init_time:.2f}s"

    def test_single_sample_loading(self, large_dataset_path):
        """Test single sample loading speed."""
        cache_dir, name, _, _ = large_dataset_path
        dataset = ImageDataset(name, cache_dir=str(cache_dir))

        # Warm up
        _ = dataset[0]

        # Benchmark single sample loading
        num_samples = 100
        start = time.perf_counter()
        for i in range(num_samples):
            idx = np.random.randint(0, len(dataset))
            _ = dataset[idx]
        elapsed = time.perf_counter() - start

        samples_per_sec = num_samples / elapsed
        ms_per_sample = (elapsed / num_samples) * 1000

        print(f"\n[SINGLE] {num_samples} random samples in {elapsed:.3f}s")
        print(f"[SINGLE] {samples_per_sec:.1f} samples/sec")
        print(f"[SINGLE] {ms_per_sample:.2f} ms/sample")

        assert samples_per_sec > 50, f"Single sample too slow: {samples_per_sec:.1f} samples/sec"

    def test_sequential_loading(self, large_dataset_path):
        """Test sequential loading speed."""
        cache_dir, name, _, _ = large_dataset_path
        dataset = ImageDataset(name, cache_dir=str(cache_dir))

        num_samples = 500
        start = time.perf_counter()
        for i in range(min(num_samples, len(dataset))):
            _ = dataset[i]
        elapsed = time.perf_counter() - start

        samples_per_sec = num_samples / elapsed

        print(f"\n[SEQUENTIAL] {num_samples} sequential samples in {elapsed:.3f}s")
        print(f"[SEQUENTIAL] {samples_per_sec:.1f} samples/sec")

        assert samples_per_sec > 100, f"Sequential loading too slow: {samples_per_sec:.1f} samples/sec"

    def test_dataloader_throughput(self, large_dataset_path):
        """Test DataLoader throughput with multiple workers."""
        cache_dir, name, _, _ = large_dataset_path
        dataset = ImageDataset(name, cache_dir=str(cache_dir))

        batch_size = 32
        num_batches = 20

        # Test with 0 workers (main process)
        loader_0 = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        start = time.perf_counter()
        for i, batch in enumerate(loader_0):
            if i >= num_batches:
                break
        elapsed_0 = time.perf_counter() - start
        throughput_0 = (num_batches * batch_size) / elapsed_0

        print(f"\n[DATALOADER num_workers=0] {num_batches} batches in {elapsed_0:.3f}s")
        print(f"[DATALOADER num_workers=0] {throughput_0:.1f} samples/sec")

        # Test with 4 workers
        loader_4 = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        start = time.perf_counter()
        for i, batch in enumerate(loader_4):
            if i >= num_batches:
                break
        elapsed_4 = time.perf_counter() - start
        throughput_4 = (num_batches * batch_size) / elapsed_4

        print(f"[DATALOADER num_workers=4] {num_batches} batches in {elapsed_4:.3f}s")
        print(f"[DATALOADER num_workers=4] {throughput_4:.1f} samples/sec")
        print(f"[DATALOADER] Speedup with 4 workers: {throughput_4/throughput_0:.2f}x")

        assert throughput_0 > 100, f"DataLoader (0 workers) too slow: {throughput_0:.1f} samples/sec"

    def test_load_chunk_performance(self, large_dataset_path):
        """Test load_chunk performance for batch loading."""
        cache_dir, name, num_episodes, _ = large_dataset_path
        dataset = ImageDataset(name, cache_dir=str(cache_dir))

        # Load chunks from multiple episodes
        num_chunks = 10
        chunk_size = 16

        episodes_idx = np.random.randint(0, num_episodes, size=num_chunks)
        starts = np.zeros(num_chunks, dtype=int)
        ends = np.full(num_chunks, chunk_size, dtype=int)

        start = time.perf_counter()
        chunks = dataset.load_chunk(episodes_idx, starts, ends)
        elapsed = time.perf_counter() - start

        total_frames = num_chunks * chunk_size
        frames_per_sec = total_frames / elapsed

        print(f"\n[LOAD_CHUNK] {num_chunks} chunks × {chunk_size} frames = {total_frames} frames")
        print(f"[LOAD_CHUNK] Loaded in {elapsed:.3f}s")
        print(f"[LOAD_CHUNK] {frames_per_sec:.1f} frames/sec")

        assert frames_per_sec > 50, f"load_chunk too slow: {frames_per_sec:.1f} frames/sec"

    def test_memory_efficiency(self, large_dataset_path):
        """Test that dataset doesn't load everything into memory."""
        import tracemalloc

        cache_dir, name, _, _ = large_dataset_path

        tracemalloc.start()
        dataset = ImageDataset(name, cache_dir=str(cache_dir))

        # Access a few samples
        for i in range(10):
            _ = dataset[i]

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / 1024 / 1024

        print(f"\n[MEMORY] Peak memory usage: {peak_mb:.1f} MB")
        print(f"[MEMORY] Dataset length: {len(dataset)}")

        # Should not load all images into memory
        # 5000 images × 64×64×3 bytes = ~60 MB if loaded
        # We should use much less since we load on demand
        assert peak_mb < 100, f"Memory usage too high: {peak_mb:.1f} MB"


class TestHDF5DatasetStress:
    """Stress tests for HDF5Dataset comparison."""

    @pytest.fixture(scope="class")
    def large_hdf5_path(self, tmp_path_factory):
        """Create a large HDF5 dataset for comparison."""
        import h5py

        path = tmp_path_factory.mktemp("stress_hdf5")
        h5_path = path / "large_hdf5.h5"

        num_episodes = 50
        steps_per_episode = 100
        total_steps = num_episodes * steps_per_episode

        ep_lengths = np.array([steps_per_episode] * num_episodes)
        ep_offsets = np.array([i * steps_per_episode for i in range(num_episodes)])

        print(f"\nCreating HDF5 with {total_steps} steps...")

        with h5py.File(h5_path, "w") as f:
            f.create_dataset("ep_len", data=ep_lengths)
            f.create_dataset("ep_offset", data=ep_offsets)
            f.create_dataset("action", data=np.random.randn(total_steps, 2).astype(np.float32))
            f.create_dataset("observation", data=np.random.randn(total_steps, 10).astype(np.float32))
            f.create_dataset("pixels", data=np.random.randint(0, 255, (total_steps, 64, 64, 3), dtype=np.uint8))

        return path, "large_hdf5"

    def test_hdf5_vs_image_comparison(self, large_hdf5_path):
        """Compare HDF5 vs ImageDataset loading speed."""
        cache_dir, name = large_hdf5_path

        dataset = HDF5Dataset(name, cache_dir=str(cache_dir))

        # Warm up
        _ = dataset[0]

        num_samples = 100
        start = time.perf_counter()
        for i in range(num_samples):
            idx = np.random.randint(0, len(dataset))
            _ = dataset[idx]
        elapsed = time.perf_counter() - start

        samples_per_sec = num_samples / elapsed

        print(f"\n[HDF5] {num_samples} random samples in {elapsed:.3f}s")
        print(f"[HDF5] {samples_per_sec:.1f} samples/sec")


if __name__ == "__main__":
    # Run standalone for quick benchmarking
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "benchmark_dataset"

        print("=" * 60)
        print("ImageDataset Stress Test")
        print("=" * 60)

        # Create dataset
        total = create_large_image_dataset(path, num_episodes=20, steps_per_episode=50)

        # Load and benchmark
        dataset = ImageDataset("benchmark_dataset", cache_dir=tmpdir)
        print(f"Dataset length: {len(dataset)}")

        # Single sample
        num_samples = 100
        start = time.perf_counter()
        for i in range(num_samples):
            _ = dataset[np.random.randint(0, len(dataset))]
        elapsed = time.perf_counter() - start
        print(f"Random access: {num_samples/elapsed:.1f} samples/sec")

        # DataLoader
        loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
        start = time.perf_counter()
        for i, batch in enumerate(loader):
            if i >= 10:
                break
        elapsed = time.perf_counter() - start
        print(f"DataLoader (4 workers): {(10*32)/elapsed:.1f} samples/sec")
