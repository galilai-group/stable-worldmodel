"""Pytest configuration and shared fixtures."""

import os
import pytest
import torch


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests (fast, no GPU required)")
    config.addinivalue_line(
        "markers", "integration: Integration tests (slow, may require GPU)"
    )
    config.addinivalue_line("markers", "gpu: Tests that require GPU")
    config.addinivalue_line("markers", "slow: Tests that take a long time to run")
    config.addinivalue_line(
        "markers", "download: Tests that download data from the internet"
    )


@pytest.fixture
def device():
    """Fixture to get appropriate device for tests."""
    if torch.cuda.is_available() and not os.environ.get("FORCE_CPU"):
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture
def mock_batch(device):
    """Create a mock batch of data."""
    batch_size = 4
    return {
        "image": torch.randn(batch_size, 3, 224, 224, device=device),
        "label": torch.randint(0, 10, (batch_size,), device=device),
        "index": torch.arange(batch_size, device=device),
    }


@pytest.fixture(scope="session")
def temp_dir(tmp_path_factory):
    """Create a temporary directory for test outputs."""
    return tmp_path_factory.mktemp("test_outputs")
