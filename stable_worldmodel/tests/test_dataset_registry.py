"""Tests for the financial dataset registry.

All tests use synthetic/dummy data and do not require any external APIs
or credentials (including Alpaca API). These tests demonstrate that users
can register their own custom data sources without needing any specific
data provider configured.
"""

import gymnasium as gym
import numpy as np
import pandas as pd
import pytest

from stable_worldmodel.envs import (
    get_registered_dataset,
    list_registered_datasets,
    register_financial_dataset,
    unregister_financial_dataset,
)


def test_register_and_get_dataset():
    """Test registering and retrieving a custom dataset loader."""

    def custom_loader(symbol, start_date, end_date, **kwargs):
        # Create synthetic data
        timestamps = pd.date_range(start=start_date, end=end_date, freq="1min")
        return pd.DataFrame(
            {
                "open": np.ones(len(timestamps)) * 100,
                "high": np.ones(len(timestamps)) * 101,
                "low": np.ones(len(timestamps)) * 99,
                "close": np.ones(len(timestamps)) * 100,
                "volume": np.ones(len(timestamps)) * 1000,
            },
            index=timestamps,
        )

    # Register the loader
    register_financial_dataset(custom_loader, name="test_loader")

    # Verify it's registered
    assert "test_loader" in list_registered_datasets()

    # Retrieve it
    loader = get_registered_dataset("test_loader")
    assert loader is not None
    assert callable(loader)

    # Test that it works
    df = loader("AAPL", "2024-01-01", "2024-01-02")
    assert isinstance(df, pd.DataFrame)
    assert "open" in df.columns
    assert "close" in df.columns

    # Clean up
    unregister_financial_dataset("test_loader")
    assert "test_loader" not in list_registered_datasets()


def test_register_dataset_validation():
    """Test that registration validates inputs correctly."""

    # Test with non-callable
    with pytest.raises(TypeError):
        register_financial_dataset("not_a_function")

    # Test with invalid name
    with pytest.raises(ValueError):
        register_financial_dataset(lambda: None, name="")


def test_environment_uses_custom_dataset():
    """Test that the environment uses a registered custom dataset."""

    def custom_loader(symbol, start_date, end_date, **kwargs):
        # Create 2 days of minute data (2 * 6.5 hours * 60 minutes = 780 minutes)
        # Add some buffer to ensure we have enough data
        timestamps = pd.date_range(start=start_date, periods=1000, freq="1min")

        # Create random walk data
        returns = np.random.normal(0, 0.001, size=len(timestamps))
        price = 100 * (1 + returns).cumprod()

        return pd.DataFrame(
            {
                "open": price * (1 + np.random.uniform(-0.001, 0.001, len(timestamps))),
                "high": price * (1 + np.random.uniform(0, 0.002, len(timestamps))),
                "low": price * (1 + np.random.uniform(-0.002, 0, len(timestamps))),
                "close": price,
                "volume": np.random.randint(1000, 10000, len(timestamps)),
            },
            index=timestamps,
        )

    # Register the custom loader
    register_financial_dataset(custom_loader, name="default")

    try:
        # Create environment - should use our custom loader
        env = gym.make(
            "swm/Financial-v0",
            start_date="2024-01-01",
            end_date="2024-01-03",
            max_steps=100,
        )
        obs, info = env.reset()

        # Verify environment works with custom data
        assert obs is not None
        assert isinstance(obs, np.ndarray)

        # Take a step to verify the environment functions properly
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs is not None
        assert isinstance(reward, int | float)

        env.close()

    finally:
        # Clean up
        unregister_financial_dataset("default")


def test_environment_requires_registered_dataset():
    """Test that the environment requires a registered dataset.

    This test verifies that without a registered dataset, the environment
    will raise a clear error message telling the user to register one.
    """
    # Ensure no custom dataset is registered
    unregister_financial_dataset("default")

    # Environment should raise an error without a registered dataset
    env = gym.make("swm/Financial-v0")

    with pytest.raises(RuntimeError, match="Data load failed"):
        env.reset()

    env.close()


def test_environment_works_with_registered_dataset():
    """Test that the environment works when a dataset is properly registered.

    This demonstrates the correct usage pattern - register a data source
    before using the financial environment.
    """

    # Register a simple synthetic loader
    def simple_loader(symbol, start_date, end_date, **kwargs):
        timestamps = pd.date_range(start=start_date, periods=500, freq="1min")
        return pd.DataFrame(
            {
                "open": np.ones(500) * 100,
                "high": np.ones(500) * 101,
                "low": np.ones(500) * 99,
                "close": np.ones(500) * 100,
                "volume": np.ones(500) * 1000,
            },
            index=timestamps,
        )

    register_financial_dataset(simple_loader, name="default")

    try:
        # Environment should work with registered dataset
        env = gym.make("swm/Financial-v0")
        assert env is not None

        # Verify it can actually be used
        obs, info = env.reset()
        assert obs is not None

        env.close()
    finally:
        # Clean up
        unregister_financial_dataset("default")


def test_multiple_named_datasets():
    """Test registering multiple datasets with different names."""

    def loader1(symbol, start_date, end_date, **kwargs):
        timestamps = pd.date_range(start=start_date, periods=100, freq="1min")
        return pd.DataFrame(
            {
                "open": np.ones(100) * 100,
                "high": np.ones(100) * 101,
                "low": np.ones(100) * 99,
                "close": np.ones(100) * 100,
            },
            index=timestamps,
        )

    def loader2(symbol, start_date, end_date, **kwargs):
        timestamps = pd.date_range(start=start_date, periods=100, freq="1min")
        return pd.DataFrame(
            {
                "open": np.ones(100) * 200,
                "high": np.ones(100) * 201,
                "low": np.ones(100) * 199,
                "close": np.ones(100) * 200,
            },
            index=timestamps,
        )

    # Register both loaders
    register_financial_dataset(loader1, name="loader1")
    register_financial_dataset(loader2, name="loader2")

    # Verify both are registered
    datasets = list_registered_datasets()
    assert "loader1" in datasets
    assert "loader2" in datasets

    # Retrieve and test both
    l1 = get_registered_dataset("loader1")
    l2 = get_registered_dataset("loader2")

    df1 = l1("AAPL", "2024-01-01", "2024-01-02")
    df2 = l2("AAPL", "2024-01-01", "2024-01-02")

    # Verify they return different data
    assert df1["close"].iloc[0] == 100
    assert df2["close"].iloc[0] == 200

    # Clean up
    unregister_financial_dataset("loader1")
    unregister_financial_dataset("loader2")


def test_unregister_nonexistent_dataset():
    """Test that unregistering a non-existent dataset doesn't raise an error."""
    # Should not raise any exception
    unregister_financial_dataset("nonexistent")


def test_get_nonexistent_dataset():
    """Test that getting a non-existent dataset returns None."""
    loader = get_registered_dataset("nonexistent")
    assert loader is None


def test_registry_works_without_external_dependencies():
    """Test that the registry system works completely independently of external APIs.

    This demonstrates that users can use the FinancialEnvironment with their own
    data sources without needing Alpaca API, AWS, databases, or any other external
    service configured.
    """

    # Create a simple in-memory data loader (could be from CSV, database, etc.)
    def my_custom_loader(symbol, start_date, end_date, **kwargs):
        """A dummy loader that could represent any user data source."""
        # User's data could come from anywhere: CSV, Parquet, SQL, NoSQL, etc.
        timestamps = pd.date_range(start=start_date, periods=200, freq="1min")
        prices = 100 + np.cumsum(np.random.randn(200) * 0.5)  # Random walk

        return pd.DataFrame(
            {
                "open": prices,
                "high": prices + abs(np.random.randn(200) * 0.2),
                "low": prices - abs(np.random.randn(200) * 0.2),
                "close": prices,
                "volume": np.random.randint(1000, 5000, 200),
            },
            index=timestamps,
        )

    # Register custom data
    register_financial_dataset(my_custom_loader, name="default")

    try:
        # Create and use environment with custom data (no external APIs needed)
        env = gym.make("swm/Financial-v0")
        obs, info = env.reset()

        # Environment works perfectly with user's own data
        assert obs is not None
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(reward, int | float)

        env.close()
    finally:
        unregister_financial_dataset("default")
