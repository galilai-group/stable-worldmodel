"""Dataset registry for financial trading environment.

This module provides a registration mechanism that allows external users to
supply their own data sources for the FinancialEnvironment without modifying
the library code.

Example:
    >>> from stable_worldmodel.envs import register_financial_dataset
    >>>
    >>> def my_custom_data_loader(symbol, start_date, end_date, **kwargs):
    ...     # Load from your own data source (database, files, API, etc.)
    ...     # Return pandas DataFrame with columns: open, high, low, close, volume
    ...     # Index should be datetime timestamps
    ...     import pandas as pd
    ...     data = load_from_my_source(symbol, start_date, end_date)
    ...     return data
    ...
    >>> register_financial_dataset(my_custom_data_loader)
"""

from collections.abc import Callable
from typing import Any

import pandas as pd


# Type for the dataset loader function
DatasetLoader = Callable[[str, str, str, Any], pd.DataFrame]

# Global registry - stores the custom get_item function
_DATASET_REGISTRY: dict[str, DatasetLoader] = {}


def register_financial_dataset(
    get_item_func: DatasetLoader,
    name: str = "default",
) -> None:
    """Register a custom dataset loader for the financial trading environment.

    This function allows external users to provide their own data source without
    modifying the library code. The registered function will be called by the
    FinancialEnvironment to load historical market data.

    Args:
        get_item_func: A callable that loads financial data. Must have signature:
            get_item_func(symbol: str, start_date: str, end_date: str, **kwargs) -> pd.DataFrame

            The function should return a pandas DataFrame with:
            - Index: DatetimeIndex with timestamps
            - Required columns: 'open', 'high', 'low', 'close'
            - Optional columns: 'volume', 'trade_count', 'vwap'
            - All price/volume columns should be numeric (float)

        name: Name for this dataset loader. Use "default" to replace the default loader.
            Default is "default".

    Raises:
        TypeError: If get_item_func is not callable.
        ValueError: If name is empty or invalid.

    Example:
        >>> def load_from_csv(symbol, start_date, end_date, **kwargs):
        ...     import pandas as pd
        ...     df = pd.read_csv(f"data/{symbol}.csv", parse_dates=['timestamp'])
        ...     df = df.set_index('timestamp')
        ...     mask = (df.index >= start_date) & (df.index <= end_date)
        ...     return df[mask]
        ...
        >>> register_financial_dataset(load_from_csv)
    """
    if not callable(get_item_func):
        raise TypeError(f"get_item_func must be callable, got {type(get_item_func).__name__}")

    if not name or not isinstance(name, str):
        raise ValueError(f"name must be a non-empty string, got {name!r}")

    _DATASET_REGISTRY[name] = get_item_func


def get_registered_dataset(name: str = "default") -> DatasetLoader | None:
    """Get a registered dataset loader by name.

    Args:
        name: Name of the registered dataset loader. Default is "default".

    Returns:
        The registered dataset loader function, or None if not found.
    """
    return _DATASET_REGISTRY.get(name)


def unregister_financial_dataset(name: str = "default") -> None:
    """Unregister a dataset loader.

    Args:
        name: Name of the dataset loader to unregister. Default is "default".
    """
    _DATASET_REGISTRY.pop(name, None)


def list_registered_datasets() -> list[str]:
    """List all registered dataset loader names.

    Returns:
        List of registered dataset names.
    """
    return list(_DATASET_REGISTRY.keys())


def _get_default_alpaca_loader() -> DatasetLoader:
    """Get the default Alpaca data loader.

    This is used as a fallback when no custom dataset is registered.

    Returns:
        A dataset loader function that uses Alpaca data.
    """
    from stable_worldmodel.finance_data.download import load_market_data

    def alpaca_loader(
        symbol: str,
        start_date: str,
        end_date: str,
        hdf_path: str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Load data from Alpaca API."""
        # Load market data using the download module
        data_array = load_market_data(
            hdf_path=hdf_path,
            tickers=[symbol],
            start_time=start_date,
            end_time=end_date,
            freq="1min",
        )

        # Convert numpy array to DataFrame
        # data_array shape: (T, 1, 7) where features are
        # [open, high, low, close, volume, trade_count, vwap]
        features = ["open", "high", "low", "close", "volume", "trade_count", "vwap"]

        # Squeeze out the stock dimension since we only have one stock
        df_data = data_array[:, 0, :]  # Shape: (T, 7)

        # Create DataFrame with timestamp index
        start_ts = pd.Timestamp(start_date)
        timestamps = pd.date_range(start=start_ts, periods=len(df_data), freq="1min")

        df = pd.DataFrame(df_data, columns=features, index=timestamps)
        df = df.dropna()

        return df

    return alpaca_loader
