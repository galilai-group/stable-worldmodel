from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from loguru import logger as logging
from scipy import stats as scipy_stats

import stable_worldmodel as swm
from stable_worldmodel.envs.dataset_registry import get_registered_dataset


def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252 * 390,
) -> float:
    """Calculate Sharpe ratio (risk-adjusted return)."""
    if len(returns) < 2:
        return 0.0

    risk_free_period = risk_free_rate / periods_per_year
    excess_returns = returns - risk_free_period

    std = np.std(excess_returns, ddof=1)
    if std == 0:
        return 0.0

    return np.mean(excess_returns) / std * np.sqrt(periods_per_year)


def calculate_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252 * 390,
) -> float:
    """Calculate Sortino ratio (downside risk-adjusted return)."""
    if len(returns) < 2:
        return 0.0

    risk_free_period = risk_free_rate / periods_per_year
    excess_returns = returns - risk_free_period
    downside_returns = returns[returns < 0]

    if len(downside_returns) == 0:
        return 0.0

    downside_std = np.std(downside_returns, ddof=1)
    if downside_std == 0:
        return 0.0

    downside_volatility = downside_std * np.sqrt(periods_per_year)
    return np.mean(excess_returns) * periods_per_year / downside_volatility


def calculate_max_drawdown(portfolio_values: Sequence[float]) -> float:
    """Calculate maximum drawdown from peak to trough."""  # codespell:ignore
    if len(portfolio_values) < 2:
        return 0.0

    values = np.array(portfolio_values)
    cumulative_max = np.maximum.accumulate(values)
    drawdowns = (values - cumulative_max) / cumulative_max
    return abs(np.min(drawdowns))


def calculate_calmar_ratio(annualized_return: float, max_drawdown: float) -> float:
    """Calculate Calmar ratio (return / max drawdown)."""
    if max_drawdown == 0:
        return 0.0
    return annualized_return / max_drawdown


def calculate_alpha_beta(
    returns: np.ndarray,
    benchmark_returns: np.ndarray,
    annualized_return: float,
    benchmark_annualized: float,
    risk_free_rate: float = 0.02,
) -> tuple[float, float]:
    """Calculate alpha and beta relative to benchmark."""
    if len(returns) < 2 or len(benchmark_returns) < 2 or len(returns) != len(benchmark_returns):
        return 0.0, 0.0

    cov_matrix = np.cov(returns, benchmark_returns)
    if cov_matrix[1, 1] == 0:
        return 0.0, 0.0

    beta = cov_matrix[0, 1] / cov_matrix[1, 1]
    alpha = annualized_return - (risk_free_rate + beta * (benchmark_annualized - risk_free_rate))

    return alpha, beta


def calculate_volatility(returns: np.ndarray, periods_per_year: int = 252 * 390) -> float:
    """Calculate annualized volatility."""
    if len(returns) < 2:
        return 0.0

    return np.std(returns, ddof=1) * np.sqrt(periods_per_year)


def calculate_downside_volatility(returns: np.ndarray, periods_per_year: int = 252 * 390) -> float:
    """Calculate downside volatility."""
    downside_returns = returns[returns < 0]
    if len(downside_returns) < 2:
        return 0.0

    return np.std(downside_returns, ddof=1) * np.sqrt(periods_per_year)


def calculate_information_ratio(
    returns: np.ndarray,
    benchmark_returns: np.ndarray,
    annualized_return: float,
    benchmark_annualized: float,
    periods_per_year: int = 252 * 390,
) -> tuple[float, float]:
    """Calculate information ratio and tracking error."""
    if len(returns) != len(benchmark_returns) or len(returns) < 2:
        return 0.0, 0.0

    active_returns = returns - benchmark_returns
    tracking_error = np.std(active_returns, ddof=1) * np.sqrt(periods_per_year)

    if tracking_error == 0:
        return 0.0, 0.0

    information_ratio = (annualized_return - benchmark_annualized) / tracking_error
    return information_ratio, tracking_error


def calculate_var_cvar(returns: np.ndarray, confidence_level: float = 0.95) -> tuple[float, float]:
    """Calculate Value at Risk (VaR) and Conditional VaR (CVaR)."""
    if len(returns) < 10:
        return 0.0, 0.0

    percentile = (1 - confidence_level) * 100
    var = np.percentile(returns, percentile)
    cvar = np.mean(returns[returns <= var])

    return var, cvar


def calculate_skewness_kurtosis(returns: np.ndarray) -> tuple[float, float]:
    """Calculate skewness and kurtosis of return distribution."""
    if len(returns) < 4:
        return 0.0, 0.0

    skewness = scipy_stats.skew(returns)
    kurtosis = scipy_stats.kurtosis(returns)

    return skewness, kurtosis


def calculate_tail_ratio(returns: np.ndarray) -> float:
    """Calculate tail ratio (95th percentile / 5th percentile)."""
    if len(returns) < 10:
        return 0.0

    p95 = np.percentile(returns, 95)
    p5 = np.percentile(returns, 5)

    if p5 == 0:
        return 0.0

    return abs(p95 / p5)


def calculate_omega_ratio(returns: np.ndarray, threshold: float = 0.0) -> float:
    """Calculate Omega ratio (probability weighted gains vs losses)."""
    if len(returns) < 2:
        return 0.0

    gains = returns[returns > threshold] - threshold
    losses = threshold - returns[returns < threshold]

    if np.sum(losses) == 0:
        return 0.0

    return np.sum(gains) / np.sum(losses)


def calculate_win_rate(trade_returns: Sequence[float]) -> float:
    """Calculate win rate (proportion of winning trades)."""
    if len(trade_returns) == 0:
        return 0.0

    winning_trades = sum(1 for r in trade_returns if r > 0)
    return winning_trades / len(trade_returns)


def calculate_profit_factor(trade_returns: Sequence[float]) -> float:
    """Calculate profit factor (total gains / total losses)."""
    if len(trade_returns) == 0:
        return 0.0

    total_gains = sum(r for r in trade_returns if r > 0)
    total_losses = abs(sum(r for r in trade_returns if r < 0))

    if total_losses == 0:
        return 0.0

    return total_gains / total_losses


def calculate_annualized_return(total_return: float, total_periods: int, periods_per_year: int = 252 * 390) -> float:
    """Calculate annualized return from total return."""
    if total_periods == 0:
        return 0.0

    years = total_periods / periods_per_year
    if years < 1 / periods_per_year:
        years = 1 / periods_per_year

    return (1 + total_return) ** (1 / years) - 1


def calculate_stability(returns: np.ndarray) -> float:
    """Calculate stability of returns (R-squared)."""
    if len(returns) < 3:
        return 0.0

    cumulative_returns = pd.Series(returns).add(1).cumprod().sub(1)
    log_cum_returns = np.log1p(cumulative_returns)
    time_index = np.arange(len(returns))

    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(time_index, log_cum_returns)

    return r_value**2


DEFAULT_VARIATIONS = ("agent.starting_balance",)


class FinancialEnvironment(gym.Env):
    """Financial backtesting environment with integrated data pipeline.

    Automatically loads minute-level OHLCV data from Alpaca API with smart caching
    and multiple compression formats. Supports time machine backtesting (reset to
    any historical date) and tracks comprehensive performance metrics.

    Action Space: 0=SELL, 1=BUY, 2=HOLD
    Observation: Historical OHLCV + portfolio state + time features
    """

    metadata = {"render_modes": []}
    reward_range = (-np.inf, np.inf)

    def __init__(
        self,
        data_dir: str | Path | None = None,
        max_steps: int = 1440,
        symbols: list | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        render_mode: str | None = None,
    ):
        """Initialize financial backtesting environment with data pipeline."""
        super().__init__()
        self.render_mode = render_mode

        self.data_dir = Path(data_dir) if data_dir else swm.data.utils.get_cache_dir()
        self.max_steps = max_steps

        self.default_symbols = symbols or self.get_available_symbols()[:5]
        self.default_start_date = start_date
        self.default_end_date = end_date

        self.action_space = gym.spaces.Discrete(3)

        self.variation_space = swm.spaces.Dict(
            {
                "agent": swm.spaces.Dict(
                    {
                        "starting_balance": swm.spaces.Box(
                            low=10000.0,
                            high=1000000.0,
                            init_value=np.array(100000.0, dtype=np.float32),
                            shape=(),
                            dtype=np.float32,
                        ),
                        "transaction_cost": swm.spaces.Box(
                            low=0.0,
                            high=0.01,
                            init_value=np.array(0.001, dtype=np.float32),
                            shape=(),
                            dtype=np.float32,
                        ),
                        "market_impact": swm.spaces.Box(
                            low=0.0,
                            high=0.001,
                            init_value=np.array(0.0001, dtype=np.float32),
                            shape=(),
                            dtype=np.float32,
                        ),
                        "window_size": swm.spaces.Box(
                            low=10.0,
                            high=390.0,
                            init_value=np.array(60.0, dtype=np.float32),
                            shape=(),
                            dtype=np.float32,
                        ),
                        "enable_shorting": swm.spaces.Discrete(n=2, init_value=1),
                    }
                ),
            },
            sampling_order=["agent"],
        )

        default_window_size = 60
        default_obs_size = default_window_size * 6 + 7
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(default_obs_size,), dtype=np.float32)

        self.window_size = None
        self.enable_shorting = None
        self.transaction_cost = None
        self.market_impact = None

        # Initialize state variables
        self.current_step = 0
        self.current_symbol = "AAPL"
        self.position = 0.0  # Continuous position: negative=short, positive=long, 0=neutral
        self.balance = 100000.0  # Will be set by variation space
        self.portfolio_value = 100000.0
        self.shares_held = 0.0
        self.trade_history = []
        self.price_history = []

        # Data storage
        self.market_data = None
        self.current_data_index = 0
        self.data_timestamps = None

        # Market impact tracking
        self.price_impacts = {}  # {timestamp: impact_amount}
        self.impact_decay_rate = 0.95  # Per-step decay (5% decay per step)
        self.impact_half_life = 20  # Steps for impact to halve
        self.cumulative_impact = 0.0  # Current total impact on price (temporary + permanent)
        self.permanent_impact = 0.0  # Permanent price shift (doesn't decay)
        self.temporary_impact = 0.0  # Temporary impact (decays quickly)

        # Impact coefficients in basis points (bps)
        # 1 bps = 0.0001 = 0.01%
        self.impact_coeff = 0.0005  # 5 bps at 100% ADV participation (default: retail)
        self.permanent_ratio = 0.1  # 10% of impact is permanent, 90% temporary

        # Performance tracking
        self.start_time = None
        self.benchmark_returns = []
        self.portfolio_returns = []

        assert self.variation_space.check(), "Invalid default variation values"

    def _get_default_backtest_config(self) -> tuple[str, str, str]:
        """Get default date range and symbol for backtesting.

        Uses a 2-day window ending 3 days ago to ensure data availability
        with Alpaca's free tier (provides ~5-15 days of historical minute data).
        """
        end_date = pd.Timestamp.now() - pd.Timedelta(days=3)
        start_date = end_date - pd.Timedelta(days=2)  # 2-day window for free tier
        symbol = "AAPL"
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), symbol

    def get_available_symbols(self) -> list[str]:
        return [
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "NVDA",
            "META",
            "TSLA",
            "BRK.B",
            "UNH",
            "JNJ",
            "V",
            "XOM",
            "WMT",
            "JPM",
            "PG",
            "MA",
            "HD",
            "CVX",
            "LLY",
            "ABBV",
            "MRK",
            "AVGO",
            "COST",
            "KO",
            "PEP",
            "ADBE",
            "TMO",
            "MCD",
            "CSCO",
            "ACN",  # codespell:ignore
            "ABT",
            "DHR",
            "NKE",
            "TXN",
            "DIS",
            "VZ",
            "CRM",
            "WFC",
            "CMCSA",
            "NEE",
            "PM",
            "NFLX",
            "BMY",
            "UPS",
            "HON",
            "ORCL",
            "T",
            "RTX",
            "QCOM",
            "INTC",
        ]

    def get_date_range_info(self, symbol: str) -> dict[str, str]:
        """Get date range availability for symbol (auto-downloads from Alpaca if needed)."""
        return {
            "symbol": symbol,
            "earliest_date": "2015-01-01",
            "latest_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
            "frequency": "1min",
            "note": "Data auto-downloaded from Alpaca if not cached locally",
        }

    @staticmethod
    def get_data_format_info() -> dict[str, Any]:
        """Get data format information (bfloat16 only)."""
        return {
            "format_class": "numpyformat",
            "encoding": "bfloat16",
            "description": "NumPy compressed format with bfloat16 differential encoding",
        }

    def _load_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Load historical OHLCV data using registered dataset loader.

        Requires a custom dataset to be registered via register_financial_dataset().
        """
        try:
            # Try to get a registered dataset loader
            loader = get_registered_dataset("default")

            # If no loader is registered, raise an error
            if loader is None:
                raise ValueError(
                    "No financial dataset has been registered. Please register a dataset loader using:\n\n"
                    "    from stable_worldmodel.envs import register_financial_dataset\n"
                    "    register_financial_dataset(your_data_loader_function)\n\n"
                    "Your data loader should return a pandas DataFrame with OHLCV data.\n"
                )

            logging.info("Using registered dataset loader")

            # Set up HDF5 cache path (loaders can ignore this if not needed)
            hdf_path = str(self.data_dir / "dataset.h5")

            # Call the loader
            df = loader(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                hdf_path=hdf_path,
            )

            # Validate the returned data
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"Dataset loader must return pd.DataFrame, got {type(df).__name__}")

            # Remove NaN rows
            df = df.dropna()

            if df.empty:
                raise ValueError(f"No data for {symbol} from {start_date} to {end_date}")

            required_columns = ["open", "high", "low", "close"]
            if missing := set(required_columns) - set(df.columns):
                raise ValueError(f"Missing required columns: {missing}")

            df = df.sort_index()
            logging.info(f"Loaded {len(df)} bars for {symbol}")
            return df

        except Exception as e:
            logging.error(f"Failed to load data for {symbol}: {e}")
            raise RuntimeError(f"Data load failed for {symbol}") from e

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed, options=options)

        if hasattr(self, "variation_space"):
            self.variation_space.seed(seed)

        options = options or {}
        self.variation_space.reset()

        variations = options.get("variation", DEFAULT_VARIATIONS)
        if not isinstance(variations, Sequence):
            raise ValueError("variation option must be a Sequence containing variation names to sample")

        self.variation_space.update(variations)
        assert self.variation_space.check(debug=True), "Variation values must be within variation space!"

        self.window_size = int(self.variation_space["agent"]["window_size"].value)
        self.enable_shorting = bool(self.variation_space["agent"]["enable_shorting"].value)
        self.transaction_cost = float(self.variation_space["agent"]["transaction_cost"].value)
        self.market_impact = float(self.variation_space["agent"]["market_impact"].value)

        obs_size = self.window_size * 6 + 7
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

        # Use provided dates/symbols or fall back to defaults
        if self.default_start_date and self.default_end_date and self.default_symbols:
            start_date = self.default_start_date
            end_date = self.default_end_date
            self.current_symbol = self.default_symbols[0]
        else:
            start_date, end_date, self.current_symbol = self._get_default_backtest_config()

        print(f"Loading historical data for backtesting: {self.current_symbol} from {start_date} to {end_date}")
        self.market_data = self._load_historical_data(self.current_symbol, start_date, end_date)

        if len(self.market_data) < self.window_size:
            raise ValueError(f"Insufficient data: {len(self.market_data)} rows, need at least {self.window_size}")

        # Initialize episode state
        self.current_step = 0
        self.current_data_index = int(self.window_size)  # Start after window (ensure it's an integer)
        self.position = 0.0
        self.balance = 100000.0  # Default starting balance
        self.portfolio_value = self.balance
        self.shares_held = 0.0
        self.trade_history = []
        self.price_history = []

        # Reset market impact tracking
        self.price_impacts = {}
        self.cumulative_impact = 0.0
        self.temporary_impact = 0.0
        self.permanent_impact = 0.0

        # Performance tracking
        self.start_time = self.market_data.index[self.current_data_index]
        self.benchmark_returns = []
        self.portfolio_returns = []

        # Store initial portfolio value for return calculation
        self.initial_portfolio_value = self.portfolio_value

        # Create initial observation
        observation = self._get_observation()
        info = {
            "symbol": self.current_symbol,
            "timestamp": str(self.start_time),  # Convert to string for serialization
            "market_data_length": int(len(self.market_data)),
            # Flatten backtest_config for easier serialization
            "backtest_start_date": start_date,
            "backtest_end_date": end_date,
            "backtest_symbol": self.current_symbol,
            "goal": np.zeros((64, 64, 3), dtype=np.uint8),  # Dummy goal for wrapper compatibility
        }

        return observation, info

    # Symbol selection is now handled dynamically in _get_symbol_from_variation()
    # using real available symbols from Alpaca.get_assets()

    def step(self, action: int):
        if self.market_data is None:
            raise RuntimeError(
                "No market data available. Please integrate Alpaca data pipeline "
                "by implementing _load_historical_data() method."
            )

        if self.current_data_index >= len(self.market_data) - 1:
            # End of data
            terminated = True
            truncated = False
            reward = 0.0
            return (
                self._get_observation(),
                reward,
                terminated,
                truncated,
                {"reason": "end_of_data"},
            )

        # Get current market data
        current_row = self.market_data.iloc[self.current_data_index]
        base_price = current_row["close"]  # Historical market price
        current_price = self._get_adjusted_price(base_price)  # Price including our cumulative impact
        current_timestamp = self.market_data.index[self.current_data_index]

        # Apply time acceleration
        time_accel = 1.0  # Default 1x speed until variation space is connected
        self.current_data_index += int(time_accel)

        # Calculate transaction costs and market impact
        transaction_cost_bps = 10.0  # Default 10 basis points
        slippage_multiplier = 1.0  # Default no additional slippage

        # Execute trading action
        action_taken = "hold"

        if action == 1:  # BUY/LONG
            action_taken = self._execute_buy_order(current_price, transaction_cost_bps, slippage_multiplier)

        elif action == 0:  # SELL/SHORT
            action_taken = self._execute_sell_order(current_price, transaction_cost_bps, slippage_multiplier)

        # action == 2 is HOLD (no action needed)

        # Update portfolio value
        previous_portfolio_value = self.portfolio_value
        self.portfolio_value = self.balance + (self.shares_held * current_price)

        # Calculate returns
        portfolio_return = (self.portfolio_value - previous_portfolio_value) / previous_portfolio_value
        benchmark_return = self._calculate_benchmark_return()

        self.portfolio_returns.append(portfolio_return)
        self.benchmark_returns.append(benchmark_return)

        # Calculate reward (risk-adjusted return)
        reward = self._calculate_reward(portfolio_return, benchmark_return)

        # Store trade history
        trade_record = {
            "step": self.current_step,
            "timestamp": current_timestamp,
            "symbol": self.current_symbol,
            "price": current_price,
            "action": action_taken,
            "position": self.position,
            "balance": self.balance,
            "shares_held": self.shares_held,
            "portfolio_value": self.portfolio_value,
            "portfolio_return": portfolio_return,
            "benchmark_return": benchmark_return,
            "reward": reward,
        }
        self.trade_history.append(trade_record)
        self.price_history.append(
            {
                "timestamp": current_timestamp,
                "price": current_price,
                "volume": current_row["volume"],
            }
        )

        self.current_step += 1

        # Check termination conditions (market_data guaranteed to exist by check above)
        max_data_steps = len(self.market_data) - self.window_size - 1
        terminated = self.current_data_index >= max_data_steps or self.current_step >= self.max_steps

        # Check for bankruptcy or margin call
        min_balance = 10000.0  # 10% of default initial balance
        truncated = self.portfolio_value < min_balance

        if truncated:
            reward -= 100  # Heavy penalty for bankruptcy

        observation = self._get_observation()
        info = {
            "timestamp": str(current_timestamp),  # Convert timestamp to string for serialization
            "symbol": self.current_symbol,
            "portfolio_value": float(self.portfolio_value),
            "balance": float(self.balance),
            "position": float(self.position),
            "shares_held": float(self.shares_held),
            "action_taken": action_taken,
            "current_price": float(current_price),
            "base_price": float(base_price),  # Original historical price
            "cumulative_impact": float(self.cumulative_impact),  # Total price impact from our trades
            "temporary_impact": float(self.temporary_impact),  # Temporary (decaying) component
            "permanent_impact": float(self.permanent_impact),  # Permanent component
            "portfolio_return": float(portfolio_return),
            "benchmark_return": float(benchmark_return),
            "total_return": float(
                (self.portfolio_value - self.initial_portfolio_value) / max(self.initial_portfolio_value, 1e-8)
            ),
            "sharpe_ratio": float(self._calculate_sharpe_ratio()),
            "market_data_length": int(len(self.market_data)),
            # Flatten backtest_config for easier serialization
            "backtest_start_date": self.market_data.index[0].strftime("%Y-%m-%d"),
            "backtest_end_date": self.market_data.index[-1].strftime("%Y-%m-%d"),
            "backtest_symbol": self.current_symbol,
            "goal": np.zeros((64, 64, 3), dtype=np.uint8),  # Dummy goal for wrapper compatibility
        }

        return observation, reward, terminated, truncated, info

    def set_trader_profile(self, profile: str = "institutional") -> None:
        """Configure market impact parameters based on trader type.

        Parameters
        ----------
        profile : str
            Trader profile type:
            - "retail": Small retail trader (< 0.1% volume, minimal impact)
            - "institutional": Mid-size institutional trader (1-5% volume, moderate impact)
            - "large_fund": Large fund/whale (> 10% volume, significant impact)
            - "hft": High-frequency trader (rapid small trades, fast decay)
        """
        profiles = {
            "retail": {
                "impact_coeff": 0.0002,  # 2 bps at 100% ADV participation
                "permanent_ratio": 0.05,  # Only 5% permanent
                "impact_half_life": 10,  # Quick decay (retail doesn't move markets long)
                "description": "Small retail trader with minimal market impact (~2 bps)",
            },
            "institutional": {
                "impact_coeff": 0.001,  # 10 bps at 100% ADV participation
                "permanent_ratio": 0.1,  # 10% permanent
                "impact_half_life": 20,  # Medium decay
                "description": "Institutional trader with moderate market impact (~10 bps)",
            },
            "large_fund": {
                "impact_coeff": 0.003,  # 30 bps at 100% ADV participation
                "permanent_ratio": 0.2,  # 20% permanent (whale moves stick)
                "impact_half_life": 40,  # Slow decay (large moves persist)
                "description": "Large fund/whale with significant market impact (~30 bps)",
            },
            "hft": {
                "impact_coeff": 0.0001,  # 1 bps at 100% ADV participation
                "permanent_ratio": 0.01,  # Almost no permanent impact
                "impact_half_life": 5,  # Very fast decay
                "description": "High-frequency trader with minimal impact (~1 bps)",
            },
        }

        if profile not in profiles:
            raise ValueError(f"Unknown profile '{profile}'. Choose from: {list(profiles.keys())}")

        config = profiles[profile]
        self.impact_coeff = config["impact_coeff"]
        self.permanent_ratio = config["permanent_ratio"]
        self.impact_half_life = config["impact_half_life"]

        logging.info(f"Set trader profile: {profile}")
        logging.info(f"  {config['description']}")
        logging.info(f"  Impact coefficient: {self.impact_coeff:.6f} ({self.impact_coeff * 10000:.1f} bps)")
        logging.info(f"  Permanent ratio: {self.permanent_ratio:.1%}")
        logging.info(f"  Impact half-life: {self.impact_half_life} steps")

    def _calculate_market_impact(self, trade_size: float, current_price: float, current_volume: float) -> float:
        """Calculate price impact from a trade based on trade size relative to volume.

        Uses square root market impact model: impact ∝ sqrt(trade_size / volume)
        This is a standard model in quantitative finance (see Almgren & Chriss, 2000)

        Parameters
        ----------
        trade_size : float
            Absolute value of shares being traded
        current_price : float
            Current market price
        current_volume : float
            Current market volume

        Returns
        -------
        float
            Price impact amount (positive = price increase)
        """
        if current_volume <= 0 or trade_size <= 0:
            return 0.0

        # Volume participation rate (what % of current volume are we trading?)
        participation_rate = trade_size / current_volume

        # Almgren-Chriss square-root impact model:
        # impact ∝ sqrt(trade_size / volume)
        # This prevents impact from exploding with large trades
        #
        # Coefficient is in basis points at 100% participation:
        # - 0.0001 = 1 bps (0.01%)
        # - 0.001 = 10 bps (0.1%)
        # - 0.01 = 100 bps (1%)
        relative_impact = self.impact_coeff * np.sqrt(participation_rate)

        # Convert to absolute price impact
        price_impact = current_price * relative_impact

        return price_impact

    def _update_cumulative_impact(self, new_impact: float, timestamp: Any) -> None:
        """Update cumulative price impact with decay.

        Splits impact into temporary (decays fast) and permanent (persists) components.
        This models:
        - Temporary: Bid-ask bounce, liquidity provision, mean reversion
        - Permanent: Information content, inventory effects, informed trading

        Parameters
        ----------
        new_impact : float
            New price impact to add (positive = upward pressure)
        timestamp : datetime-like
            Timestamp of the new impact
        """
        # Split into temporary and permanent components
        permanent_component = new_impact * self.permanent_ratio
        temporary_component = new_impact * (1 - self.permanent_ratio)

        # Decay existing temporary impact (exponential decay)
        decay_factor = np.exp(-np.log(2) / self.impact_half_life)
        self.temporary_impact *= decay_factor

        # Add new impacts
        self.temporary_impact += temporary_component
        self.permanent_impact += permanent_component

        # Total cumulative impact
        self.cumulative_impact = self.temporary_impact + self.permanent_impact

        # Store individual impact for tracking
        if timestamp not in self.price_impacts:
            self.price_impacts[timestamp] = 0.0
        self.price_impacts[timestamp] += new_impact

    def _get_adjusted_price(self, base_price: float) -> float:
        """Get market price adjusted for accumulated trading impact.

        Parameters
        ----------
        base_price : float
            Original historical market price

        Returns
        -------
        float
            Price adjusted for cumulative market impact from agent's trades
        """
        return base_price + self.cumulative_impact

    def _execute_buy_order(self, price: float, transaction_cost_bps: float, slippage_multiplier: float) -> str:
        """Execute a buy order with realistic transaction costs and market impact."""
        if self.balance <= 0:
            return "hold"  # No money to buy

        # If we have a short position, close it first
        if self.shares_held < 0:
            return self._close_short_position(price, transaction_cost_bps, slippage_multiplier)

        # Get current volume for impact calculation
        current_row = self.market_data.iloc[self.current_data_index]
        current_volume = current_row.get("volume", 1000000.0)

        # Calculate shares to buy (use only a fraction of balance to prevent extreme leverage)
        max_position_size = 0.95  # Use 95% of balance to leave buffer
        transaction_cost = transaction_cost_bps / 10000.0
        slippage = price * 0.0001 * slippage_multiplier

        effective_price = price + slippage
        available_capital = self.balance * max_position_size
        shares_to_buy = available_capital / (effective_price * (1 + transaction_cost))

        if shares_to_buy > 0:
            # Calculate market impact from this trade
            trade_impact = self._calculate_market_impact(shares_to_buy, price, current_volume)

            # Impact pushes price UP for buy orders
            effective_price_with_impact = effective_price + trade_impact

            total_cost = shares_to_buy * effective_price_with_impact * (1 + transaction_cost)

            if total_cost <= self.balance:
                self.shares_held += shares_to_buy
                self.balance -= total_cost
                self.position = self.shares_held  # Position tracks shares_held
                return "buy"

        return "hold"

    def _execute_sell_order(self, price: float, transaction_cost_bps: float, slippage_multiplier: float) -> str:
        """Execute a sell order with realistic transaction costs and market impact."""
        if self.shares_held <= 0 and not self.enable_shorting:
            return "hold"  # No shares to sell and shorting disabled

        # Get current volume for impact calculation
        current_row = self.market_data.iloc[self.current_data_index]
        current_volume = current_row.get("volume", 1000000.0)

        transaction_cost = transaction_cost_bps / 10000.0
        slippage = price * 0.0001 * slippage_multiplier
        effective_price = price - slippage

        if self.shares_held > 0:
            # Calculate market impact from selling
            trade_impact = self._calculate_market_impact(self.shares_held, price, current_volume)

            # Impact pushes price DOWN for sell orders (negative impact)
            effective_price_with_impact = effective_price - trade_impact

            # Sell existing long position
            revenue = self.shares_held * effective_price_with_impact * (1 - transaction_cost)
            self.balance += revenue
            self.shares_held = 0.0
            self.position = 0.0

            # Update cumulative market impact (negative for sells)
            timestamp = self.market_data.index[self.current_data_index]
            self._update_cumulative_impact(-trade_impact, timestamp)

            return "sell"

        elif self.enable_shorting and self.shares_held == 0:
            # Enter short position
            # Use only a fraction of balance for shorting to prevent extreme leverage
            max_short_value = self.balance * 0.5  # Max 50% of balance for short positions
            shares_to_short = max_short_value / (effective_price * (1 + transaction_cost))

            if shares_to_short > 0:
                # Calculate market impact from short selling
                trade_impact = self._calculate_market_impact(shares_to_short, price, current_volume)

                # Impact pushes price DOWN for short sales (negative impact)
                effective_price_with_impact = effective_price - trade_impact

                # When shorting, we receive cash but owe shares
                # The cash goes into our balance, but we track the liability via negative shares_held
                revenue = shares_to_short * effective_price_with_impact * (1 - transaction_cost)
                self.balance += revenue  # Receive proceeds from short sale
                self.shares_held = -shares_to_short  # Track short position as negative shares
                self.position = self.shares_held  # Negative position

                # Update cumulative market impact (negative for short sales)
                timestamp = self.market_data.index[self.current_data_index]
                self._update_cumulative_impact(-trade_impact, timestamp)

                return "short"

        return "hold"

    def _close_short_position(self, price: float, transaction_cost_bps: float, slippage_multiplier: float) -> str:
        """Close an existing short position by buying back shares."""
        if self.shares_held >= 0:
            return "hold"  # No short position to close

        # Get current volume for impact calculation
        current_row = self.market_data.iloc[self.current_data_index]
        current_volume = current_row.get("volume", 1000000.0)

        transaction_cost = transaction_cost_bps / 10000.0
        slippage = price * 0.0001 * slippage_multiplier
        effective_price = price + slippage  # Pay ask price when buying to cover

        shares_to_cover = abs(self.shares_held)

        # Calculate market impact from covering short
        trade_impact = self._calculate_market_impact(shares_to_cover, price, current_volume)

        # Impact pushes price UP when buying to cover
        effective_price_with_impact = effective_price + trade_impact

        total_cost = shares_to_cover * effective_price_with_impact * (1 + transaction_cost)

        if total_cost <= self.balance:
            # Buy back the shares to close short
            self.balance -= total_cost
            self.shares_held = 0.0
            self.position = 0.0

            # Update cumulative market impact (positive for buy to cover)
            timestamp = self.market_data.index[self.current_data_index]
            self._update_cumulative_impact(trade_impact, timestamp)

            return "cover_short"
        else:
            # Not enough cash to cover - this is a margin call situation
            # In reality, broker would force liquidation, but we'll just hold
            return "hold"

    def _calculate_benchmark_return(self) -> float:
        """Calculate benchmark return (buy-and-hold strategy)."""
        if len(self.price_history) < 2:
            return 0.0

        current_price = self.price_history[-1]["price"]
        previous_price = self.price_history[-2]["price"]

        return (current_price - previous_price) / previous_price

    def _calculate_reward(self, portfolio_return: float, benchmark_return: float) -> float:
        """Calculate reward based on risk-adjusted performance."""
        # Base reward is excess return over benchmark
        excess_return = portfolio_return - benchmark_return

        # Add penalty for high volatility (simplified Sharpe ratio concept)
        if len(self.portfolio_returns) > 10:
            volatility = np.std(self.portfolio_returns[-10:])
            volatility_penalty = volatility * 0.5
        else:
            volatility_penalty = 0.0

        # Scale reward for learning
        reward = (excess_return * 100) - volatility_penalty

        return float(reward)

    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio for current performance."""
        return calculate_sharpe_ratio(np.array(self.portfolio_returns))

    def _get_observation(self) -> np.ndarray:
        """Create comprehensive observation from current market state."""
        if self.market_data is None or self.current_data_index < self.window_size:
            # Return zero observation if no data available
            # Calculate expected observation size: window_size * 6 + 1 + 1 + 1 + 4
            obs_size = self.window_size * 6 + 1 + 1 + 1 + 4
            return np.zeros(obs_size, dtype=np.float32)

        # Get historical price window
        start_idx = max(0, self.current_data_index - self.window_size)
        end_idx = self.current_data_index

        price_window_data = self.market_data.iloc[start_idx:end_idx]

        # Extract OHLCV features
        price_features = []
        for _, row in price_window_data.iterrows():
            # Normalize prices by current close for scale invariance
            current_close = self.market_data.iloc[self.current_data_index]["close"]

            price_features.extend(
                [
                    row["open"] / current_close,
                    row["high"] / current_close,
                    row["low"] / current_close,
                    row["close"] / current_close,
                    np.log(row["volume"] + 1) / 20.0,  # Log-normalized volume
                    (row["high"] + row["low"]) / (2 * current_close),  # Volume-weighted price proxy
                ]
            )

        # Pad if necessary
        expected_price_features = self.window_size * 6
        while len(price_features) < expected_price_features:
            price_features.insert(0, 0.0)

        price_features = price_features[-expected_price_features:]  # Take last window_size elements

        # Portfolio state features
        starting_value = 100000.0  # Default starting balance
        normalized_position = self.position / 1000.0  # Scale position
        normalized_balance = self.balance / starting_value
        normalized_portfolio = self.portfolio_value / starting_value

        # Time features for intraday patterns
        current_timestamp = self.market_data.index[self.current_data_index]
        hour_of_day = current_timestamp.hour / 24.0
        day_of_week = current_timestamp.weekday() / 7.0
        month_of_year = current_timestamp.month / 12.0
        year_progress = (current_timestamp.dayofyear) / 365.0

        # Combine all features
        observation = np.array(
            price_features
            + [normalized_position]
            + [normalized_balance]
            + [normalized_portfolio]
            + [hour_of_day, day_of_week, month_of_year, year_progress],
            dtype=np.float32,
        )

        # Ensure observation matches expected shape
        expected_size = self.window_size * 6 + 1 + 1 + 1 + 4  # Same calculation as in __init__
        if len(observation) != expected_size:
            # Pad or truncate to match expected size
            if len(observation) < expected_size:
                observation = np.pad(observation, (0, expected_size - len(observation)))
            else:
                observation = observation[:expected_size]

        return observation

    def render(self, mode: str | None = None):
        """
        Provide dummy pixel output for compatibility with image-based wrappers.

        Financial environments don't have visual rendering, so we return a small
        dummy array that FinancialWrapper will strip out from the info dict.
        """
        # Return a small dummy RGB array (64x64x3) for AddPixelsWrapper compatibility
        return np.zeros((64, 64, 3), dtype=np.uint8)

    def close(self):
        """Clean up environment resources."""
        pass

    def get_backtest_results(self) -> dict[str, Any]:
        """Get comprehensive backtesting performance metrics."""
        if not self.trade_history:
            return {}

        # Extract time series data
        returns = np.array(self.portfolio_returns)
        benchmark_returns = np.array(self.benchmark_returns)
        portfolio_values = [t["portfolio_value"] for t in self.trade_history]
        timestamps = [t["timestamp"] for t in self.trade_history]

        # Calculate time period
        start_time = timestamps[0]
        end_time = timestamps[-1]
        trading_days = len({t.date() for t in timestamps})
        total_periods = len(returns)

        # Returns metrics
        total_return = (self.portfolio_value - self.initial_portfolio_value) / max(self.initial_portfolio_value, 1e-8)

        # Annualized return (assuming 252 trading days, 390 minutes per day)
        periods_per_year = 252 * 390
        annualized_return = calculate_annualized_return(total_return, total_periods, periods_per_year)

        # Cumulative returns
        cumulative_returns = pd.Series(returns).add(1).cumprod().sub(1)

        # Benchmark performance
        initial_price = self.trade_history[0]["price"]
        final_price = self.trade_history[-1]["price"]
        benchmark_return = (final_price - initial_price) / initial_price
        benchmark_annualized = calculate_annualized_return(benchmark_return, total_periods, periods_per_year)

        # Risk metrics
        risk_free_rate = 0.02

        # Volatility
        volatility = calculate_volatility(returns, periods_per_year)
        downside_volatility = calculate_downside_volatility(returns, periods_per_year)

        # Sharpe ratio
        sharpe_ratio = calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)

        # Sortino ratio (using downside deviation)
        sortino_ratio = calculate_sortino_ratio(returns, risk_free_rate, periods_per_year)

        # Calmar ratio (return / max drawdown)
        max_drawdown = self._calculate_max_drawdown()
        calmar_ratio = calculate_calmar_ratio(annualized_return, max_drawdown)

        # Alpha and Beta (relative to benchmark)
        alpha, beta = calculate_alpha_beta(
            returns,
            benchmark_returns,
            annualized_return,
            benchmark_annualized,
            risk_free_rate,
        )

        # Information ratio (excess return / tracking error)
        information_ratio, tracking_error = calculate_information_ratio(
            returns,
            benchmark_returns,
            annualized_return,
            benchmark_annualized,
            periods_per_year,
        )

        # Drawdown analysis
        drawdown_info = self._analyze_drawdowns(portfolio_values, timestamps)

        # Distribution metrics
        skewness, kurtosis = calculate_skewness_kurtosis(returns)

        # Value at Risk (95% and 99%)
        var_95, cvar_95 = calculate_var_cvar(returns, confidence_level=0.95)
        var_99, cvar_99 = calculate_var_cvar(returns, confidence_level=0.99)

        # Tail ratio
        tail_ratio = calculate_tail_ratio(returns)

        # Omega ratio
        risk_free_period = risk_free_rate / periods_per_year
        omega_ratio = calculate_omega_ratio(returns, threshold=risk_free_period)

        # Trading statistics
        trade_stats = self._analyze_trades()

        # Position analysis
        position_stats = self._analyze_positions()

        # Stability
        # Measure consistency of returns (R-squared of returns vs time)
        stability = calculate_stability(returns)

        return {
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            "annualized_return": annualized_return,
            "annualized_return_pct": annualized_return * 100,
            "cumulative_return": (cumulative_returns.iloc[-1] if len(cumulative_returns) > 0 else 0.0),
            "benchmark_return": benchmark_return,
            "benchmark_return_pct": benchmark_return * 100,
            "benchmark_annualized": benchmark_annualized,
            "benchmark_annualized_pct": benchmark_annualized * 100,
            "excess_return": total_return - benchmark_return,
            "excess_return_pct": (total_return - benchmark_return) * 100,
            "alpha": alpha,
            "alpha_pct": alpha * 100,
            "beta": beta,
            "volatility": volatility,
            "volatility_pct": volatility * 100,
            "downside_volatility": downside_volatility,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            "information_ratio": information_ratio,
            "tracking_error": tracking_error,
            "stability": stability,
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": max_drawdown * 100,
            **drawdown_info,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "var_95": var_95,
            "var_95_pct": var_95 * 100,
            "var_99": var_99,
            "var_99_pct": var_99 * 100,
            "cvar_95": cvar_95,
            "cvar_99": cvar_99,
            "tail_ratio": tail_ratio,
            "omega_ratio": omega_ratio,
            "starting_value": self.initial_portfolio_value,
            "final_value": self.portfolio_value,
            "current_balance": self.balance,
            "current_position": self.position,
            "shares_held": self.shares_held,
            **position_stats,
            **trade_stats,
            "start_date": start_time,
            "end_date": end_time,
            "trading_days": trading_days,
            "total_periods": total_periods,
            "years": total_periods / periods_per_year if total_periods > 0 else 0,
            "symbol": self.current_symbol,
            "market_regime": "normal",
            "transaction_cost_bps": 10.0,
        }

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from peak portfolio value."""
        if not self.trade_history:
            return 0.0

        portfolio_values = [t["portfolio_value"] for t in self.trade_history]
        return calculate_max_drawdown(portfolio_values)

    def _analyze_drawdowns(self, portfolio_values: list[float], timestamps: list) -> dict[str, Any]:
        """
        Analyze drawdown periods following PyFolio methodology.

        Returns information about:
        - Maximum drawdown
        - Longest drawdown duration
        - Top N drawdown periods
        - Current drawdown
        - Recovery times
        """
        if len(portfolio_values) < 2:
            return {
                "max_drawdown_duration": 0,
                "current_drawdown": 0.0,
                "avg_drawdown": 0.0,
                "avg_drawdown_duration": 0,
                "num_drawdown_periods": 0,
            }

        # Calculate running drawdowns
        values_series = pd.Series(portfolio_values, index=timestamps)
        running_max = values_series.expanding().max()
        drawdowns = (values_series - running_max) / running_max

        # Find drawdown periods (contiguous periods where drawdown < 0)
        in_drawdown = drawdowns < 0
        drawdown_periods = []

        start_idx = None
        for i, is_dd in enumerate(in_drawdown):
            if is_dd and start_idx is None:
                start_idx = i
            elif not is_dd and start_idx is not None:
                # End of drawdown period
                dd_period = drawdowns.iloc[start_idx:i]
                max_dd = dd_period.min()
                duration = i - start_idx

                drawdown_periods.append(
                    {
                        "start": timestamps[start_idx],
                        "end": timestamps[i - 1],
                        "duration": duration,
                        "max_drawdown": abs(max_dd),
                        "recovery_time": 0,  # Will be updated if recovered
                    }
                )
                start_idx = None

        # Handle ongoing drawdown
        current_drawdown = abs(drawdowns.iloc[-1]) if drawdowns.iloc[-1] < 0 else 0.0

        # Statistics
        if drawdown_periods:
            avg_drawdown = np.mean([p["max_drawdown"] for p in drawdown_periods])
            avg_duration = np.mean([p["duration"] for p in drawdown_periods])
            max_duration = max([p["duration"] for p in drawdown_periods])
        else:
            avg_drawdown = 0.0
            avg_duration = 0
            max_duration = 0

        return {
            "max_drawdown_duration": max_duration,
            "current_drawdown": current_drawdown,
            "current_drawdown_pct": current_drawdown * 100,
            "avg_drawdown": avg_drawdown,
            "avg_drawdown_pct": avg_drawdown * 100,
            "avg_drawdown_duration": avg_duration,
            "num_drawdown_periods": len(drawdown_periods),
            "drawdown_periods": drawdown_periods[:5],  # Top 5 for reporting
        }

    def _analyze_trades(self) -> dict[str, Any]:
        """
        Analyze trading statistics following Zipline/PyFolio methodology.

        Returns metrics about:
        - Trade counts (total, buy, sell, short)
        - Win/loss statistics
        - Profit factor
        - Average trade metrics
        - Trade frequency and turnover
        """
        if not self.trade_history:
            return {
                "total_trades": 0,
                "buy_trades": 0,
                "sell_trades": 0,
                "short_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_trade_return": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
                "trading_frequency": 0.0,
            }

        # Extract trades (only actual buy/sell/short actions)
        trades = [t for t in self.trade_history if t["action"] in ["buy", "sell", "short"]]

        # Count trade types
        buy_trades = len([t for t in trades if t["action"] == "buy"])
        sell_trades = len([t for t in trades if t["action"] == "sell"])
        short_trades = len([t for t in trades if t["action"] == "short"])
        total_trades = len(trades)

        # Analyze trade returns (looking at returns after each trade)
        trade_returns = []
        for i, trade in enumerate(trades):
            if i > 0:
                prev_value = trades[i - 1]["portfolio_value"]
                current_value = trade["portfolio_value"]
                trade_return = (current_value - prev_value) / prev_value if prev_value > 0 else 0.0
                trade_returns.append(trade_return)

        if trade_returns:
            winning_trades = [r for r in trade_returns if r > 0]
            losing_trades = [r for r in trade_returns if r < 0]

            win_rate = len(winning_trades) / len(trade_returns) if trade_returns else 0.0

            # Profit factor: sum of wins / abs(sum of losses)
            total_wins = sum(winning_trades) if winning_trades else 0.0
            total_losses = abs(sum(losing_trades)) if losing_trades else 0.0
            profit_factor = total_wins / total_losses if total_losses > 0 else float("inf") if total_wins > 0 else 0.0

            avg_trade_return = np.mean(trade_returns)
            avg_win = np.mean(winning_trades) if winning_trades else 0.0
            avg_loss = np.mean(losing_trades) if losing_trades else 0.0
            largest_win = max(winning_trades) if winning_trades else 0.0
            largest_loss = min(losing_trades) if losing_trades else 0.0
        else:
            win_rate = 0.0
            profit_factor = 0.0
            avg_trade_return = 0.0
            avg_win = 0.0
            avg_loss = 0.0
            largest_win = 0.0
            largest_loss = 0.0

        # Trading frequency (trades per day)
        if self.trade_history:
            start_time = self.trade_history[0]["timestamp"]
            end_time = self.trade_history[-1]["timestamp"]
            trading_days = (end_time - start_time).days + 1
            trading_frequency = total_trades / max(trading_days, 1)
        else:
            trading_frequency = 0.0

        return {
            "total_trades": total_trades,
            "buy_trades": buy_trades,
            "sell_trades": sell_trades,
            "short_trades": short_trades,
            "winning_trades": len(winning_trades) if trade_returns else 0,
            "losing_trades": len(losing_trades) if trade_returns else 0,
            "win_rate": win_rate,
            "win_rate_pct": win_rate * 100,
            "profit_factor": profit_factor if profit_factor != float("inf") else 999.99,
            "avg_trade_return": avg_trade_return,
            "avg_trade_return_pct": avg_trade_return * 100,
            "avg_win": avg_win,
            "avg_win_pct": avg_win * 100,
            "avg_loss": avg_loss,
            "avg_loss_pct": avg_loss * 100,
            "largest_win": largest_win,
            "largest_win_pct": largest_win * 100,
            "largest_loss": largest_loss,
            "largest_loss_pct": largest_loss * 100,
            "trading_frequency": trading_frequency,
        }

    def _analyze_positions(self) -> dict[str, Any]:
        """
        Analyze position and portfolio statistics following Zipline/PyFolio methodology.

        Returns metrics about:
        - Position concentration
        - Leverage (gross, net)
        - Exposure (long, short)
        - Turnover
        - Cash usage
        """
        if not self.trade_history:
            return {
                "avg_position_size": 0.0,
                "max_position_size": 0.0,
                "avg_leverage": 0.0,
                "max_leverage": 0.0,
                "avg_exposure": 0.0,
                "max_exposure": 0.0,
            }

        # Extract position data over time
        positions = [t["shares_held"] * t["price"] for t in self.trade_history]
        portfolio_values = [t["portfolio_value"] for t in self.trade_history]

        # Position concentration (position value / portfolio value)
        position_concentrations = [abs(pos) / pv if pv > 0 else 0.0 for pos, pv in zip(positions, portfolio_values)]

        # Leverage (position value / portfolio value)
        leverages = [abs(pos) / pv if pv > 0 else 0.0 for pos, pv in zip(positions, portfolio_values)]

        # Exposure (signed position value / portfolio value)
        exposures = [pos / pv if pv > 0 else 0.0 for pos, pv in zip(positions, portfolio_values)]

        # Calculate statistics
        avg_position_size = np.mean([abs(p) for p in positions])
        max_position_size = max([abs(p) for p in positions]) if positions else 0.0

        avg_leverage = np.mean(leverages) if leverages else 0.0
        max_leverage = max(leverages) if leverages else 0.0

        avg_exposure = np.mean(exposures) if exposures else 0.0
        max_exposure = max([abs(e) for e in exposures]) if exposures else 0.0

        # Turnover analysis (how much trading relative to portfolio size)
        # Sum of absolute value of trades / average portfolio value
        trade_volumes = []
        for i in range(1, len(self.trade_history)):
            prev_shares = self.trade_history[i - 1]["shares_held"]
            curr_shares = self.trade_history[i]["shares_held"]
            curr_price = self.trade_history[i]["price"]

            trade_volume = abs(curr_shares - prev_shares) * curr_price
            trade_volumes.append(trade_volume)

        avg_portfolio_value = np.mean(portfolio_values)
        total_turnover = sum(trade_volumes) / avg_portfolio_value if avg_portfolio_value > 0 else 0.0

        return {
            "avg_position_size": avg_position_size,
            "max_position_size": max_position_size,
            "avg_position_concentration": (np.mean(position_concentrations) if position_concentrations else 0.0),
            "max_position_concentration": (max(position_concentrations) if position_concentrations else 0.0),
            "avg_leverage": avg_leverage,
            "max_leverage": max_leverage,
            "avg_exposure": avg_exposure,
            "avg_exposure_pct": avg_exposure * 100,
            "max_exposure": max_exposure,
            "max_exposure_pct": max_exposure * 100,
            "total_turnover": total_turnover,
            "avg_portfolio_value": avg_portfolio_value,
        }

    def get_returns_tear_sheet_data(self) -> dict[str, Any]:
        """
        Generate data for returns tear sheet analysis (PyFolio style).

        Includes:
        - Rolling metrics (returns, Sharpe, volatility, beta)
        - Drawdown analysis
        - Return distribution
        - Monthly/annual returns
        - Return quantiles
        """
        if not self.trade_history or len(self.portfolio_returns) < 2:
            return {}

        returns = pd.Series(self.portfolio_returns)
        benchmark_returns = pd.Series(self.benchmark_returns)
        timestamps = [t["timestamp"] for t in self.trade_history]
        returns.index = timestamps
        benchmark_returns.index = timestamps[: len(benchmark_returns)]

        # Rolling metrics (using 60-period window ~ 1 hour)
        rolling_window = min(60, len(returns) // 2)
        if rolling_window >= 2:
            rolling_returns = returns.rolling(rolling_window).mean()
            rolling_vol = returns.rolling(rolling_window).std() * np.sqrt(252 * 390)
            rolling_sharpe = (rolling_returns * 252 * 390) / (
                returns.rolling(rolling_window).std() * np.sqrt(252 * 390)
            )

            # Rolling beta
            if len(returns) == len(benchmark_returns):
                rolling_beta = (
                    returns.rolling(rolling_window).cov(benchmark_returns)
                    / benchmark_returns.rolling(rolling_window).var()
                )
            else:
                rolling_beta = pd.Series([0.0] * len(returns), index=returns.index)
        else:
            rolling_returns = returns
            rolling_vol = pd.Series([0.0] * len(returns), index=returns.index)
            rolling_sharpe = pd.Series([0.0] * len(returns), index=returns.index)
            rolling_beta = pd.Series([0.0] * len(returns), index=returns.index)

        # Cumulative returns
        cumulative_returns = (1 + returns).cumprod() - 1

        # Drawdown underwater plot data
        running_max = (1 + returns).cumprod().expanding().max()
        underwater = ((1 + returns).cumprod() / running_max) - 1

        # Monthly/annual aggregation
        try:
            monthly_returns = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
            annual_returns = returns.resample("YE").apply(lambda x: (1 + x).prod() - 1)
        except Exception:
            # If resampling fails, create empty series
            monthly_returns = pd.Series(dtype=float)
            annual_returns = pd.Series(dtype=float)

        # Return distribution quantiles
        quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
        return_quantiles = {f"q{int(q * 100)}": np.quantile(returns, q) for q in quantiles}

        return {
            "returns": returns.tolist(),
            "cumulative_returns": cumulative_returns.tolist(),
            "rolling_returns": rolling_returns.tolist(),
            "rolling_volatility": rolling_vol.tolist(),
            "rolling_sharpe": rolling_sharpe.tolist(),
            "rolling_beta": rolling_beta.tolist(),
            "underwater": underwater.tolist(),
            "monthly_returns": (monthly_returns.to_dict() if len(monthly_returns) > 0 else {}),
            "annual_returns": (annual_returns.to_dict() if len(annual_returns) > 0 else {}),
            "return_quantiles": return_quantiles,
            "timestamps": [str(t) for t in timestamps],
        }

    def get_position_tear_sheet_data(self) -> dict[str, Any]:
        """
        Generate data for position analysis tear sheet (PyFolio style).

        Includes:
        - Position exposure over time
        - Long/short holdings
        - Position concentration
        - Gross/net leverage
        - Holdings breakdown
        """
        if not self.trade_history:
            return {}

        timestamps = [t["timestamp"] for t in self.trade_history]
        portfolio_values = [t["portfolio_value"] for t in self.trade_history]
        positions = [t["shares_held"] * t["price"] for t in self.trade_history]

        # Position exposure (% of portfolio)
        exposures = [pos / pv if pv > 0 else 0.0 for pos, pv in zip(positions, portfolio_values)]

        # Long/short split
        long_exposure = [max(e, 0) for e in exposures]
        short_exposure = [min(e, 0) for e in exposures]

        # Gross leverage (sum of absolute values)
        gross_leverage = [abs(e) for e in exposures]

        # Net exposure (long - short)
        net_exposure = exposures

        # Top positions (just one symbol in this simple implementation)
        top_holdings = {
            self.current_symbol: {
                "avg_weight": np.mean([abs(e) for e in exposures]),
                "max_weight": max([abs(e) for e in exposures]) if exposures else 0.0,
                "current_weight": abs(exposures[-1]) if exposures else 0.0,
            }
        }

        return {
            "timestamps": [str(t) for t in timestamps],
            "gross_exposure": [abs(e) for e in exposures],
            "net_exposure": net_exposure,
            "long_exposure": long_exposure,
            "short_exposure": short_exposure,
            "gross_leverage": gross_leverage,
            "top_holdings": top_holdings,
            "position_concentration": [abs(e) for e in exposures],
        }

    def get_transaction_tear_sheet_data(self) -> dict[str, Any]:
        """
        Generate data for transaction analysis tear sheet (PyFolio style).

        Includes:
        - Transaction volume over time
        - Turnover analysis
        - Trade timing distribution
        - Transaction costs
        - Slippage analysis
        """
        if not self.trade_history:
            return {}

        # Extract only actual transactions (not holds)
        transactions = [t for t in self.trade_history if t["action"] in ["buy", "sell", "short"]]

        if not transactions:
            return {}

        # Calculate volumes
        volumes = []
        for trans in transactions:
            volume = abs(trans["shares_held"]) * trans["price"]
            volumes.append(volume)

        # Turnover over time (transaction volume / portfolio value)
        turnovers = []
        for trans in transactions:
            turnover = abs(trans["shares_held"]) * trans["price"] / trans["portfolio_value"]
            turnovers.append(turnover)

        # Time distribution (by hour of day)
        hours = [t["timestamp"].hour for t in transactions]
        hour_distribution = pd.Series(hours).value_counts().sort_index().to_dict()

        # Daily turnover
        timestamps = [t["timestamp"] for t in transactions]
        daily_turnover = pd.Series(turnovers, index=timestamps).resample("D").sum().to_dict()

        return {
            "transaction_count": len(transactions),
            "transaction_timestamps": [str(t["timestamp"]) for t in transactions],
            "transaction_volumes": volumes,
            "average_volume": np.mean(volumes) if volumes else 0.0,
            "total_volume": sum(volumes),
            "turnovers": turnovers,
            "average_turnover": np.mean(turnovers) if turnovers else 0.0,
            "hour_distribution": hour_distribution,
            "daily_turnover": {str(k): v for k, v in daily_turnover.items()},
            "transaction_types": {
                "buy": len([t for t in transactions if t["action"] == "buy"]),
                "sell": len([t for t in transactions if t["action"] == "sell"]),
                "short": len([t for t in transactions if t["action"] == "short"]),
            },
        }

    def get_round_trip_tear_sheet_data(self) -> dict[str, Any]:
        """
        Generate data for round-trip trade analysis (PyFolio style).

        A round trip is a complete trade cycle (entry to exit).
        Includes:
        - Individual round trip returns
        - Round trip duration
        - PnL distribution
        - Win/loss analysis
        """
        if not self.trade_history:
            return {}

        # Identify round trips (pairs of opposing trades)
        round_trips = []
        entry_trade = None
        entry_price = 0.0
        entry_shares = 0.0

        for trade in self.trade_history:
            action = trade["action"]

            if action in ["buy", "short"]:
                if entry_trade is None:
                    # Start of new round trip
                    entry_trade = trade
                    entry_price = trade["price"]
                    entry_shares = abs(trade["shares_held"])

            elif action == "sell" and entry_trade is not None:
                # End of round trip
                exit_price = trade["price"]
                exit_time = trade["timestamp"]

                # Calculate PnL
                if entry_trade["action"] == "buy":
                    pnl = (exit_price - entry_price) * entry_shares
                    pnl_pct = (exit_price / entry_price - 1) if entry_price > 0 else 0.0
                else:  # short
                    pnl = (entry_price - exit_price) * entry_shares
                    pnl_pct = (entry_price / exit_price - 1) if exit_price > 0 else 0.0

                duration = (exit_time - entry_trade["timestamp"]).total_seconds() / 60  # minutes

                round_trips.append(
                    {
                        "entry_time": entry_trade["timestamp"],
                        "exit_time": exit_time,
                        "duration_minutes": duration,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "shares": entry_shares,
                        "pnl": pnl,
                        "pnl_pct": pnl_pct,
                        "direction": entry_trade["action"],
                    }
                )

                entry_trade = None

        if not round_trips:
            return {}

        # Analyze round trips
        pnls = [rt["pnl"] for rt in round_trips]
        pnl_pcts = [rt["pnl_pct"] for rt in round_trips]
        durations = [rt["duration_minutes"] for rt in round_trips]

        winning_trips = [rt for rt in round_trips if rt["pnl"] > 0]
        losing_trips = [rt for rt in round_trips if rt["pnl"] < 0]

        return {
            "total_round_trips": len(round_trips),
            "winning_trips": len(winning_trips),
            "losing_trips": len(losing_trips),
            "win_rate": len(winning_trips) / len(round_trips) if round_trips else 0.0,
            "avg_pnl": np.mean(pnls),
            "avg_pnl_pct": np.mean(pnl_pcts) * 100,
            "avg_winner": (np.mean([rt["pnl"] for rt in winning_trips]) if winning_trips else 0.0),
            "avg_loser": (np.mean([rt["pnl"] for rt in losing_trips]) if losing_trips else 0.0),
            "best_trade": max(pnls) if pnls else 0.0,
            "worst_trade": min(pnls) if pnls else 0.0,
            "avg_duration_minutes": np.mean(durations),
            "total_pnl": sum(pnls),
            "round_trips": round_trips[:10],  # Return top 10 for reporting
        }

    def get_interesting_periods_analysis(self, periods: dict[str, tuple[str, str]] | None = None) -> dict[str, Any]:
        """
        Analyze returns during historically interesting market periods (PyFolio style).

        Examples: market crashes, flash crashes, significant events, etc.

        Parameters
        ----------
        periods : dict
            Dictionary mapping period names to (start_date, end_date) tuples
            Example: {"Flash Crash": ("2010-05-06", "2010-05-07")}
        """
        if not self.trade_history or periods is None:
            return {}

        returns = pd.Series(self.portfolio_returns)
        timestamps = [t["timestamp"] for t in self.trade_history]
        returns.index = timestamps

        period_analysis = {}

        for period_name, (start_str, end_str) in periods.items():
            try:
                start_date = pd.Timestamp(start_str)
                end_date = pd.Timestamp(end_str)

                # Extract returns for this period
                period_returns = returns[(returns.index >= start_date) & (returns.index <= end_date)]

                if len(period_returns) > 0:
                    period_analysis[period_name] = {
                        "total_return": (1 + period_returns).prod() - 1,
                        "mean_return": period_returns.mean(),
                        "volatility": period_returns.std(),
                        "sharpe": (period_returns.mean() / period_returns.std() if period_returns.std() > 0 else 0.0),
                        "max_drawdown": self._calculate_period_max_drawdown(period_returns),
                        "num_observations": len(period_returns),
                    }
            except Exception:
                continue

        return period_analysis

    def _calculate_period_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate max drawdown for a specific period."""
        if len(returns) < 2:
            return 0.0

        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        return abs(drawdown.min()) if len(drawdown) > 0 else 0.0

    def get_capacity_analysis(self, volume_limits: dict[str, float] | None = None) -> dict[str, Any]:
        """
        Analyze strategy capacity constraints (PyFolio style).

        Capacity analysis helps determine:
        - Maximum AUM the strategy can handle
        - Impact of increased trading on performance
        - Liquidity constraints

        Parameters
        ----------
        volume_limits : dict
            Dictionary of volume constraints
            Example: {"daily_volume_limit": 0.05, "liquidation_limit": 0.2}
        """
        if not self.trade_history or not self.price_history:
            return {}

        if volume_limits is None:
            volume_limits = {
                "daily_volume_limit": 0.05,  # 5% of daily volume
                "liquidation_limit": 0.2,  # 20% for full liquidation
            }

        # Calculate average daily volume
        daily_volumes = {}
        for price_record in self.price_history:
            date = price_record["timestamp"].date()
            volume = price_record["volume"]

            if date not in daily_volumes:
                daily_volumes[date] = []
            daily_volumes[date].append(volume)

        avg_daily_volumes = {date: np.mean(vols) for date, vols in daily_volumes.items()}
        overall_avg_volume = np.mean(list(avg_daily_volumes.values())) if avg_daily_volumes else 0.0

        # Calculate maximum position size based on volume constraints
        daily_limit = volume_limits["daily_volume_limit"]
        max_position_shares = overall_avg_volume * daily_limit

        # Calculate actual positions as % of volume limit
        actual_positions = [t["shares_held"] for t in self.trade_history]
        position_pcts = [
            abs(pos) / max_position_shares if max_position_shares > 0 else 0.0 for pos in actual_positions
        ]

        # Estimate capacity (how much we can scale up)
        max_position_pct = max(position_pcts) if position_pcts else 0.0
        capacity_multiplier = 1.0 / max_position_pct if max_position_pct > 0 else float("inf")

        return {
            "avg_daily_volume": overall_avg_volume,
            "volume_limit_pct": daily_limit * 100,
            "max_position_shares": max_position_shares,
            "max_position_pct_of_volume": max_position_pct * 100,
            "capacity_multiplier": min(capacity_multiplier, 999.99),  # Cap at reasonable value
            "current_max_position": (max([abs(p) for p in actual_positions]) if actual_positions else 0.0),
            "liquidity_constrained": max_position_pct > 0.5,  # Flag if using >50% of limit
        }

    def get_data_pipeline_stats(self) -> dict[str, Any]:
        raise NotImplementedError(
            "Real data pipeline stats required. Please integrate actual performance metrics from Alpaca data pipeline."
        )

    def reset_to_date(self, target_date: str, symbol: str | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset environment to specific historical date and symbol."""
        if symbol:
            self.current_symbol = symbol

        # Load historical data for the target date
        extended_end_date = pd.to_datetime(target_date) + timedelta(days=90)  # Load extra data for episode
        self.market_data = self._load_historical_data(
            self.current_symbol, target_date, extended_end_date.strftime("%Y-%m-%d")
        )

        # Ensure window_size is set (use default if not already set from variation)
        if self.window_size is None:
            self.window_size = 60  # Default window size in minutes

        # Reset environment state
        self.current_step = 0
        self.current_data_index = int(self.window_size)  # Ensure it's an integer
        self.position = 0.0
        self.balance = 100000.0  # Default starting balance
        self.portfolio_value = self.balance
        self.shares_held = 0.0
        self.trade_history = []
        self.price_history = []
        self.portfolio_returns = []
        self.benchmark_returns = []

        self.start_time = self.market_data.index[self.current_data_index]
        self.initial_portfolio_value = self.portfolio_value

        observation = self._get_observation()
        info = {
            "time_machine_date": target_date,
            "symbol": self.current_symbol,
            "market_data_length": len(self.market_data),
            "start_timestamp": self.start_time,
        }

        return observation, info
