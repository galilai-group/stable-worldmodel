"""Tests for Financial Backtesting Environment.

All tests in this module use synthetic data via an autouse fixture.
No external API credentials or data sources are required.
The library will work for users even without a registered dataset,
as it falls back to the default Alpaca loader (or synthetic data in tests).
"""

import os

import gymnasium as gym
import numpy as np
import pandas as pd
import pytest

from stable_worldmodel.envs import (
    register_financial_dataset,
    unregister_financial_dataset,
)


# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Check if Alpaca credentials are available
ALPACA_CONFIGURED = bool(os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_SECRET_KEY"))

# Mark for tests that require actual data loading (DEPRECATED - now using synthetic data)
requires_data = pytest.mark.skipif(
    False,  # Always run tests with synthetic data
    reason="DEPRECATED: All tests now use synthetic data",
)


@pytest.fixture(autouse=True)
def use_synthetic_data():
    """Automatically use synthetic data for all tests in this module.

    This fixture registers a synthetic data loader before each test and
    unregisters it after, ensuring all tests pass without requiring
    external API credentials.
    """

    def synthetic_loader(symbol, start_date, end_date, **kwargs):
        """Generate synthetic financial data for testing."""
        # Generate enough data for any test (at least 2000 data points)
        timestamps = pd.date_range(start=start_date, periods=2000, freq="1min")

        # Create realistic random walk price data
        np.random.seed(42)  # Consistent data for tests
        returns = np.random.normal(0, 0.001, size=len(timestamps))
        price = 100 * (1 + returns).cumprod()

        return pd.DataFrame(
            {
                "open": price * (1 + np.random.uniform(-0.001, 0.001, len(timestamps))),
                "high": price * (1 + np.random.uniform(0, 0.002, len(timestamps))),
                "low": price * (1 + np.random.uniform(-0.002, 0, len(timestamps))),
                "close": price,
                "volume": np.random.randint(1000, 10000, len(timestamps)),
                "trade_count": np.random.randint(10, 100, len(timestamps)),
                "vwap": price * (1 + np.random.uniform(-0.0005, 0.0005, len(timestamps))),
            },
            index=timestamps,
        )

    # Register the synthetic data loader
    register_financial_dataset(synthetic_loader, name="default")

    # Run the test
    yield

    # Clean up after the test
    unregister_financial_dataset("default")


def test_environment_creation():
    """Test that the environment can be created successfully."""
    env = gym.make("swm/Financial-v0")
    assert env is not None
    env.close()


def test_action_space():
    """Test that action space is correctly defined."""
    env = gym.make("swm/Financial-v0")
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == 3  # BUY, SELL, HOLD
    env.close()


def test_observation_space():
    """Test that observation space is correctly defined."""
    env = gym.make("swm/Financial-v0")
    assert isinstance(env.observation_space, gym.spaces.Box)
    # window_size*6 + position + balance + portfolio_value + time_features(4)
    # Default window_size is 60
    expected_shape = (60 * 6 + 1 + 1 + 1 + 4,)
    assert env.observation_space.shape == expected_shape
    env.close()


def test_reset():
    """Test environment reset functionality."""
    env = gym.make("swm/Financial-v0")
    obs, info = env.reset(seed=42)

    assert obs is not None
    # Default window_size is 60, so: 60*6 + position + balance + portfolio + time_features(4)
    assert len(obs) == 60 * 6 + 1 + 1 + 1 + 4
    assert isinstance(info, dict)
    assert "symbol" in info
    assert "market_data_length" in info
    env.close()


def test_step():
    """Test environment step functionality."""
    env = gym.make("swm/Financial-v0", max_steps=10)
    obs, info = env.reset(seed=42)

    # Test BUY action
    obs, reward, terminated, truncated, info = env.step(1)  # BUY
    assert obs is not None
    assert isinstance(reward, int | float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool | np.bool_)
    assert isinstance(info, dict)

    # Test SELL action
    obs, reward, terminated, truncated, info = env.step(0)  # SELL
    assert obs is not None

    # Test HOLD action
    obs, reward, terminated, truncated, info = env.step(2)  # HOLD
    assert obs is not None
    env.close()


def test_variation_space_structure():
    """Test that variation space has complete structure with Alpaca integration."""
    env = gym.make("swm/Financial-v0")
    unwrapped_env = env.unwrapped

    assert hasattr(unwrapped_env, "variation_space")
    assert "agent" in unwrapped_env.variation_space.spaces

    agent_space = unwrapped_env.variation_space["agent"]
    # All agent parameters should be present with Alpaca integration
    assert "starting_balance" in agent_space.spaces
    assert "transaction_cost" in agent_space.spaces
    assert "enable_shorting" in agent_space.spaces
    assert "market_impact" in agent_space.spaces  # Renamed from slippage_pct
    assert "window_size" in agent_space.spaces

    env.close()


def test_trading_logic():
    """Test basic trading logic (position management)."""
    env = gym.make("swm/Financial-v0", max_steps=20)
    obs, info = env.reset(seed=42)

    initial_position = env.unwrapped.position
    initial_shares = env.unwrapped.shares_held

    assert initial_position == 0.0
    assert initial_shares == 0.0

    # Execute a BUY action
    obs, reward, terminated, truncated, info = env.step(1)

    # Position and shares should change (depending on market conditions)
    assert isinstance(env.unwrapped.position, int | float | np.floating)
    assert env.unwrapped.portfolio_value > 0

    env.close()


# ===== COMPREHENSIVE BACKTEST RESULTS TESTS =====


def test_backtest_results_basic_metrics():
    """Test basic return metrics calculation."""
    env = gym.make("swm/Financial-v0", max_steps=20)
    obs, info = env.reset(seed=42)

    # Run a few steps to generate history
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    results = env.unwrapped.get_backtest_results()

    # Check basic metrics exist
    assert "total_return" in results
    assert "annualized_return" in results
    assert "cumulative_return" in results
    assert "benchmark_return" in results
    assert "excess_return" in results

    # Check they are numeric
    assert isinstance(results["total_return"], int | float | np.floating)
    assert isinstance(results["annualized_return"], int | float | np.floating)

    env.close()


def test_backtest_results_risk_metrics():
    """Test risk metrics calculation (Sharpe, Sortino, Calmar, volatility)."""
    env = gym.make("swm/Financial-v0", max_steps=30)
    obs, info = env.reset(seed=42)

    # Generate sufficient trading history
    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    results = env.unwrapped.get_backtest_results()

    # Check risk metrics
    assert "sharpe_ratio" in results
    assert "sortino_ratio" in results
    assert "calmar_ratio" in results
    assert "volatility" in results
    assert "downside_volatility" in results
    assert "alpha" in results
    assert "beta" in results
    assert "information_ratio" in results
    assert "tracking_error" in results
    assert "stability" in results

    # All should be numeric
    assert isinstance(results["sharpe_ratio"], int | float)
    assert isinstance(results["volatility"], int | float)

    env.close()


def test_backtest_results_drawdown_metrics():
    """Test drawdown analysis metrics."""
    env = gym.make("swm/Financial-v0", max_steps=30)
    obs, info = env.reset(seed=42)

    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    results = env.unwrapped.get_backtest_results()

    # Check drawdown metrics
    assert "max_drawdown" in results
    assert "max_drawdown_pct" in results
    assert "max_drawdown_duration" in results
    assert "current_drawdown" in results
    assert "avg_drawdown" in results
    assert "avg_drawdown_duration" in results
    assert "num_drawdown_periods" in results

    # Drawdowns should be non-negative
    assert results["max_drawdown"] >= 0
    assert results["current_drawdown"] >= 0

    env.close()


def test_backtest_results_distribution_metrics():
    """Test return distribution metrics (skew, kurtosis, VaR, tail ratio)."""
    env = gym.make("swm/Financial-v0", max_steps=30)
    obs, info = env.reset(seed=42)

    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    results = env.unwrapped.get_backtest_results()

    # Check distribution metrics
    assert "skewness" in results
    assert "kurtosis" in results
    assert "var_95" in results
    assert "var_99" in results
    assert "cvar_95" in results
    assert "cvar_99" in results
    assert "tail_ratio" in results
    assert "omega_ratio" in results

    # All should be numeric
    assert isinstance(results["skewness"], int | float | np.floating)
    assert isinstance(results["tail_ratio"], int | float | np.floating)

    env.close()


def test_backtest_results_trading_stats():
    """Test trading statistics (trades, win rate, profit factor)."""
    env = gym.make("swm/Financial-v0", max_steps=30)
    obs, info = env.reset(seed=42)

    # Execute varied trades
    for i in range(20):
        action = 1 if i % 3 == 0 else (0 if i % 3 == 1 else 2)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    results = env.unwrapped.get_backtest_results()

    # Check trading stats
    assert "total_trades" in results
    assert "buy_trades" in results
    assert "sell_trades" in results
    assert "short_trades" in results
    assert "winning_trades" in results
    assert "losing_trades" in results
    assert "win_rate" in results
    assert "profit_factor" in results
    assert "avg_trade_return" in results
    assert "trading_frequency" in results

    # Trade counts should be non-negative integers
    assert results["total_trades"] >= 0
    assert results["buy_trades"] >= 0

    env.close()


def test_backtest_results_position_stats():
    """Test position analysis statistics (leverage, exposure, turnover)."""
    env = gym.make("swm/Financial-v0", max_steps=30)
    obs, info = env.reset(seed=42)

    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    results = env.unwrapped.get_backtest_results()

    # Check position stats
    assert "avg_position_size" in results
    assert "max_position_size" in results
    assert "avg_leverage" in results
    assert "max_leverage" in results
    assert "avg_exposure" in results
    assert "max_exposure" in results
    assert "total_turnover" in results

    # All should be numeric and non-negative
    assert results["avg_position_size"] >= 0
    assert results["max_leverage"] >= 0

    env.close()


def test_backtest_results_empty_history():
    """Test backtest results with no trading history."""
    env = gym.make("swm/Financial-v0")
    obs, info = env.reset(seed=42)

    # Get results immediately without any steps
    results = env.unwrapped.get_backtest_results()

    # Should return empty dict or safe defaults
    assert isinstance(results, dict)

    env.close()


# ===== TEAR SHEET DATA TESTS =====


def test_returns_tear_sheet_data():
    """Test returns tear sheet data generation."""
    env = gym.make("swm/Financial-v0", max_steps=30)
    obs, info = env.reset(seed=42)

    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    tear_sheet_data = env.unwrapped.get_returns_tear_sheet_data()

    assert isinstance(tear_sheet_data, dict)
    assert "returns" in tear_sheet_data
    assert "cumulative_returns" in tear_sheet_data
    assert "rolling_returns" in tear_sheet_data
    assert "rolling_volatility" in tear_sheet_data
    assert "rolling_sharpe" in tear_sheet_data
    assert "rolling_beta" in tear_sheet_data
    assert "underwater" in tear_sheet_data
    assert "return_quantiles" in tear_sheet_data
    assert "timestamps" in tear_sheet_data

    env.close()


def test_position_tear_sheet_data():
    """Test position tear sheet data generation."""
    env = gym.make("swm/Financial-v0", max_steps=30)
    obs, info = env.reset(seed=42)

    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    tear_sheet_data = env.unwrapped.get_position_tear_sheet_data()

    assert isinstance(tear_sheet_data, dict)
    assert "gross_exposure" in tear_sheet_data
    assert "net_exposure" in tear_sheet_data
    assert "long_exposure" in tear_sheet_data
    assert "short_exposure" in tear_sheet_data
    assert "gross_leverage" in tear_sheet_data
    assert "top_holdings" in tear_sheet_data
    assert "position_concentration" in tear_sheet_data
    assert "timestamps" in tear_sheet_data

    env.close()


def test_transaction_tear_sheet_data():
    """Test transaction tear sheet data generation."""
    env = gym.make("swm/Financial-v0", max_steps=30)
    obs, info = env.reset(seed=42)

    # Execute multiple transactions
    for i in range(20):
        action = 1 if i % 2 == 0 else 0  # Alternate buy/sell
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    tear_sheet_data = env.unwrapped.get_transaction_tear_sheet_data()

    assert isinstance(tear_sheet_data, dict)
    if tear_sheet_data:  # Only if transactions occurred
        assert "transaction_count" in tear_sheet_data
        assert "transaction_volumes" in tear_sheet_data
        assert "turnovers" in tear_sheet_data
        assert "hour_distribution" in tear_sheet_data
        assert "transaction_types" in tear_sheet_data

    env.close()


def test_round_trip_tear_sheet_data():
    """Test round-trip trade analysis data generation."""
    env = gym.make("swm/Financial-v0", max_steps=40)
    obs, info = env.reset(seed=42)

    # Execute complete round trips (buy then sell)
    for i in range(30):
        if i % 4 < 2:
            action = 1  # Buy
        else:
            action = 0  # Sell (complete round trip)

        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    tear_sheet_data = env.unwrapped.get_round_trip_tear_sheet_data()

    assert isinstance(tear_sheet_data, dict)
    if tear_sheet_data:  # Only if round trips completed
        assert "total_round_trips" in tear_sheet_data
        assert "winning_trips" in tear_sheet_data
        assert "losing_trips" in tear_sheet_data
        assert "win_rate" in tear_sheet_data
        assert "avg_pnl" in tear_sheet_data
        assert "avg_duration_minutes" in tear_sheet_data

    env.close()


def test_capacity_analysis():
    """Test capacity analysis for liquidity constraints."""
    env = gym.make("swm/Financial-v0", max_steps=30)
    obs, info = env.reset(seed=42)

    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    capacity_data = env.unwrapped.get_capacity_analysis()

    assert isinstance(capacity_data, dict)
    if capacity_data:  # Only if data available
        assert "avg_daily_volume" in capacity_data
        assert "max_position_shares" in capacity_data
        assert "capacity_multiplier" in capacity_data
        assert "liquidity_constrained" in capacity_data

    env.close()


# ===== TIME MACHINE FUNCTIONALITY TESTS =====


def test_time_machine_reset_to_date():
    """Test time machine functionality for backtesting at specific dates."""
    import pandas as pd

    # The autouse fixture provides synthetic data automatically
    env = gym.make("swm/Financial-v0")
    unwrapped_env = env.unwrapped

    # Use a recent date for testing
    target_date = (pd.Timestamp.now() - pd.Timedelta(days=4)).strftime("%Y-%m-%d")

    obs, info = unwrapped_env.reset_to_date(target_date, symbol="AAPL")

    assert obs is not None
    assert "start_timestamp" in info
    assert "time_machine_date" in info
    assert info["time_machine_date"] == target_date

    # Verify we got a valid timestamp close to the target
    start_ts = pd.Timestamp(info["start_timestamp"])
    target_ts = pd.Timestamp(target_date)
    days_diff = abs((start_ts - target_ts).days)
    assert days_diff <= 5, f"Start date {start_ts} is too far from target {target_ts}"

    env.close()


# ===== RENDER AND VISUALIZATION TESTS =====


def test_render_returns_dummy_array():
    """Test that render returns a dummy array for wrapper compatibility."""
    env = gym.make("swm/Financial-v0", max_steps=10)
    obs, info = env.reset(seed=42)

    # Take a few actions
    for _ in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    # Render returns a dummy array for AddPixelsWrapper compatibility
    output = env.render()
    assert output is not None
    assert isinstance(output, np.ndarray)
    assert output.shape == (64, 64, 3)
    assert output.dtype == np.uint8

    env.close()


# ===== INTEGRATION TESTS =====


def test_full_episode_workflow():
    """Test a complete episode workflow with all features."""
    env = gym.make("swm/Financial-v0", max_steps=50)
    obs, info = env.reset(seed=42)

    episode_returns = []
    steps = 0

    while steps < 40:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        episode_returns.append(reward)
        steps += 1

        if terminated or truncated:
            break

    # Get comprehensive results
    results = env.unwrapped.get_backtest_results()

    # Verify complete results
    assert len(results) > 20  # Should have many metrics
    assert "total_return" in results
    assert "sharpe_ratio" in results
    assert "max_drawdown" in results
    assert "total_trades" in results

    # Get all tear sheet data
    returns_data = env.unwrapped.get_returns_tear_sheet_data()
    position_data = env.unwrapped.get_position_tear_sheet_data()
    transaction_data = env.unwrapped.get_transaction_tear_sheet_data()

    assert isinstance(returns_data, dict)
    assert isinstance(position_data, dict)
    assert isinstance(transaction_data, dict)

    env.close()


# ===== UNIT TESTS FOR METRICS CALCULATIONS =====


def test_sharpe_ratio_calculation():
    """Test Sharpe ratio calculation with known values."""
    from stable_worldmodel.envs.financial_trading import calculate_sharpe_ratio

    # Test data
    np.random.seed(42)
    returns = np.array([0.001, 0.002, -0.001, 0.0015, 0.003, -0.002, 0.0025])

    # Calculate using our function
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.02, periods_per_year=252 * 390)

    # Manual calculation to verify
    risk_free_period = 0.02 / (252 * 390)
    excess_returns = returns - risk_free_period
    expected_sharpe = np.mean(excess_returns) / np.std(excess_returns, ddof=1) * np.sqrt(252 * 390)

    assert abs(sharpe - expected_sharpe) < 1e-10
    assert not np.isnan(sharpe)
    assert isinstance(sharpe, int | float)

    # Test edge case: constant returns -> should return 0
    constant_returns = np.array([0.01, 0.01, 0.01, 0.01])
    sharpe_constant = calculate_sharpe_ratio(constant_returns)
    assert sharpe_constant == 0.0

    # Test edge case: single return -> should return 0
    single_return = np.array([0.01])
    sharpe_single = calculate_sharpe_ratio(single_return)
    assert sharpe_single == 0.0


def test_sortino_ratio_calculation():
    """Test Sortino ratio calculation with known values."""
    from stable_worldmodel.envs.financial_trading import calculate_sortino_ratio

    np.random.seed(42)
    returns = np.array([0.02, 0.01, -0.01, 0.015, -0.005, 0.03, -0.02])

    sortino = calculate_sortino_ratio(returns, risk_free_rate=0.02)

    assert not np.isnan(sortino)
    assert isinstance(sortino, int | float)

    # Edge case: all positive returns -> should return 0 (no downside vol)
    positive_returns = np.array([0.01, 0.02, 0.015, 0.03])
    sortino_positive = calculate_sortino_ratio(positive_returns)
    assert sortino_positive == 0.0


def test_max_drawdown_calculation():
    """Test maximum drawdown calculation."""
    from stable_worldmodel.envs.financial_trading import calculate_max_drawdown

    portfolio_values = [100, 110, 105, 95, 90, 100, 120, 115]

    max_dd = calculate_max_drawdown(portfolio_values)

    # The max drawdown should be from peak (110) to trough (90)  # codespell:ignore
    # (90 - 110) / 110 = -0.1818...
    assert max_dd > 0
    assert abs(max_dd - 0.1818) < 0.01  # approximately 18.18%


def test_calmar_ratio_calculation():
    """Test Calmar ratio (return / max drawdown)."""
    from stable_worldmodel.envs.financial_trading import calculate_calmar_ratio

    annualized_return = 0.15  # 15% annual return
    max_drawdown = 0.10  # 10% max drawdown

    calmar = calculate_calmar_ratio(annualized_return, max_drawdown)
    assert abs(calmar - 1.5) < 1e-10

    # Edge case: zero drawdown -> should return 0
    calmar_zero_dd = calculate_calmar_ratio(0.15, 0.0)
    assert calmar_zero_dd == 0.0


def test_alpha_beta_calculation():
    """Test alpha and beta calculations."""
    from stable_worldmodel.envs.financial_trading import calculate_alpha_beta

    np.random.seed(42)

    benchmark_returns = np.array([0.01, 0.02, -0.01, 0.015, -0.005])
    # Portfolio returns with some alpha
    portfolio_returns = benchmark_returns * 1.2 + 0.002  # Beta=1.2, alpha=0.2%

    # Calculate annualized returns (simplified for test)
    portfolio_annualized = np.mean(portfolio_returns) * 252 * 390
    benchmark_annualized = np.mean(benchmark_returns) * 252 * 390

    alpha, beta = calculate_alpha_beta(
        portfolio_returns,
        benchmark_returns,
        portfolio_annualized,
        benchmark_annualized,
    )

    assert not np.isnan(beta)
    assert isinstance(beta, int | float)
    # Beta should be close to 1.2
    assert 1.0 < beta < 1.5


def test_volatility_calculation():
    """Test volatility (standard deviation of returns)."""
    from stable_worldmodel.envs.financial_trading import calculate_volatility

    np.random.seed(42)
    returns = np.array([0.01, 0.02, -0.01, 0.015, -0.005, 0.03, -0.02])

    volatility = calculate_volatility(returns)

    # Manual calculation
    expected_volatility = np.std(returns, ddof=1) * np.sqrt(252 * 390)

    assert abs(volatility - expected_volatility) < 1e-10
    assert volatility > 0
    assert not np.isnan(volatility)


def test_information_ratio_calculation():
    """Test information ratio (excess return / tracking error)."""
    from stable_worldmodel.envs.financial_trading import calculate_information_ratio

    np.random.seed(42)

    portfolio_returns = np.array([0.01, 0.02, -0.01, 0.015, -0.005])
    benchmark_returns = np.array([0.008, 0.018, -0.012, 0.014, -0.006])

    # Calculate annualized returns
    portfolio_annualized = np.mean(portfolio_returns) * 252 * 390
    benchmark_annualized = np.mean(benchmark_returns) * 252 * 390

    ir, tracking_error = calculate_information_ratio(
        portfolio_returns,
        benchmark_returns,
        portfolio_annualized,
        benchmark_annualized,
    )

    assert not np.isnan(ir)
    assert not np.isnan(tracking_error)
    assert tracking_error > 0


def test_var_cvar_calculation():
    """Test Value at Risk (VaR) and Conditional VaR calculations."""
    from stable_worldmodel.envs.financial_trading import calculate_var_cvar

    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 1000)

    var_95, cvar_95 = calculate_var_cvar(returns, confidence_level=0.95)

    assert var_95 < 0  # Should be negative (loss)
    assert cvar_95 < var_95  # CVaR should be worse than VaR
    assert not np.isnan(var_95)
    assert not np.isnan(cvar_95)


def test_skewness_kurtosis_calculation():
    """Test skewness and kurtosis calculations."""
    from stable_worldmodel.envs.financial_trading import calculate_skewness_kurtosis

    np.random.seed(42)

    # Normal distribution should have skew~0, kurtosis~0
    normal_returns = np.random.normal(0, 1, 1000)
    skew, kurt = calculate_skewness_kurtosis(normal_returns)

    assert abs(skew) < 0.5  # Should be close to 0
    assert abs(kurt) < 1.0  # Should be close to 0

    # Right-skewed distribution (positive skew)
    right_skewed = np.random.exponential(1, 1000)
    skew_right, _ = calculate_skewness_kurtosis(right_skewed)
    assert skew_right > 0  # Should be positive


def test_tail_ratio_calculation():
    """Test tail ratio (95th percentile / 5th percentile)."""
    from stable_worldmodel.envs.financial_trading import calculate_tail_ratio

    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 1000)

    tail_ratio = calculate_tail_ratio(returns)

    assert tail_ratio > 0
    assert not np.isnan(tail_ratio)


def test_omega_ratio_calculation():
    """Test omega ratio (probability weighted gains vs losses)."""
    from stable_worldmodel.envs.financial_trading import calculate_omega_ratio

    np.random.seed(42)
    returns = np.array([0.02, 0.01, -0.01, 0.015, -0.005, 0.03, -0.02])

    omega = calculate_omega_ratio(returns, threshold=0.0)

    assert omega > 0
    assert not np.isnan(omega)


def test_win_rate_calculation():
    """Test win rate (winning trades / total trades)."""
    from stable_worldmodel.envs.financial_trading import calculate_win_rate

    trade_returns = [0.02, -0.01, 0.015, 0.01, -0.005, 0.03, -0.02, 0.008]

    win_rate = calculate_win_rate(trade_returns)
    expected_win_rate = 5 / 8  # 5 winning out of 8 total

    assert abs(win_rate - expected_win_rate) < 1e-10
    assert 0 <= win_rate <= 1


def test_profit_factor_calculation():
    """Test profit factor (total gains / total losses)."""
    from stable_worldmodel.envs.financial_trading import calculate_profit_factor

    trade_returns = [0.02, -0.01, 0.015, 0.01, -0.005, 0.03, -0.02, 0.008]

    profit_factor = calculate_profit_factor(trade_returns)

    assert profit_factor > 0
    # In this example: (0.02+0.015+0.01+0.03+0.008) / (0.01+0.005+0.02)
    expected = 0.083 / 0.035
    assert abs(profit_factor - expected) < 0.1


def test_metrics_edge_cases():
    """Test edge cases for all metrics."""
    # All zero returns
    zero_returns = np.array([0.0, 0.0, 0.0])
    assert np.std(zero_returns) == 0

    # All identical returns
    identical = np.array([0.01, 0.01, 0.01])
    assert np.std(identical) == 0


def test_annualized_return_calculation():
    """Test annualized return calculation."""
    from stable_worldmodel.envs.financial_trading import calculate_annualized_return

    # Test simple case: 10% total return over 1 year (252*390 periods)
    total_return = 0.10
    periods_per_year = 252 * 390
    total_periods = periods_per_year  # 1 year

    annualized = calculate_annualized_return(total_return, total_periods, periods_per_year)
    assert abs(annualized - 0.10) < 1e-10  # Should be same as total return

    # Test 6 months: 10% return in 6 months -> ~21% annualized
    total_periods_half = periods_per_year // 2
    annualized_half = calculate_annualized_return(total_return, total_periods_half, periods_per_year)
    expected = (1.10) ** 2 - 1  # ~21%
    assert abs(annualized_half - expected) < 1e-10


def test_cumulative_returns_calculation():
    """Test cumulative returns calculation."""
    import pandas as pd

    # Simple period returns
    returns = [0.01, 0.02, -0.01, 0.015]

    # Cumulative: (1+r1)*(1+r2)*(1+r3)*(1+r4) - 1
    cumulative = pd.Series(returns).add(1).cumprod().sub(1)

    # Manual calculation
    expected = (1.01 * 1.02 * 0.99 * 1.015) - 1
    assert abs(cumulative.iloc[-1] - expected) < 1e-10


def test_downside_volatility_calculation():
    """Test downside volatility (only considers negative returns)."""
    from stable_worldmodel.envs.financial_trading import (
        calculate_downside_volatility,
    )

    returns = np.array([0.02, 0.01, -0.01, 0.015, -0.005, 0.03, -0.02])

    downside_vol = calculate_downside_volatility(returns)

    # Manual calculation
    downside_returns = returns[returns < 0]
    expected = np.std(downside_returns, ddof=1) * np.sqrt(252 * 390)

    assert abs(downside_vol - expected) < 1e-10
    assert downside_vol > 0
    assert not np.isnan(downside_vol)


def test_stability_calculation():
    """Test stability (R-squared of returns vs time)."""
    from stable_worldmodel.envs.financial_trading import calculate_stability

    np.random.seed(42)
    # Create upward trending returns
    returns = np.array([0.001 + 0.0001 * i for i in range(100)])

    stability = calculate_stability(returns)

    # Should be close to 1 for consistent upward trend
    assert stability > 0.5
    assert stability <= 1.0
    assert not np.isnan(stability)


# ===== FINANCIAL DATASET INTEGRATION TEST =====


# ===== COMPREHENSIVE EDGE CASE TESTS FOR UTILITY FUNCTIONS =====


def test_sortino_ratio_edge_cases():
    """Test Sortino ratio edge cases for lines 59, 65, 70."""
    from stable_worldmodel.envs.financial_trading import calculate_sortino_ratio

    # Edge case: single element (line 59)
    single_return = np.array([0.01])
    assert calculate_sortino_ratio(single_return) == 0.0

    # Edge case: no downside returns (line 65) - all positive
    positive_returns = np.array([0.01, 0.02, 0.03, 0.015])
    assert calculate_sortino_ratio(positive_returns) == 0.0

    # Edge case: all downside returns are identical (line 70)
    identical_negative = np.array([0.01, -0.01, -0.01, -0.01, 0.02])
    sortino = calculate_sortino_ratio(identical_negative)
    assert sortino == 0.0


def test_max_drawdown_edge_cases():
    """Test max drawdown edge cases for line 79."""
    from stable_worldmodel.envs.financial_trading import calculate_max_drawdown

    # Edge case: single value (line 79)
    single_value = [100.0]
    assert calculate_max_drawdown(single_value) == 0.0

    # Edge case: monotonically increasing (no drawdown)
    increasing = [100, 110, 120, 130]
    dd = calculate_max_drawdown(increasing)
    assert dd >= 0.0


def test_calmar_ratio_edge_cases():
    """Test Calmar ratio edge case for line 93."""
    from stable_worldmodel.envs.financial_trading import calculate_calmar_ratio

    # Edge case: zero max_drawdown (line 93)
    assert calculate_calmar_ratio(0.15, 0.0) == 0.0
    assert calculate_calmar_ratio(0.0, 0.0) == 0.0


def test_alpha_beta_edge_cases():
    """Test alpha and beta edge cases for lines 103, 107."""
    from stable_worldmodel.envs.financial_trading import calculate_alpha_beta

    # Edge case: single return (line 103)
    single_ret = np.array([0.01])
    single_bench = np.array([0.01])
    alpha, beta = calculate_alpha_beta(single_ret, single_bench, 0.1, 0.1)
    assert alpha == 0.0
    assert beta == 0.0

    # Edge case: mismatched lengths (line 103)
    short_ret = np.array([0.01, 0.02])
    long_bench = np.array([0.01, 0.02, 0.03])
    alpha, beta = calculate_alpha_beta(short_ret, long_bench, 0.1, 0.1)
    assert alpha == 0.0
    assert beta == 0.0

    # Edge case: zero benchmark variance (line 107)
    returns = np.array([0.01, 0.02, -0.01, 0.015])
    benchmark_constant = np.array([0.01, 0.01, 0.01, 0.01])
    alpha, beta = calculate_alpha_beta(returns, benchmark_constant, 0.1, 0.1)
    assert alpha == 0.0
    assert beta == 0.0


def test_volatility_edge_cases():
    """Test volatility edge case for line 118."""
    from stable_worldmodel.envs.financial_trading import calculate_volatility

    # Edge case: single return (line 118)
    single_return = np.array([0.01])
    assert calculate_volatility(single_return) == 0.0


def test_downside_volatility_edge_cases():
    """Test downside volatility edge case for line 127."""
    from stable_worldmodel.envs.financial_trading import (
        calculate_downside_volatility,
    )

    # Edge case: no downside returns (line 127)
    positive_returns = np.array([0.01, 0.02, 0.03, 0.015])
    assert calculate_downside_volatility(positive_returns) == 0.0

    # Edge case: single downside return (line 127)
    one_negative = np.array([0.01, 0.02, -0.01])
    assert calculate_downside_volatility(one_negative) == 0.0


def test_information_ratio_edge_cases():
    """Test information ratio edge cases for lines 141, 147."""
    from stable_worldmodel.envs.financial_trading import calculate_information_ratio

    # Edge case: mismatched lengths (line 141)
    short_ret = np.array([0.01, 0.02])
    long_bench = np.array([0.01, 0.02, 0.03])
    ir, tracking_error = calculate_information_ratio(short_ret, long_bench, 0.1, 0.1)
    assert ir == 0.0
    assert tracking_error == 0.0

    # Edge case: single return (line 141)
    single_ret = np.array([0.01])
    single_bench = np.array([0.01])
    ir, tracking_error = calculate_information_ratio(single_ret, single_bench, 0.1, 0.1)
    assert ir == 0.0
    assert tracking_error == 0.0

    # Edge case: identical returns -> zero tracking error (line 147)
    identical_ret = np.array([0.01, 0.02, 0.015, 0.018])
    identical_bench = identical_ret.copy()
    ir, tracking_error = calculate_information_ratio(identical_ret, identical_bench, 0.1, 0.1)
    assert ir == 0.0
    assert tracking_error == 0.0


def test_var_cvar_edge_cases():
    """Test VaR/CVaR edge case for line 156."""
    from stable_worldmodel.envs.financial_trading import calculate_var_cvar

    # Edge case: fewer than 10 returns (line 156)
    few_returns = np.array([0.01, -0.01, 0.02, -0.005])
    var, cvar = calculate_var_cvar(few_returns)
    assert var == 0.0
    assert cvar == 0.0


def test_skewness_kurtosis_edge_cases():
    """Test skewness/kurtosis edge case for line 168."""
    from stable_worldmodel.envs.financial_trading import calculate_skewness_kurtosis

    # Edge case: fewer than 4 returns (line 168)
    few_returns = np.array([0.01, 0.02])
    skew, kurt = calculate_skewness_kurtosis(few_returns)
    assert skew == 0.0
    assert kurt == 0.0


def test_tail_ratio_edge_cases():
    """Test tail ratio edge cases for lines 179, 185."""
    from stable_worldmodel.envs.financial_trading import calculate_tail_ratio

    # Edge case: fewer than 10 returns (line 179)
    few_returns = np.array([0.01, -0.01, 0.02])
    assert calculate_tail_ratio(few_returns) == 0.0

    # Edge case: p5 is zero (line 185)
    # Create returns where 5th percentile is exactly 0
    returns_with_zero_p5 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05])
    tr = calculate_tail_ratio(returns_with_zero_p5)
    assert tr == 0.0


def test_omega_ratio_edge_cases():
    """Test Omega ratio edge cases for lines 193, 199."""
    from stable_worldmodel.envs.financial_trading import calculate_omega_ratio

    # Edge case: single return (line 193)
    single_return = np.array([0.01])
    assert calculate_omega_ratio(single_return) == 0.0

    # Edge case: no losses (line 199) - all gains
    all_positive = np.array([0.01, 0.02, 0.03, 0.015])
    omega = calculate_omega_ratio(all_positive, threshold=0.0)
    assert omega == 0.0


def test_win_rate_edge_cases():
    """Test win rate edge case for line 207."""
    from stable_worldmodel.envs.financial_trading import calculate_win_rate

    # Edge case: empty list (line 207)
    empty_trades = []
    assert calculate_win_rate(empty_trades) == 0.0


def test_profit_factor_edge_cases():
    """Test profit factor edge cases for lines 216, 222."""
    from stable_worldmodel.envs.financial_trading import calculate_profit_factor

    # Edge case: empty list (line 216)
    empty_trades = []
    assert calculate_profit_factor(empty_trades) == 0.0

    # Edge case: no losses (line 222) - all winning trades
    all_wins = [0.01, 0.02, 0.03, 0.015]
    pf = calculate_profit_factor(all_wins)
    assert pf == 0.0


def test_annualized_return_edge_cases():
    """Test annualized return edge cases for lines 230, 234."""
    from stable_worldmodel.envs.financial_trading import calculate_annualized_return

    # Edge case: zero periods (line 230)
    assert calculate_annualized_return(0.10, 0) == 0.0

    # Edge case: very few periods (line 234)
    # When years < 1/periods_per_year, it gets clamped
    periods_per_year = 252 * 390
    total_return = 0.001  # Small return to avoid overflow
    very_few_periods = 10  # Less than 1 trading period
    ar = calculate_annualized_return(total_return, very_few_periods, periods_per_year)
    # Should use clamped years value and not raise overflow
    assert isinstance(ar, float)


def test_stability_edge_cases():
    """Test stability edge case for line 242."""
    from stable_worldmodel.envs.financial_trading import calculate_stability

    # Edge case: fewer than 3 returns (line 242)
    few_returns = np.array([0.01, 0.02])
    assert calculate_stability(few_returns) == 0.0


# ===== ENVIRONMENT EDGE CASE TESTS =====


def test_get_available_symbols():
    """Test get_available_symbols for line 382."""
    env = gym.make("swm/Financial-v0")
    unwrapped_env = env.unwrapped

    symbols = unwrapped_env.get_available_symbols()
    assert isinstance(symbols, list)
    assert len(symbols) > 0  # Should have S&P 500 symbols
    assert "AAPL" in symbols

    env.close()


def test_get_date_range_info():
    """Test get_date_range_info for line 393."""
    env = gym.make("swm/Financial-v0")
    unwrapped_env = env.unwrapped

    info = unwrapped_env.get_date_range_info("AAPL")
    assert isinstance(info, dict)
    assert "symbol" in info
    assert "earliest_date" in info
    assert "latest_date" in info
    assert "frequency" in info
    assert info["symbol"] == "AAPL"

    env.close()


def test_step_end_of_data():
    """Test step at end of market data for lines 503-506."""
    env = gym.make("swm/Financial-v0", max_steps=5)
    obs, info = env.reset(seed=42)

    # Force current_data_index to end of data
    unwrapped_env = env.unwrapped
    unwrapped_env.current_data_index = len(unwrapped_env.market_data) - 1

    # Step should detect end of data (lines 503-506)
    obs, reward, terminated, truncated, info = env.step(1)

    assert terminated is True
    assert truncated is False
    assert reward == 0.0
    assert "reason" in info
    assert info["reason"] == "end_of_data"

    env.close()


def test_execute_buy_order_no_money():
    """Test buy order with no balance for line 587."""
    env = gym.make("swm/Financial-v0", max_steps=10)
    obs, info = env.reset(seed=42)

    unwrapped_env = env.unwrapped

    # Force balance to zero
    unwrapped_env.balance = 0.0

    # Execute buy order - should return 'hold' (line 587)
    result = unwrapped_env._execute_buy_order(100.0, 10.0, 1.0)
    assert result == "hold"

    env.close()


def test_execute_buy_order_close_short():
    """Test buy order when short position exists for line 611."""
    env = gym.make("swm/Financial-v0", max_steps=10)
    obs, info = env.reset(seed=42)

    unwrapped_env = env.unwrapped
    unwrapped_env.enable_shorting = True

    # Create short position
    unwrapped_env.shares_held = -10.0
    unwrapped_env.position = -10.0

    # Buy order should close short position (line 611)
    result = unwrapped_env._execute_buy_order(100.0, 10.0, 1.0)
    # Should attempt to close short
    assert result in ["cover_short", "hold"]

    env.close()


def test_execute_buy_order_insufficient_funds():
    """Test buy order with insufficient funds for line 635, 640."""
    env = gym.make("swm/Financial-v0", max_steps=10)
    obs, info = env.reset(seed=42)

    unwrapped_env = env.unwrapped

    # Force zero balance to trigger line 640 (return 'hold')
    unwrapped_env.balance = 0.0

    # Try to buy - should return 'hold' because shares_to_buy will be 0 (line 640)
    result = unwrapped_env._execute_buy_order(100.0, 10.0, 1.0)
    assert result == "hold"

    env.close()


def test_execute_sell_order_no_shares():
    """Test sell order with no shares and shorting disabled for line 640."""
    env = gym.make("swm/Financial-v0", max_steps=10)
    obs, info = env.reset(seed=42)

    unwrapped_env = env.unwrapped
    unwrapped_env.enable_shorting = False
    unwrapped_env.shares_held = 0.0

    # Try to sell with no shares (line 640)
    result = unwrapped_env._execute_sell_order(100.0, 10.0, 1.0)
    assert result == "hold"

    env.close()


def test_execute_sell_order_enter_short():
    """Test sell order entering short position for line 674."""
    env = gym.make("swm/Financial-v0", max_steps=10)
    obs, info = env.reset(seed=42)

    unwrapped_env = env.unwrapped
    unwrapped_env.enable_shorting = True
    unwrapped_env.shares_held = 0.0
    unwrapped_env.balance = 100000.0

    # Sell to enter short position (line 674)
    result = unwrapped_env._execute_sell_order(100.0, 10.0, 1.0)
    # Should enter short position
    assert result == "short"
    assert unwrapped_env.shares_held < 0

    env.close()


def test_close_short_position_success():
    """Test closing short position for line 692."""
    env = gym.make("swm/Financial-v0", max_steps=10)
    obs, info = env.reset(seed=42)

    unwrapped_env = env.unwrapped
    unwrapped_env.enable_shorting = True

    # Create short position
    unwrapped_env.shares_held = -10.0
    unwrapped_env.position = -10.0
    unwrapped_env.balance = 100000.0

    # Close short position (line 692)
    result = unwrapped_env._close_short_position(100.0, 10.0, 1.0)
    assert result == "cover_short"
    assert unwrapped_env.shares_held == 0.0

    env.close()


def test_close_short_position_no_short():
    """Test close short when no short position exists for line 730."""
    env = gym.make("swm/Financial-v0", max_steps=10)
    obs, info = env.reset(seed=42)

    unwrapped_env = env.unwrapped
    unwrapped_env.shares_held = 10.0  # Long position, not short

    # Try to close short (line 730)
    result = unwrapped_env._close_short_position(100.0, 10.0, 1.0)
    assert result == "hold"

    env.close()


def test_close_short_position_insufficient_funds():
    """Test close short with insufficient funds for line 731."""
    env = gym.make("swm/Financial-v0", max_steps=10)
    obs, info = env.reset(seed=42)

    unwrapped_env = env.unwrapped
    unwrapped_env.enable_shorting = True

    # Create large short position with insufficient balance
    unwrapped_env.shares_held = -1000.0
    unwrapped_env.position = -1000.0
    unwrapped_env.balance = 10.0  # Not enough to cover

    # Try to close short (line 731)
    result = unwrapped_env._close_short_position(100.0, 10.0, 1.0)
    # Should return 'hold' because balance < total_cost
    assert result == "hold"

    env.close()


def test_calculate_benchmark_return_insufficient_history():
    """Test benchmark return with insufficient history for line 759."""
    env = gym.make("swm/Financial-v0", max_steps=10)
    obs, info = env.reset(seed=42)

    unwrapped_env = env.unwrapped

    # Clear price history
    unwrapped_env.price_history = []

    # Should return 0.0 (line 759)
    benchmark_return = unwrapped_env._calculate_benchmark_return()
    assert benchmark_return == 0.0

    # Test with single price
    unwrapped_env.price_history = [{"price": 100.0}]
    benchmark_return = unwrapped_env._calculate_benchmark_return()
    assert benchmark_return == 0.0

    env.close()


def test_calculate_reward_insufficient_history():
    """Test reward calculation with insufficient history for lines 790-793."""
    env = gym.make("swm/Financial-v0", max_steps=10)
    obs, info = env.reset(seed=42)

    unwrapped_env = env.unwrapped

    # Clear portfolio returns
    unwrapped_env.portfolio_returns = []

    # Calculate reward with no history (line 790-793)
    reward = unwrapped_env._calculate_reward(0.01, 0.005)
    # volatility_penalty should be 0.0 when len < 10
    assert isinstance(reward, float)

    # Add some returns but less than 10
    unwrapped_env.portfolio_returns = [0.01, -0.005, 0.02]
    reward = unwrapped_env._calculate_reward(0.01, 0.005)
    # volatility_penalty should still be 0.0
    assert isinstance(reward, float)

    env.close()


def test_calculate_max_drawdown_no_history():
    """Test max drawdown with no trade history for line 968."""
    env = gym.make("swm/Financial-v0", max_steps=10)
    obs, info = env.reset(seed=42)

    unwrapped_env = env.unwrapped

    # Clear trade history
    unwrapped_env.trade_history = []

    # Should return 0.0 (line 968)
    max_dd = unwrapped_env._calculate_max_drawdown()
    assert max_dd == 0.0

    env.close()


def test_analyze_drawdowns_insufficient_data():
    """Test drawdown analysis with insufficient data for line 985."""
    env = gym.make("swm/Financial-v0", max_steps=10)
    obs, info = env.reset(seed=42)

    unwrapped_env = env.unwrapped

    # Test with single value (line 985)
    portfolio_values = [100.0]
    timestamps = [unwrapped_env.market_data.index[0]]

    drawdown_analysis = unwrapped_env._analyze_drawdowns(portfolio_values, timestamps)

    assert drawdown_analysis["max_drawdown_duration"] == 0
    assert drawdown_analysis["current_drawdown"] == 0.0
    assert drawdown_analysis["num_drawdown_periods"] == 0

    env.close()


def test_analyze_trades_no_history():
    """Test trade analysis with no history for line 1059."""
    env = gym.make("swm/Financial-v0", max_steps=10)
    obs, info = env.reset(seed=42)

    unwrapped_env = env.unwrapped

    # Clear trade history
    unwrapped_env.trade_history = []

    # Should return defaults (line 1059)
    trade_analysis = unwrapped_env._analyze_trades()

    assert trade_analysis["total_trades"] == 0
    assert trade_analysis["win_rate"] == 0.0
    assert trade_analysis["profit_factor"] == 0.0

    env.close()


def test_analyze_trades_edge_cases():
    """Test trade analysis edge cases for lines 1111-1117."""
    env = gym.make("swm/Financial-v0", max_steps=10)
    obs, info = env.reset(seed=42)

    unwrapped_env = env.unwrapped

    # Create trade history without trade returns (lines 1111-1117)
    unwrapped_env.trade_history = [
        {
            "action": "buy",
            "portfolio_value": 100000,
            "timestamp": unwrapped_env.market_data.index[0],
        }
    ]

    trade_analysis = unwrapped_env._analyze_trades()

    # With only 1 trade, should have defaults for return metrics
    assert trade_analysis["total_trades"] == 1
    assert trade_analysis["win_rate"] == 0.0
    assert trade_analysis["profit_factor"] == 0.0

    env.close()


def test_analyze_trades_trading_frequency():
    """Test trading frequency calculation for line 1126."""
    env = gym.make("swm/Financial-v0", max_steps=10)
    obs, info = env.reset(seed=42)

    unwrapped_env = env.unwrapped

    # Test with empty trade history (line 1126)
    unwrapped_env.trade_history = []
    trade_analysis = unwrapped_env._analyze_trades()
    assert trade_analysis["trading_frequency"] == 0.0

    env.close()


def test_analyze_positions_no_history():
    """Test position analysis with no history for line 1163."""
    env = gym.make("swm/Financial-v0", max_steps=10)
    obs, info = env.reset(seed=42)

    unwrapped_env = env.unwrapped

    # Clear trade history
    unwrapped_env.trade_history = []

    # Should return defaults (line 1163)
    position_analysis = unwrapped_env._analyze_positions()

    assert position_analysis["avg_position_size"] == 0.0
    assert position_analysis["max_position_size"] == 0.0
    assert position_analysis["avg_leverage"] == 0.0

    env.close()


def test_get_returns_tear_sheet_empty():
    """Test returns tear sheet with empty data for line 1236."""
    env = gym.make("swm/Financial-v0", max_steps=10)
    obs, info = env.reset(seed=42)

    unwrapped_env = env.unwrapped

    # Clear returns
    unwrapped_env.portfolio_returns = []
    unwrapped_env.trade_history = []

    # Should return empty dict (line 1236)
    tear_sheet = unwrapped_env.get_returns_tear_sheet_data()
    assert tear_sheet == {}

    # Test with single return (line 1236)
    unwrapped_env.portfolio_returns = [0.01]
    tear_sheet = unwrapped_env.get_returns_tear_sheet_data()
    assert tear_sheet == {}

    env.close()


def test_get_returns_tear_sheet_rolling_window():
    """Test returns tear sheet rolling calculations for lines 1260-1265."""
    env = gym.make("swm/Financial-v0", max_steps=10)
    obs, info = env.reset(seed=42)

    unwrapped_env = env.unwrapped

    # Create minimal data to test rolling window logic (lines 1260-1265)
    # With very few returns, rolling_window will be small
    unwrapped_env.portfolio_returns = [0.01, 0.02, -0.01]
    unwrapped_env.benchmark_returns = [0.005, 0.015, -0.008]
    unwrapped_env.trade_history = [{"timestamp": unwrapped_env.market_data.index[i]} for i in range(3)]

    tear_sheet = unwrapped_env.get_returns_tear_sheet_data()

    # Should still generate tear sheet with small rolling window
    assert isinstance(tear_sheet, dict)
    assert "rolling_returns" in tear_sheet
    assert "rolling_volatility" in tear_sheet

    env.close()


def test_get_position_tear_sheet_empty():
    """Test position tear sheet with empty data for line 1313."""
    env = gym.make("swm/Financial-v0", max_steps=10)
    obs, info = env.reset(seed=42)

    unwrapped_env = env.unwrapped

    # Clear trade history
    unwrapped_env.trade_history = []

    # Should return empty dict (line 1313)
    tear_sheet = unwrapped_env.get_position_tear_sheet_data()
    assert tear_sheet == {}

    env.close()


def test_get_transaction_tear_sheet_empty():
    """Test transaction tear sheet with empty data for lines 1364, 1370."""
    env = gym.make("swm/Financial-v0", max_steps=10)
    obs, info = env.reset(seed=42)

    unwrapped_env = env.unwrapped

    # Clear trade history (line 1364)
    unwrapped_env.trade_history = []
    tear_sheet = unwrapped_env.get_transaction_tear_sheet_data()
    assert tear_sheet == {}

    # Create history with only 'hold' actions (line 1370)
    unwrapped_env.trade_history = [
        {
            "action": "hold",
            "shares_held": 0.0,
            "price": 100.0,
            "portfolio_value": 100000,
            "timestamp": unwrapped_env.market_data.index[0],
        }
    ]
    tear_sheet = unwrapped_env.get_transaction_tear_sheet_data()
    assert tear_sheet == {}

    env.close()


def test_get_round_trip_tear_sheet_empty():
    """Test round-trip tear sheet with empty data for lines 1421, 1471."""
    env = gym.make("swm/Financial-v0", max_steps=10)
    obs, info = env.reset(seed=42)

    unwrapped_env = env.unwrapped

    # Clear trade history (line 1421)
    unwrapped_env.trade_history = []
    tear_sheet = unwrapped_env.get_round_trip_tear_sheet_data()
    assert tear_sheet == {}

    # Create history with no completed round trips (line 1471)
    unwrapped_env.trade_history = [
        {
            "action": "buy",
            "shares_held": 10.0,
            "price": 100.0,
            "timestamp": unwrapped_env.market_data.index[0],
        }
    ]
    tear_sheet = unwrapped_env.get_round_trip_tear_sheet_data()
    assert tear_sheet == {}

    env.close()


def test_get_interesting_periods_analysis():
    """Test interesting periods analysis for lines 1509-1538."""
    env = gym.make("swm/Financial-v0", max_steps=20)
    obs, info = env.reset(seed=42)

    # Run a few steps
    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            break

    unwrapped_env = env.unwrapped

    # Test with None periods (line 1509)
    analysis = unwrapped_env.get_interesting_periods_analysis(periods=None)
    assert analysis == {}

    # Test with empty trade history (line 1509)
    unwrapped_env.trade_history = []
    analysis = unwrapped_env.get_interesting_periods_analysis(periods={"Test Period": ("2024-01-01", "2024-01-02")})
    assert analysis == {}

    env.close()


def test_calculate_period_max_drawdown():
    """Test period max drawdown for lines 1542-1549."""
    import pandas as pd

    env = gym.make("swm/Financial-v0", max_steps=10)
    obs, info = env.reset(seed=42)

    unwrapped_env = env.unwrapped

    # Test with single return (line 1542)
    single_return = pd.Series([0.01])
    dd = unwrapped_env._calculate_period_max_drawdown(single_return)
    assert dd == 0.0

    # Test with empty series (line 1549)
    empty_returns = pd.Series([])
    dd = unwrapped_env._calculate_period_max_drawdown(empty_returns)
    assert dd == 0.0

    env.close()


def test_get_capacity_analysis_empty():
    """Test capacity analysis with empty data for line 1567."""
    env = gym.make("swm/Financial-v0", max_steps=10)
    obs, info = env.reset(seed=42)

    unwrapped_env = env.unwrapped

    # Clear histories (line 1567)
    unwrapped_env.trade_history = []
    unwrapped_env.price_history = []

    capacity = unwrapped_env.get_capacity_analysis()
    assert capacity == {}

    env.close()


# ===== ADDITIONAL COVERAGE TESTS FOR REMAINING LINES =====


def test_get_data_format_info():
    """Test get_data_format_info static method for line 393."""
    env = gym.make("swm/Financial-v0")
    unwrapped_env = env.unwrapped

    # Test static method (line 393)
    format_info = unwrapped_env.get_data_format_info()
    assert isinstance(format_info, dict)
    assert "format_class" in format_info
    assert "encoding" in format_info
    assert format_info["encoding"] == "bfloat16"

    env.close()


def test_step_bankruptcy_truncation():
    """Test step with bankruptcy (truncation) for line 587."""
    env = gym.make("swm/Financial-v0", max_steps=50)
    obs, info = env.reset(seed=42)

    unwrapped_env = env.unwrapped

    # Force balance and shares to create bankruptcy scenario
    # After portfolio_value calculation in step, it should be below min_balance
    unwrapped_env.balance = 1000.0  # Very low balance
    unwrapped_env.shares_held = 0.0  # No shares
    # This will result in portfolio_value = balance + shares*price = 1000 < 10000

    # Take a step - should trigger truncation (line 587)
    obs, reward, terminated, truncated, info = env.step(2)  # HOLD action

    # Line 587: should apply penalty and truncate
    assert truncated  # Check for truncation (works with both bool and numpy.bool_)
    # Penalty of -100 should have been applied to reward
    assert reward <= 0

    env.close()


def test_execute_buy_order_actual_purchase():
    """Test successful buy order for line 635."""
    env = gym.make("swm/Financial-v0", max_steps=10)
    obs, info = env.reset(seed=42)

    unwrapped_env = env.unwrapped

    # Set sufficient balance
    unwrapped_env.balance = 100000.0
    unwrapped_env.shares_held = 0.0

    # Execute buy order (line 635 should execute buy)
    result = unwrapped_env._execute_buy_order(100.0, 10.0, 1.0)
    assert result == "buy"
    assert unwrapped_env.shares_held > 0

    env.close()


def test_get_observation_no_market_data():
    """Test _get_observation with no market data for lines 730-731."""
    env = gym.make("swm/Financial-v0", max_steps=10)
    obs, info = env.reset(seed=42)

    unwrapped_env = env.unwrapped

    # Force market_data to None (line 730)
    unwrapped_env.market_data = None

    # Get observation - should return zeros
    obs = unwrapped_env._get_observation()

    expected_size = unwrapped_env.window_size * 6 + 1 + 1 + 1 + 4
    assert obs.shape == (expected_size,)
    assert np.allclose(obs, 0.0)

    env.close()


def test_get_observation_insufficient_data_index():
    """Test _get_observation with insufficient data index for line 730."""
    env = gym.make("swm/Financial-v0", max_steps=10)
    obs, info = env.reset(seed=42)

    unwrapped_env = env.unwrapped

    # Force current_data_index below window_size (line 730)
    unwrapped_env.current_data_index = unwrapped_env.window_size - 5

    # Get observation - should return zeros
    obs = unwrapped_env._get_observation()

    expected_size = unwrapped_env.window_size * 6 + 1 + 1 + 1 + 4
    assert obs.shape == (expected_size,)
    assert np.allclose(obs, 0.0)

    env.close()


def test_calculate_benchmark_return_with_single_price():
    """Test benchmark return calculation with single price for line 759."""
    env = gym.make("swm/Financial-v0", max_steps=10)
    obs, info = env.reset(seed=42)

    unwrapped_env = env.unwrapped

    # Set single price in history (line 759)
    unwrapped_env.price_history = [{"price": 100.0}]

    benchmark_return = unwrapped_env._calculate_benchmark_return()
    assert benchmark_return == 0.0

    env.close()


def test_calculate_reward_with_volatility():
    """Test reward calculation with sufficient history for lines 790-793."""
    env = gym.make("swm/Financial-v0", max_steps=20)
    obs, info = env.reset(seed=42)

    unwrapped_env = env.unwrapped

    # Build up portfolio returns (> 10 to trigger volatility calculation)
    unwrapped_env.portfolio_returns = [
        0.01,
        -0.005,
        0.02,
        -0.01,
        0.015,
        0.008,
        -0.003,
        0.012,
        -0.007,
        0.009,
        0.011,
        -0.004,
    ]  # 12 returns

    # Calculate reward - should include volatility penalty (lines 790-793)
    reward = unwrapped_env._calculate_reward(0.01, 0.005)

    # With volatility, reward should be different than without
    assert isinstance(reward, float)

    env.close()


def test_analyze_drawdowns_with_periods():
    """Test drawdown analysis with actual drawdown periods for lines 1008-1021."""
    env = gym.make("swm/Financial-v0", max_steps=30)
    obs, info = env.reset(seed=42)

    unwrapped_env = env.unwrapped

    # Create portfolio values with clear drawdown pattern
    timestamps = unwrapped_env.market_data.index[:10].tolist()
    portfolio_values = [
        100000,
        110000,
        105000,
        95000,
        90000,
        100000,
        110000,
        105000,
        98000,
        102000,
    ]

    # Analyze drawdowns (lines 1008-1021 for drawdown periods)
    drawdown_analysis = unwrapped_env._analyze_drawdowns(portfolio_values, timestamps)

    assert isinstance(drawdown_analysis, dict)
    assert "num_drawdown_periods" in drawdown_analysis
    assert "max_drawdown_duration" in drawdown_analysis
    # Should detect at least one drawdown period
    assert drawdown_analysis["num_drawdown_periods"] >= 0

    env.close()


def test_analyze_drawdowns_ongoing():
    """Test drawdown analysis with ongoing drawdown for lines 1028-1030."""
    env = gym.make("swm/Financial-v0", max_steps=30)
    obs, info = env.reset(seed=42)

    unwrapped_env = env.unwrapped

    # Create portfolio values ending in drawdown
    timestamps = unwrapped_env.market_data.index[:5].tolist()
    portfolio_values = [100000, 110000, 105000, 95000, 90000]  # Ends in drawdown

    # Analyze drawdowns (lines 1028-1030 for statistics)
    drawdown_analysis = unwrapped_env._analyze_drawdowns(portfolio_values, timestamps)

    assert isinstance(drawdown_analysis, dict)
    assert "avg_drawdown" in drawdown_analysis
    assert "avg_drawdown_duration" in drawdown_analysis

    env.close()


def test_analyze_trades_with_valid_history():
    """Test trade analysis with valid trade history for line 1126."""
    env = gym.make("swm/Financial-v0", max_steps=30)
    obs, info = env.reset(seed=42)

    unwrapped_env = env.unwrapped

    # Create realistic trade history with timestamps
    import pandas as pd

    base_timestamp = unwrapped_env.market_data.index[0]

    unwrapped_env.trade_history = [
        {
            "action": "buy",
            "portfolio_value": 100000,
            "timestamp": base_timestamp,
            "shares_held": 10.0,
            "price": 100.0,
        },
        {
            "action": "sell",
            "portfolio_value": 105000,
            "timestamp": base_timestamp + pd.Timedelta(days=1),
            "shares_held": 0.0,
            "price": 105.0,
        },
        {
            "action": "buy",
            "portfolio_value": 105000,
            "timestamp": base_timestamp + pd.Timedelta(days=2),
            "shares_held": 10.0,
            "price": 105.0,
        },
    ]

    # Analyze trades (line 1126 for trading frequency calculation)
    trade_analysis = unwrapped_env._analyze_trades()

    assert trade_analysis["total_trades"] == 3
    assert "trading_frequency" in trade_analysis
    assert trade_analysis["trading_frequency"] > 0

    env.close()


def test_get_returns_tear_sheet_mismatched_lengths():
    """Test returns tear sheet with mismatched return lengths for line 1260."""
    env = gym.make("swm/Financial-v0", max_steps=20)
    obs, info = env.reset(seed=42)

    unwrapped_env = env.unwrapped

    # Create mismatched return lengths (line 1260 else branch)
    unwrapped_env.portfolio_returns = [0.01, 0.02, -0.01, 0.015, 0.008]
    unwrapped_env.benchmark_returns = [0.005, 0.015]  # Different length
    unwrapped_env.trade_history = [{"timestamp": unwrapped_env.market_data.index[i]} for i in range(5)]

    tear_sheet = unwrapped_env.get_returns_tear_sheet_data()

    # Should still generate tear sheet, using rolling_beta fallback (line 1260)
    assert isinstance(tear_sheet, dict)
    assert "rolling_beta" in tear_sheet

    env.close()


def test_get_interesting_periods_with_valid_periods():
    """Test interesting periods analysis with valid data for lines 1512-1538."""
    env = gym.make("swm/Financial-v0", max_steps=30)
    obs, info = env.reset(seed=42)

    # Run a few steps to generate data
    for _ in range(15):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            break

    unwrapped_env = env.unwrapped

    # Get timestamp range from actual data
    if unwrapped_env.trade_history:
        first_ts = unwrapped_env.trade_history[0]["timestamp"]
        last_ts = unwrapped_env.trade_history[-1]["timestamp"]

        # Create period that matches actual data (lines 1512-1538)
        periods = {"Test Period": (first_ts.strftime("%Y-%m-%d"), last_ts.strftime("%Y-%m-%d"))}

        analysis = unwrapped_env.get_interesting_periods_analysis(periods=periods)

        # Should have analysis for the period (line 1527)
        if "Test Period" in analysis:
            assert "total_return" in analysis["Test Period"]
            assert "sharpe" in analysis["Test Period"]

    env.close()


def test_get_interesting_periods_exception_handling():
    """Test interesting periods analysis exception handling for line 1538."""
    env = gym.make("swm/Financial-v0", max_steps=20)
    obs, info = env.reset(seed=42)

    # Run a few steps
    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            break

    unwrapped_env = env.unwrapped

    # Create period with invalid dates to trigger exception (line 1538)
    periods = {"Invalid Period": ("invalid-date", "2024-01-02")}

    analysis = unwrapped_env.get_interesting_periods_analysis(periods=periods)

    # Should handle exception and return empty for invalid period
    assert isinstance(analysis, dict)
    # Invalid period should not be in results due to exception
    assert "Invalid Period" not in analysis

    env.close()


def test_calculate_period_max_drawdown_valid():
    """Test period max drawdown with valid data for lines 1545-1549."""
    import pandas as pd

    env = gym.make("swm/Financial-v0", max_steps=10)
    obs, info = env.reset(seed=42)

    unwrapped_env = env.unwrapped

    # Test with valid returns series (lines 1545-1549)
    valid_returns = pd.Series([0.01, 0.02, -0.03, 0.015, -0.01])
    dd = unwrapped_env._calculate_period_max_drawdown(valid_returns)

    # Should calculate actual drawdown
    assert dd >= 0.0
    assert isinstance(dd, float)

    env.close()


# ===== TESTS FOR FINAL UNCOVERED LINES =====


def test_annualized_return_years_clamping():
    """Test annualized return with years clamping for line 234."""
    from stable_worldmodel.envs.financial_trading import calculate_annualized_return

    # Test very small periods that trigger years clamping (line 234)
    periods_per_year = 252 * 390
    total_return = 0.001
    very_few_periods = 5  # Much less than 1 period

    # This should clamp years to 1/periods_per_year (line 234)
    ar = calculate_annualized_return(total_return, very_few_periods, periods_per_year)

    # Should not raise an exception and return a finite value
    assert np.isfinite(ar)
    assert isinstance(ar, float)


def test_execute_buy_with_total_cost_check():
    """Test buy order total cost check for line 635."""
    env = gym.make("swm/Financial-v0", max_steps=10)
    obs, info = env.reset(seed=42)

    unwrapped_env = env.unwrapped

    # Set balance and test the exact condition on line 635
    unwrapped_env.balance = 1000.0
    unwrapped_env.shares_held = 0.0

    # Execute buy with price that allows purchase (line 635: if total_cost <= self.balance)
    result = unwrapped_env._execute_buy_order(10.0, 10.0, 1.0)

    # Should execute buy since total_cost will be <= balance
    assert result == "buy"
    assert unwrapped_env.shares_held > 0

    env.close()


def test_calculate_benchmark_return_with_two_prices():
    """Test benchmark return with exactly 2 prices for line 759."""
    env = gym.make("swm/Financial-v0", max_steps=10)
    obs, info = env.reset(seed=42)

    unwrapped_env = env.unwrapped

    # Set exactly 2 prices to test the calculation (line 759)
    unwrapped_env.price_history = [{"price": 100.0}, {"price": 105.0}]

    benchmark_return = unwrapped_env._calculate_benchmark_return()

    # Should calculate (105-100)/100 = 0.05
    assert abs(benchmark_return - 0.05) < 1e-6

    env.close()


def test_calculate_reward_with_exactly_10_returns():
    """Test reward calculation with exactly 10 returns for lines 790-793."""
    env = gym.make("swm/Financial-v0", max_steps=20)
    obs, info = env.reset(seed=42)

    unwrapped_env = env.unwrapped

    # Set exactly 10 returns to trigger volatility calculation (line 790)
    unwrapped_env.portfolio_returns = [
        0.01,
        -0.005,
        0.02,
        -0.01,
        0.015,
        0.008,
        -0.003,
        0.012,
        -0.007,
        0.009,
    ]

    # Calculate reward - should include volatility penalty (lines 790-793)
    reward = unwrapped_env._calculate_reward(0.01, 0.005)

    # With exactly 10 returns, volatility calculation should occur
    assert isinstance(reward, float)
    assert np.isfinite(reward)

    env.close()


def test_analyze_trades_with_same_day_trades():
    """Test trade analysis with trades on same day for line 1126."""
    env = gym.make("swm/Financial-v0", max_steps=30)
    obs, info = env.reset(seed=42)

    unwrapped_env = env.unwrapped

    # Create trades with 0 day difference to test max(trading_days, 1) (line 1126)

    same_timestamp = unwrapped_env.market_data.index[0]

    unwrapped_env.trade_history = [
        {
            "action": "buy",
            "portfolio_value": 100000,
            "timestamp": same_timestamp,
            "shares_held": 10.0,
            "price": 100.0,
        },
        {
            "action": "sell",
            "portfolio_value": 105000,
            "timestamp": same_timestamp,  # Same day
            "shares_held": 0.0,
            "price": 105.0,
        },
    ]

    # Analyze trades - trading_days will be 1 (line 1126: max(trading_days, 1))
    trade_analysis = unwrapped_env._analyze_trades()

    assert trade_analysis["total_trades"] == 2
    assert "trading_frequency" in trade_analysis
    assert trade_analysis["trading_frequency"] > 0

    env.close()


def test_get_returns_tear_sheet_with_matching_lengths():
    """Test returns tear sheet with matching return lengths for line 1260."""
    env = gym.make("swm/Financial-v0", max_steps=20)
    obs, info = env.reset(seed=42)

    unwrapped_env = env.unwrapped

    # Create matching length returns to test the if branch on line 1260
    unwrapped_env.portfolio_returns = [0.01, 0.02, -0.01, 0.015, 0.008]
    unwrapped_env.benchmark_returns = [
        0.005,
        0.015,
        -0.008,
        0.012,
        0.007,
    ]  # Same length
    unwrapped_env.trade_history = [{"timestamp": unwrapped_env.market_data.index[i]} for i in range(5)]

    tear_sheet = unwrapped_env.get_returns_tear_sheet_data()

    # Should generate tear sheet with rolling_beta calculated (line 1260)
    assert isinstance(tear_sheet, dict)
    assert "rolling_beta" in tear_sheet
    assert len(tear_sheet["rolling_beta"]) == 5

    env.close()


def test_get_interesting_periods_with_valid_period_data():
    """Test interesting periods analysis with period containing data for line 1527."""
    env = gym.make("swm/Financial-v0", max_steps=30)
    obs, info = env.reset(seed=42)

    # Run steps to generate data
    for _ in range(15):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            break

    unwrapped_env = env.unwrapped

    # Get timestamp range from actual data
    if unwrapped_env.trade_history and len(unwrapped_env.trade_history) > 5:
        first_ts = unwrapped_env.trade_history[2]["timestamp"]
        last_ts = unwrapped_env.trade_history[-2]["timestamp"]

        # Create period that matches actual data with returns (line 1527)
        periods = {"Test Period": (first_ts.strftime("%Y-%m-%d"), last_ts.strftime("%Y-%m-%d"))}

        analysis = unwrapped_env.get_interesting_periods_analysis(periods=periods)

        # Should have analysis for the period with data (line 1527)
        if "Test Period" in analysis:
            assert "total_return" in analysis["Test Period"]
            assert "mean_return" in analysis["Test Period"]
            assert "sharpe" in analysis["Test Period"]

    env.close()
