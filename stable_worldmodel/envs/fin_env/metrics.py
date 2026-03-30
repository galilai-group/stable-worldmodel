"""Financial performance metric utility functions.

These functions are used externally by benchmark eval scripts.
They are NOT called inside FinancialEnvironment itself.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import stats as scipy_stats

PERIODS_PER_YEAR: int = 252


def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.02,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> float:
    if len(returns) < 2:
        return 0.0
    risk_free_period = risk_free_rate / periods_per_year
    excess_returns = returns - risk_free_period
    std = np.std(excess_returns, ddof=1)
    if std == 0:
        return 0.0
    return float(np.mean(excess_returns) / std * np.sqrt(periods_per_year))


def calculate_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.02,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> float:
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
    return float(
        np.mean(excess_returns) * periods_per_year / downside_volatility
    )


def calculate_max_drawdown(portfolio_values: Sequence[float]) -> float:
    if len(portfolio_values) < 2:
        return 0.0
    values = np.array(portfolio_values)
    cumulative_max = np.maximum.accumulate(values)
    drawdowns = (values - cumulative_max) / cumulative_max
    return float(abs(np.min(drawdowns)))


def calculate_information_ratio(
    returns: np.ndarray,
    benchmark_returns: np.ndarray,
    annualized_return: float,
    benchmark_annualized: float,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> tuple[float, float]:
    if len(returns) != len(benchmark_returns) or len(returns) < 2:
        return 0.0, 0.0
    active_returns = returns - benchmark_returns
    tracking_error = np.std(active_returns, ddof=1) * np.sqrt(periods_per_year)
    if tracking_error == 0:
        return 0.0, 0.0
    information_ratio = (
        annualized_return - benchmark_annualized
    ) / tracking_error
    return float(information_ratio), float(tracking_error)


def calculate_annualized_return(
    total_return: float,
    total_periods: int,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> float:
    if total_periods == 0:
        return 0.0
    years = total_periods / periods_per_year
    if years < 1 / periods_per_year:
        years = 1 / periods_per_year
    return float((1 + total_return) ** (1 / years) - 1)


def calculate_annualized_volatility(
    returns: np.ndarray,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> float:
    """Annualized standard deviation of returns (AVol)."""
    if len(returns) < 2:
        return 0.0
    return float(np.std(returns, ddof=1) * np.sqrt(periods_per_year))


def calculate_ic(
    predicted_returns: np.ndarray,
    realized_returns: np.ndarray,
) -> float:
    """Information Coefficient — Pearson correlation between predicted and
    realized cross-sectional returns at a single timestep.

    IC > 0 means predictions are positively correlated with outcomes.
    IC = 1 means perfect prediction.
    """
    if (
        len(predicted_returns) != len(realized_returns)
        or len(predicted_returns) < 2
    ):
        return 0.0
    std_p = np.std(predicted_returns)
    std_r = np.std(realized_returns)
    if std_p == 0 or std_r == 0:
        return 0.0
    return float(np.corrcoef(predicted_returns, realized_returns)[0, 1])


def calculate_icir(
    ic_series: np.ndarray,
) -> float:
    """IC Information Ratio — mean IC divided by std of IC over time.

    Measures consistency of prediction skill.
    A high ICIR means the model's IC is stable across time, not just
    occasionally lucky.

    Args:
        ic_series: Array of per-timestep IC values computed via calculate_ic.
    """
    if len(ic_series) < 2:
        return 0.0
    std = np.std(ic_series, ddof=1)
    if std == 0:
        return 0.0
    return float(np.mean(ic_series) / std)


def calculate_rank_ic(
    predicted_returns: np.ndarray,
    realized_returns: np.ndarray,
) -> float:
    """Rank IC — Spearman rank correlation between predicted and realized
    cross-sectional returns at a single timestep.

    More robust than IC because it is insensitive to outliers and does not
    assume a linear relationship between predictions and outcomes. This is
    the primary ranking metric for cross-sectional models.
    """
    if (
        len(predicted_returns) != len(realized_returns)
        or len(predicted_returns) < 2
    ):
        return 0.0
    correlation, _ = scipy_stats.spearmanr(predicted_returns, realized_returns)
    if np.isnan(correlation):
        return 0.0
    return float(correlation)


def calculate_rank_icir(
    rank_ic_series: np.ndarray,
) -> float:
    """Rank ICIR — mean Rank IC divided by std of Rank IC over time.

    Measures consistency of cross-sectional ranking skill.
    Analogous to ICIR but based on Spearman correlation, making it more
    robust to outlier days.

    Args:
        rank_ic_series: Array of per-timestep Rank IC values computed
            via calculate_rank_ic.
    """
    if len(rank_ic_series) < 2:
        return 0.0
    std = np.std(rank_ic_series, ddof=1)
    if std == 0:
        return 0.0
    return float(np.mean(rank_ic_series) / std)


def calculate_ic_series(
    predicted_returns_sequence: Sequence[np.ndarray],
    realized_returns_sequence: Sequence[np.ndarray],
) -> dict[str, float]:
    """Compute IC, ICIR, RankIC, RankICIR from sequences of per-timestep arrays.

    This is the primary entry point for ranking metric computation in eval
    scripts. Pass in one array of predicted returns per timestep and one array
    of realized returns per timestep.

    Returns a dict with keys: ic_mean, icir, rank_ic_mean, rank_icir.

    Example::

        results = calculate_ic_series(
            predicted_returns_sequence=model_predictions,  # list of (n_stocks,) arrays
            realized_returns_sequence=actual_returns,      # list of (n_stocks,) arrays
        )
    """
    ic_vals = []
    rank_ic_vals = []

    for pred, real in zip(
        predicted_returns_sequence, realized_returns_sequence
    ):
        pred = np.asarray(pred, dtype=np.float32)
        real = np.asarray(real, dtype=np.float32)
        ic_vals.append(calculate_ic(pred, real))
        rank_ic_vals.append(calculate_rank_ic(pred, real))

    ic_arr = np.array(ic_vals, dtype=np.float32)
    rank_ic_arr = np.array(rank_ic_vals, dtype=np.float32)

    return {
        'ic_mean': float(np.mean(ic_arr)),
        'icir': calculate_icir(ic_arr),
        'rank_ic_mean': float(np.mean(rank_ic_arr)),
        'rank_icir': calculate_rank_icir(rank_ic_arr),
    }
