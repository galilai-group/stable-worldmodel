---
title: Financial Markets (STONK)
summary: Daily cross-section portfolio allocation in financial markets
---

## Description

A daily cross-section financial backtesting environment where the agent allocates a portfolio across a universe of stocks. At each timestep the agent observes a heatmap encoding every stock's daily return and the current portfolio weights, then assigns each stock to a quintile — strong short, short, hold, long, or strong long. The environment is **data-agnostic**: any OHLCV dataset can be plugged in via `register_financial_dataset`.

Unlike standard reinforcement learning benchmarks, state transitions are fully determined by historical price data, making the environment suitable for testing world-model representations of financial cross-sections.

```python
import stable_worldmodel as swm
from stable_worldmodel.envs.dataset_registry import register_financial_dataset

# Register your own OHLCV loader first (see Dataset Registration below)
register_financial_dataset(my_loader, name="default", max_stocks=500)

world = swm.World('swm/STONK-v0', num_envs=4)
```

## Environment Specs

| Property | Value |
|----------|-------|
| Action Space | `MultiDiscrete([5] * n_stocks)` — per-stock quintile assignment |
| Observation Space | `Box(0, 255, shape=(A, A, 6), dtype=uint8)` — stacked RGB heatmaps |
| Reward | Portfolio return minus transaction costs (higher is better) |
| Episode Length | All trading days between `start_date` and `end_date` |
| Render Size | A×A upscaled to 224×224 (rgb_array mode) |
| Physics | Tabular price replay (no simulation) |
| Environment ID | `swm/STONK-v0` |

`A` is the smallest integer satisfying `A × A ≥ max_stocks` across all registered datasets (e.g. A = 23 for 500 stocks).

### Action Space

Each element of the `MultiDiscrete` action vector assigns one stock to a quintile:

| Quintile | Value | Raw weight | Description |
|----------|-------|------------|-------------|
| 0 | `strong short` | −1.0 | Maximum short position |
| 1 | `short` | −0.5 | Partial short position |
| 2 | `hold` | 0.0 | No position |
| 3 | `long` | +0.5 | Partial long position |
| 4 | `strong long` | +1.0 | Maximum long position |

Raw weights are normalized before execution: long weights sum to 1 and short weights sum to −1. Stocks assigned `hold` contribute nothing to the portfolio.

### Observation Details

The observation is a `(A, A, 6)` uint8 tensor formed by stacking two RGB heatmaps along the channel axis:

| Channels | Image | Encoding |
|----------|-------|----------|
| 0–2 | Market heatmap | Each pixel = one stock's daily return |
| 3–5 | Portfolio heatmap | Each pixel = one stock's current portfolio weight |

In both images, pixel intensity encodes magnitude and colour encodes direction:

- **Green** — positive return / long position (intensity ∝ magnitude)
- **Red** — negative return / short position (intensity ∝ |magnitude|)
- **Black** — zero return or no position

For the market heatmap, intensity saturates at `agent.return_clip` (default ±5%). For the portfolio heatmap, intensity saturates at a weight magnitude of 1.0.

Stocks are sorted alphabetically and arranged row-major in the grid. Unused pixels (when `n_stocks < A²`) are black.

### Info Dictionary

The `info` dict returned by `step()` and `reset()` contains:

| Key | Description |
|-----|-------------|
| `date` | Current trading date string (`'YYYY-MM-DD'`) |
| `portfolio_value` | Current portfolio value in currency units |
| `daily_return` | Portfolio return for the last step (before costs) |
| `benchmark_return` | Equal-weight benchmark return for the last step |
| `drawdown` | Current drawdown from episode peak portfolio value |
| `universe` | Name of the active registered dataset universe |
| `n_stocks` | Number of stocks active in the current episode |
| `goal` | Zero array matching observation shape (placeholder) |

## Variation Space

All variation factors are randomized by default at each `reset()`.

| Factor | Type | Default | Description |
|--------|------|---------|-------------|
| `market.start_date_idx` | `Discrete(n_dates)` | `0` | Index into the list of annual episode start dates |
| `market.universe_idx` | `Discrete(n_universes)` | last | Index of the registered dataset universe to use |
| `agent.starting_balance` | `Box(10000, 1000000)` | `100000` | Initial portfolio value in currency units |
| `agent.transaction_cost` | `Box(0.0, 0.01)` | `0.001` | Cost per unit of L1 portfolio turnover |
| `agent.return_clip` | `Box(0.01, 0.20)` | `0.05` | Daily return threshold for full colour saturation (±5%) |

Episode start dates are derived automatically from the registered dataset's `start_date` / `end_date`, with at least one year of data reserved before `end_date`. The set of available universes is the list of registered dataset names.

To fix specific factors rather than randomizing them:

```python
# Fix the starting balance and transaction cost; randomize the rest
world.reset(options={
    'variation': ['market.start_date_idx', 'market.universe_idx', 'agent.return_clip'],
})

# Randomize everything (default behaviour)
world.reset(options={'variation': ['all']})
```

## Dataset Registration

`FinancialEnvironment` requires at least one dataset to be registered before calling `reset()`. Register a loader once at startup:

```python
from stable_worldmodel.envs.dataset_registry import register_financial_dataset

def my_loader(
    symbols=None,
    start_date: str = '2015-01-01',
    end_date: str = '2024-12-31',
    universe: str = 'default',
) -> pd.DataFrame:
    """Return daily OHLCV data as a MultiIndex DataFrame."""
    ...

register_financial_dataset(
    my_loader,
    name='default',      # registry key / universe name
    start_date='2015-01-01',
    end_date='2024-12-31',
    max_stocks=500,       # maximum stocks across all universes
)
```

### Loader Contract

The loader callable must satisfy:

| Requirement | Detail |
|-------------|--------|
| **Signature** | `loader(symbols, start_date, end_date, universe) -> pd.DataFrame` |
| **Index** | `pd.MultiIndex` with levels `date` (str, `'YYYY-MM-DD'`) and `symbol` (str) |
| **Columns** | `open`, `high`, `low`, `close`, `volume` (all numeric, daily frequency) |
| **Sort order** | Chronological (the environment will re-sort if needed) |
| **Empty data** | Raises `ValueError` inside the environment — loader must return non-empty data |

Multiple loaders can be registered under different names to create several universes that the variation space samples from:

```python
register_financial_dataset(sp500_loader,  name='sp500',  max_stocks=500)
register_financial_dataset(russell_loader, name='russell', max_stocks=2000)
# market.universe_idx will sample uniformly from ['sp500', 'russell']
```

## Performance Metrics

`stable_worldmodel.envs.fin_env.metrics` provides utility functions for evaluating trained agents. These are **not** called inside the environment itself — use them in evaluation scripts.

```python
from stable_worldmodel.envs.fin_env.metrics import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_information_ratio,
    calculate_annualized_return,
    calculate_annualized_volatility,
    calculate_ic,
    calculate_icir,
    calculate_rank_ic,
    calculate_rank_icir,
    calculate_ic_series,
)
```

### Return-Based Metrics

| Function | Description |
|----------|-------------|
| `calculate_sharpe_ratio(returns)` | Annualized Sharpe ratio (risk-free rate = 2%) |
| `calculate_sortino_ratio(returns)` | Annualized Sortino ratio (downside deviation only) |
| `calculate_max_drawdown(portfolio_values)` | Maximum peak-to-trough drawdown fraction |
| `calculate_annualized_return(total_return, n_periods)` | Compound annual growth rate |
| `calculate_annualized_volatility(returns)` | Annualized standard deviation of daily returns |
| `calculate_information_ratio(returns, benchmark_returns, ...)` | Active return / tracking error; returns `(IR, tracking_error)` |

All functions assume 252 trading periods per year (configurable via `periods_per_year`).

### Cross-Sectional Ranking Metrics

Cross-sectional metrics measure how well the agent ranks stocks relative to realized outcomes at each timestep, regardless of position sizing.

| Function | Description |
|----------|-------------|
| `calculate_ic(predicted, realized)` | Pearson correlation between predicted and realized returns at one step |
| `calculate_icir(ic_series)` | IC Information Ratio — mean IC / std IC over time |
| `calculate_rank_ic(predicted, realized)` | Spearman rank correlation between predicted and realized returns at one step |
| `calculate_rank_icir(rank_ic_series)` | Rank ICIR — mean Rank IC / std Rank IC; preferred ranking metric |
| `calculate_ic_series(predicted_seq, realized_seq)` | Compute IC, ICIR, Rank IC, and Rank ICIR in one call |

**IC > 0** means predictions are positively correlated with outcomes. **Rank IC** is preferred over IC because it is insensitive to outliers and does not assume a linear relationship between predictions and realized returns.

```python
# Collect per-step arrays during an episode
predicted_list, realized_list = [], []
obs, _ = env.reset()
for _ in range(len(env._dates) - 1):
    action = policy(obs)                        # MultiDiscrete quintile action
    obs, reward, terminated, _, info = env.step(action)

    # Map quintile actions to predicted weights for IC calculation
    predicted_list.append(weights_from_action(action))
    realized_list.append(get_realized_returns(info))

metrics = calculate_ic_series(predicted_list, realized_list)
print(metrics)  # {'ic_mean': ..., 'icir': ..., 'rank_ic_mean': ..., 'rank_icir': ...}
```
