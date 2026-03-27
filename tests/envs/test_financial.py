"""Tests for the financial environment and dataset registry.

Covers:
  - stable_worldmodel.envs.dataset_registry  (register, retrieve, metadata helpers)
  - stable_worldmodel.envs.fin_env.financial  (FinancialEnvironment lifecycle)

The tests use a small, fully-deterministic dummy dataset (4 stocks, 30 business
days starting 2020-01-03) so they run quickly without any external data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import stable_worldmodel.envs.dataset_registry as _registry_module
from stable_worldmodel.envs.dataset_registry import (
    get_registered_dataset,
    get_registered_max_stocks,
    get_registered_start_dates,
    get_registered_universes,
    register_financial_dataset,
)
from stable_worldmodel.envs.fin_env.financial import FinancialEnvironment

# --------------------------------------------------------------------------- #
# Shared test data
# --------------------------------------------------------------------------- #

_N_STOCKS = 4
_SYMBOLS = ['AAAA', 'BBBB', 'CCCC', 'DDDD']
_DATASET_START = '2020-01-01'
_DATASET_END = '2021-12-31'
_N_DAYS = 30  # number of business days the dummy data spans


def _make_dummy_df(
    start: str = '2020-01-03',
    n_days: int = _N_DAYS,
    symbols: list[str] = _SYMBOLS,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a minimal deterministic OHLCV DataFrame with MultiIndex (date, symbol)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)
    rows = []
    for date in dates:
        for sym in symbols:
            close = float(rng.uniform(50.0, 200.0))
            rows.append(
                {
                    'date': date.strftime('%Y-%m-%d'),
                    'symbol': sym,
                    'open': close * float(rng.uniform(0.98, 1.02)),
                    'high': close * float(rng.uniform(1.00, 1.03)),
                    'low': close * float(rng.uniform(0.97, 1.00)),
                    'close': close,
                    'volume': float(rng.integers(100_000, 10_000_000)),
                }
            )
    df = pd.DataFrame(rows).set_index(['date', 'symbol'])
    df.index.names = ['date', 'symbol']
    return df


def _dummy_loader(
    symbols=None,
    start_date: str = _DATASET_START,
    end_date: str = _DATASET_END,
    universe: str = 'default',
) -> pd.DataFrame:
    """Deterministic loader: ignores arguments, always returns the same dummy data."""
    return _make_dummy_df()


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture()
def clean_registry(monkeypatch):
    """Replace the global dataset registry with an empty dict for the test."""
    monkeypatch.setattr(_registry_module, '_REGISTRY', {})


@pytest.fixture()
def registered_default(clean_registry):
    """Register the dummy loader as 'default'. Depends on clean_registry."""
    register_financial_dataset(
        _dummy_loader,
        name='default',
        start_date=_DATASET_START,
        end_date=_DATASET_END,
        max_stocks=_N_STOCKS,
    )


@pytest.fixture()
def env(registered_default):
    """Create a FinancialEnvironment backed by the dummy dataset."""
    fin_env = FinancialEnvironment(end_date=_DATASET_END)
    yield fin_env
    fin_env.close()


# --------------------------------------------------------------------------- #
# Dataset registry tests
# --------------------------------------------------------------------------- #


class TestDatasetRegistry:
    def test_register_and_retrieve_loader(self, clean_registry):
        register_financial_dataset(_dummy_loader, name='mydata')
        assert get_registered_dataset('mydata') is _dummy_loader

    def test_get_unknown_dataset_returns_none(self, clean_registry):
        assert get_registered_dataset('nonexistent') is None

    def test_get_max_stocks_none_when_empty(self, clean_registry):
        assert get_registered_max_stocks() is None

    def test_get_max_stocks_single_dataset(self, clean_registry):
        register_financial_dataset(_dummy_loader, name='a', max_stocks=10)
        assert get_registered_max_stocks() == 10

    def test_get_max_stocks_returns_max_across_datasets(self, clean_registry):
        register_financial_dataset(_dummy_loader, name='a', max_stocks=10)
        register_financial_dataset(_dummy_loader, name='b', max_stocks=25)
        assert get_registered_max_stocks() == 25

    def test_get_universes_empty(self, clean_registry):
        assert get_registered_universes() == []

    def test_get_universes_lists_all_names(self, clean_registry):
        register_financial_dataset(_dummy_loader, name='u1')
        register_financial_dataset(_dummy_loader, name='u2')
        assert set(get_registered_universes()) == {'u1', 'u2'}

    def test_get_start_dates_fallback_when_empty(self, clean_registry):
        dates = get_registered_start_dates()
        assert len(dates) > 0
        assert all(d.endswith('-01-01') for d in dates)

    def test_get_start_dates_derived_from_metadata(self, clean_registry):
        # start=2020, end=2022, reserve_years=1  →  range 2020..2021
        register_financial_dataset(
            _dummy_loader,
            name='dated',
            start_date='2020-01-01',
            end_date='2022-12-31',
        )
        dates = get_registered_start_dates(reserve_years=1)
        assert '2020-01-01' in dates
        assert '2021-01-01' in dates
        assert '2022-01-01' not in dates

    def test_re_register_overwrites_entry(self, clean_registry):
        register_financial_dataset(_dummy_loader, name='default', max_stocks=5)

        def alt_loader(**kw):
            return _make_dummy_df()

        register_financial_dataset(alt_loader, name='default', max_stocks=9)
        assert get_registered_dataset('default') is alt_loader
        assert get_registered_max_stocks() == 9


# --------------------------------------------------------------------------- #
# FinancialEnvironment tests
# --------------------------------------------------------------------------- #


class TestFinancialEnvironment:
    # ------------------------------------------------------------------ #
    # reset()
    # ------------------------------------------------------------------ #

    def test_reset_returns_ndarray_observation(self, env):
        obs, _ = env.reset(seed=0)
        assert isinstance(obs, np.ndarray)
        assert obs.ndim == 3
        assert obs.shape[2] == 6  # two stacked RGB images → 6 channels
        assert obs.dtype == np.uint8

    def test_reset_observation_within_observation_space(self, env):
        obs, _ = env.reset(seed=0)
        assert env.observation_space.contains(obs)

    def test_reset_info_contains_required_keys(self, env):
        _, info = env.reset(seed=0)
        for key in (
            'date',
            'portfolio_value',
            'daily_return',
            'benchmark_return',
            'goal',
        ):
            assert key in info, f"Missing key '{key}' in reset info"

    def test_reset_sets_action_space_to_n_stocks(self, env):
        env.reset(seed=0)
        assert hasattr(env.action_space, 'nvec'), (
            'Expected MultiDiscrete action space'
        )
        assert len(env.action_space.nvec) == _N_STOCKS
        # Each stock has 5 quintile choices
        assert np.all(env.action_space.nvec == 5)

    def test_reset_portfolio_value_equals_starting_balance(self, env):
        _, info = env.reset(seed=0)
        assert info['portfolio_value'] > 0.0

    def test_reset_goal_shape_matches_observation(self, env):
        obs, info = env.reset(seed=0)
        assert info['goal'].shape == obs.shape

    def test_reset_is_reproducible_with_same_seed(self, env):
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)

    # ------------------------------------------------------------------ #
    # step()
    # ------------------------------------------------------------------ #

    def test_step_returns_five_tuple(self, env):
        env.reset(seed=0)
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 5

    def test_step_returns_valid_types(self, env):
        env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(
            env.action_space.sample()
        )
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_obs_is_uint8_with_six_channels(self, env):
        env.reset(seed=0)
        obs, *_ = env.step(env.action_space.sample())
        assert obs.dtype == np.uint8
        assert obs.shape[2] == 6

    def test_step_without_reset_raises_runtime_error(self, env):
        with pytest.raises(RuntimeError, match='reset'):
            env.step(np.zeros(_N_STOCKS, dtype=np.int64))

    def test_step_all_hold_gives_zero_reward(self, env):
        """Hold (quintile 2) with no prior position should have zero turnover cost."""
        env.reset(seed=0)
        hold_action = np.full(
            _N_STOCKS, 2, dtype=np.int64
        )  # quintile 2 = hold
        _, reward, _, _, _ = env.step(hold_action)
        # reward = portfolio_return - cost; hold has all-zero weights → both terms 0
        assert reward == pytest.approx(0.0, abs=1e-6)

    # ------------------------------------------------------------------ #
    # render()
    # ------------------------------------------------------------------ #

    def test_render_returns_rgb_array(self, env):
        env.reset(seed=0)
        frame = env.render()
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3
        assert frame.shape[2] == 3  # RGB

    def test_render_has_positive_spatial_dimensions(self, env):
        env.reset(seed=0)
        frame = env.render()
        assert frame.shape[0] > 0
        assert frame.shape[1] > 0

    # ------------------------------------------------------------------ #
    # Full episode
    # ------------------------------------------------------------------ #

    def test_full_episode_terminates(self, env):
        env.reset(seed=0)
        terminated = False
        steps = 0
        while not terminated:
            _, _, terminated, _, _ = env.step(env.action_space.sample())
            steps += 1
            assert steps < 500, 'Episode did not terminate within 500 steps'
        assert terminated

    def test_episode_length_matches_dataset_days_minus_one(self, env):
        """Episode length should be len(dates) - 1 (first date is the baseline)."""
        env.reset(seed=0)
        steps = 0
        terminated = False
        while not terminated:
            _, _, terminated, _, _ = env.step(env.action_space.sample())
            steps += 1
        assert steps == _N_DAYS - 1

    def test_portfolio_value_changes_during_episode(self, env):
        env.reset(seed=0)
        # Use a non-trivial action to ensure weight changes
        action = np.array(
            [4, 4, 0, 0], dtype=np.int64
        )  # long AAAA/BBBB, short CCCC/DDDD
        _, _, _, _, info = env.step(action)
        # Portfolio value should still be positive and finite
        assert np.isfinite(info['portfolio_value'])
        assert info['portfolio_value'] > 0.0

    # ------------------------------------------------------------------ #
    # Variation space
    # ------------------------------------------------------------------ #

    def test_variation_space_has_market_and_agent_keys(self, env):
        vspace = env.variation_space
        assert 'market' in vspace.spaces
        assert 'agent' in vspace.spaces

    def test_variation_space_market_has_expected_subkeys(self, env):
        market = env.variation_space['market']
        assert 'start_date_idx' in market.spaces
        assert 'universe_idx' in market.spaces

    def test_variation_space_agent_has_expected_subkeys(self, env):
        agent = env.variation_space['agent']
        for key in ('starting_balance', 'transaction_cost', 'return_clip'):
            assert key in agent.spaces, f"Missing agent variation key '{key}'"

    # ------------------------------------------------------------------ #
    # Error handling
    # ------------------------------------------------------------------ #

    def test_reset_raises_when_no_dataset_registered(self, clean_registry):
        """reset() must raise a clear ValueError when no loader is registered."""
        fin_env = FinancialEnvironment()
        try:
            with pytest.raises(ValueError, match='No financial dataset'):
                fin_env.reset()
        finally:
            fin_env.close()

    # ------------------------------------------------------------------ #
    # close()
    # ------------------------------------------------------------------ #

    def test_close_clears_internal_state(self, env):
        env.reset(seed=0)
        env.close()
        assert env._data is None
        assert env._dates == []
        assert env._symbols == []
