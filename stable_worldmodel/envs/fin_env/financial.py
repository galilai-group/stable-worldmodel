"""STONK — financial market cross-section environment for stable-worldmodel.

Observation: A×A RGB heatmap where each pixel encodes one stock's daily
return (green = positive, red = negative, black = zero/missing).  A is the
smallest integer satisfying A*A >= max_stocks across all registered datasets.

Action: portfolio weight vector over all stocks (long/short).

Data: pluggable via ``register_financial_dataset``.
"""

from __future__ import annotations

import math

import cv2
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from stable_worldmodel import spaces as swm_spaces
from stable_worldmodel.envs.dataset_registry import (
    get_registered_dataset,
    get_registered_max_stocks,
    get_registered_start_dates,
    get_registered_universes,
    register_financial_dataset as register_financial_dataset,
)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DEFAULT_VARIATIONS = (
    'market.start_date_idx',
    'market.universe_idx',
    'agent.starting_balance',
    'agent.transaction_cost',
    'agent.return_clip',
)

_FALLBACK_IMG_SIZE = 64
_MAX_RETURN_CLIP = 0.05  # ±5 % daily return → full colour saturation


def _compute_grid_size(n_stocks: int) -> int:
    """Return the side length of the smallest square grid that fits *n_stocks* pixels."""
    return max(1, math.ceil(math.sqrt(n_stocks)))


# Quintile action values: strong short, short, hold, long, strong long
_N_QUINTILES = 5
_QUINTILE_WEIGHTS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)

# Maximum portfolio colour saturation at this weight magnitude
_MAX_WEIGHT_CLIP = 1.0


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #


class FinancialEnvironment(gym.Env):
    """Daily cross-section financial backtesting environment.

    Observation space: ``(A, A, 6)`` uint8 — two stacked RGB images (market
    heatmap and portfolio heatmap), where A is the smallest integer satisfying
    ``A*A >= max_stocks`` across all registered datasets.
    Action space:      ``MultiDiscrete([5] * n_stocks)`` — per-stock quintile
    assignment (0=strong short … 4=strong long).
    """

    metadata = {'render_modes': ['rgb_array'], 'render_fps': 1}
    reward_range = (-np.inf, np.inf)

    def __init__(
        self,
        end_date: str | None = None,
        render_mode: str | None = 'rgb_array',
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.end_date = end_date or '2024-12-31'

        # Compute the smallest square grid that fits all registered stocks.
        _max_stocks = get_registered_max_stocks()
        self._img_size: int = (
            _compute_grid_size(_max_stocks)
            if _max_stocks is not None
            else _FALLBACK_IMG_SIZE
        )

        # Observation space: two stacked RGB images (market + portfolio heatmap).
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self._img_size, self._img_size, 6),
            dtype=np.uint8,
        )
        # Placeholder action space; replaced in reset() once n_stocks is known.
        self.action_space = spaces.MultiDiscrete(
            np.full(1, _N_QUINTILES, dtype=np.int64)
        )

        self._start_dates: list[str] = get_registered_start_dates()
        self.variation_space = self._build_variation_space()

        assert self.variation_space.check(), 'Invalid default variation values'

        # Episode state (populated by reset)
        self._data: pd.DataFrame | None = None
        self._dates: list[str] = []
        self._symbols: list[str] = []
        self._current_date_idx: int = 0
        self._portfolio_value: float = 100000.0
        self._prev_weights: np.ndarray | None = None
        self._transaction_cost: float = 0.001
        self._starting_balance: float = 100000.0
        self._return_clip: float = _MAX_RETURN_CLIP
        self._close_pivot: pd.DataFrame | None = None
        self._current_weights: np.ndarray | None = (
            None  # current normalised weights (n_stocks,)
        )
        self._universe: str = 'full'  # active universe name
        self._max_portfolio_value: float = 100000.0  # for drawdown tracking

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def _build_variation_space(self) -> swm_spaces.Dict:
        """Build the variation space from ``self._start_dates`` and registered universes."""
        universes = get_registered_universes()
        n_universes = max(
            len(universes), 1
        )  # avoid n=0 before any dataset is registered
        default_universe_idx = max(n_universes - 1, 0)

        n_start_dates = max(
            len(self._start_dates), 1
        )  # avoid n=0 if list is empty

        return swm_spaces.Dict(
            {
                'market': swm_spaces.Dict(
                    {
                        'start_date_idx': swm_spaces.Discrete(
                            n=n_start_dates,
                            start=0,
                            init_value=0,
                        ),
                        'universe_idx': swm_spaces.Discrete(
                            n=n_universes,
                            start=0,
                            init_value=default_universe_idx,
                        ),
                    }
                ),
                'agent': swm_spaces.Dict(
                    {
                        'starting_balance': swm_spaces.Box(
                            low=10000.0,
                            high=1000000.0,
                            init_value=np.array(100000.0, dtype=np.float32),
                            shape=(),
                            dtype=np.float32,
                        ),
                        'transaction_cost': swm_spaces.Box(
                            low=0.0,
                            high=0.01,
                            init_value=np.array(0.001, dtype=np.float32),
                            shape=(),
                            dtype=np.float32,
                        ),
                        'return_clip': swm_spaces.Box(
                            low=0.01,
                            high=0.20,
                            init_value=np.array(
                                _MAX_RETURN_CLIP, dtype=np.float32
                            ),
                            shape=(),
                            dtype=np.float32,
                        ),
                    }
                ),
            },
            sampling_order=['market', 'agent'],
        )

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed, options=options)
        options = options or {}

        swm_spaces.reset_variation_space(
            self.variation_space, seed, options, DEFAULT_VARIATIONS
        )

        loader = get_registered_dataset('default')
        if loader is None:
            raise ValueError(
                'No financial dataset has been registered. Register one with:\n\n'
                '    from stable_worldmodel.envs.dataset_registry import register_financial_dataset\n'
                '    register_financial_dataset(your_loader)\n\n'
                'The loader must accept (symbols, start_date, end_date, universe) keyword arguments\n'
                'and return a pd.DataFrame with MultiIndex (date, symbol) and columns\n'
                'open, high, low, close, volume (daily frequency).'
            )

        start_date: str = self._start_dates[
            int(self.variation_space['market']['start_date_idx'].value)
        ]
        end_date: str = self.end_date
        universes = get_registered_universes()
        self._universe = universes[
            int(self.variation_space['market']['universe_idx'].value)
        ]
        self._starting_balance = float(
            self.variation_space['agent']['starting_balance'].value
        )
        self._transaction_cost = float(
            self.variation_space['agent']['transaction_cost'].value
        )
        self._return_clip = float(
            self.variation_space['agent']['return_clip'].value
        )

        df = loader(
            symbols=None,
            start_date=start_date,
            end_date=end_date,
            universe=self._universe,
        )
        self._validate_data(df)

        df = df.sort_index()
        self._data = df
        self._dates = sorted(
            df.index.get_level_values('date').unique().tolist()
        )
        self._symbols = sorted(
            df.index.get_level_values('symbol').unique().tolist()
        )

        self._close_pivot = (
            df['close']
            .unstack(level='symbol')
            .reindex(columns=self._symbols)
            .sort_index()
        )

        n_stocks = len(self._symbols)

        # Action space: per-stock quintile assignment
        # Each stock gets assigned to one of 5 quintiles: {0,1,2,3,4}
        # = {strong short, short, hold, long, strong long}
        self.action_space = spaces.MultiDiscrete(
            np.full(n_stocks, _N_QUINTILES, dtype=np.int64)
        )

        self._current_date_idx = 1
        self._portfolio_value = self._starting_balance
        self._max_portfolio_value = self._starting_balance
        self._current_weights = np.zeros(n_stocks, dtype=np.float32)
        self._prev_weights = np.zeros(n_stocks, dtype=np.float32)

        observation = self._get_observation()
        info = self._get_info()
        return observation, info

    def step(self, action: np.ndarray):
        if self._data is None:
            raise RuntimeError(
                'Environment not initialised — call reset() first.'
            )

        # Convert quintile assignments to normalised portfolio weights
        action = np.asarray(action, dtype=np.int64)
        weights = self._quintiles_to_weights(action)

        returns = self._get_daily_returns(self._current_date_idx)

        # Portfolio return
        portfolio_return = float(np.dot(weights, returns))
        benchmark_return = float(np.nanmean(returns))

        # Transaction cost proportional to L1 turnover
        turnover = float(np.sum(np.abs(weights - self._prev_weights)))
        cost = turnover * self._transaction_cost
        reward = portfolio_return - cost

        self._portfolio_value *= 1.0 + reward
        self._max_portfolio_value = max(
            self._max_portfolio_value, self._portfolio_value
        )
        self._prev_weights = weights.copy()
        self._current_weights = weights.copy()

        self._current_date_idx += 1
        terminated = self._current_date_idx >= len(self._dates)
        truncated = False

        observation = (
            self._get_observation() if not terminated else self._black_frame()
        )
        info = self._get_info(
            daily_return=portfolio_return, benchmark_return=benchmark_return
        )
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != 'rgb_array':
            return None
        obs = self._get_observation()
        market_img = obs[:, :, :3]  # first three channels only
        return cv2.resize(
            market_img, (224, 224), interpolation=cv2.INTER_NEAREST
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def close(self):
        """Clean up resources. Called by SWM when the world is torn down."""
        self._data = None
        self._close_pivot = None
        self._dates = []
        self._symbols = []

    @staticmethod
    def _validate_data(df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f'Dataset loader must return pd.DataFrame, got {type(df).__name__}'
            )
        if not isinstance(df.index, pd.MultiIndex):
            raise ValueError(
                'DataFrame must have a MultiIndex with levels (date, symbol)'
            )
        index_names = [n or '' for n in df.index.names]
        if 'date' not in index_names or 'symbol' not in index_names:
            raise ValueError(
                f"MultiIndex must have levels named 'date' and 'symbol', got {df.index.names}"
            )
        required = {'open', 'high', 'low', 'close', 'volume'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f'Missing required columns: {missing}')
        if df.empty:
            raise ValueError('Dataset loader returned an empty DataFrame')

    def _get_daily_returns(self, date_idx: int) -> np.ndarray:
        """Vectorised close-to-close daily return for the given date index."""
        if date_idx <= 0 or self._close_pivot is None:
            return np.zeros(len(self._symbols), dtype=np.float32)

        cur = self._close_pivot.iloc[date_idx].values
        prev = self._close_pivot.iloc[date_idx - 1].values

        with np.errstate(divide='ignore', invalid='ignore'):
            returns = np.where(prev != 0.0, (cur - prev) / prev, 0.0)

        return np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32
        )

    def _get_observation(self) -> np.ndarray:
        """Return the stacked RGB observation for the current date index."""
        if self._close_pivot is None or self._current_date_idx >= len(
            self._dates
        ):
            return self._black_frame()

        returns = self._get_daily_returns(self._current_date_idx)

        # Clip and normalise to [-1, 1]
        clipped = np.clip(returns / self._return_clip, -1.0, 1.0)

        # Pad to exactly img_size*img_size pixels
        n_pixels = self._img_size * self._img_size
        padded = np.zeros(n_pixels, dtype=np.float32)
        n = min(len(clipped), n_pixels)
        padded[:n] = clipped[:n]

        # Build RGB channels vectorised
        image = np.zeros((n_pixels, 3), dtype=np.uint8)
        pos_mask = padded > 0.0
        neg_mask = padded < 0.0
        image[pos_mask, 1] = (padded[pos_mask] * 255).astype(
            np.uint8
        )  # green channel
        image[neg_mask, 0] = (-padded[neg_mask] * 255).astype(
            np.uint8
        )  # red channel

        market_img = image.reshape(self._img_size, self._img_size, 3)
        portfolio_img = self._get_portfolio_heatmap()

        # Stack along channel axis: (A, A, 6)
        return np.concatenate([market_img, portfolio_img], axis=2)

    def _black_frame(self) -> np.ndarray:
        return np.zeros((self._img_size, self._img_size, 6), dtype=np.uint8)

    def _get_info(
        self,
        daily_return: float = 0.0,
        benchmark_return: float = 0.0,
    ) -> dict:
        idx = min(self._current_date_idx, len(self._dates) - 1)
        current_date = self._dates[idx] if self._dates else ''

        drawdown = 0.0
        if self._max_portfolio_value > 0:
            drawdown = (
                self._max_portfolio_value - self._portfolio_value
            ) / self._max_portfolio_value

        return {
            'date': current_date,
            'portfolio_value': float(self._portfolio_value),
            'daily_return': daily_return,
            'benchmark_return': benchmark_return,
            'drawdown': float(drawdown),
            'universe': self._universe,
            'n_stocks': len(self._symbols),
            'goal': np.zeros(
                (self._img_size, self._img_size, 6), dtype=np.uint8
            ),
        }

    @staticmethod
    def _quintiles_to_weights(action: np.ndarray) -> np.ndarray:
        """Convert per-stock quintile assignments to normalised portfolio weights.

        Quintile mapping:
            0 = strong short  (-1.0)
            1 = short         (-0.5)
            2 = hold          ( 0.0)
            3 = long          (+0.5)
            4 = strong long   (+1.0)

        Long positions are normalised to sum to 1.
        Short positions are normalised to sum to -1.
        """
        raw = _QUINTILE_WEIGHTS[action]  # map quintile indices to raw weights

        long_w = np.where(raw > 0.0, raw, 0.0)
        short_w = np.where(raw < 0.0, raw, 0.0)

        long_sum = float(long_w.sum())
        short_sum = float(abs(short_w.sum()))

        if long_sum > 0.0:
            long_w /= long_sum
        if short_sum > 0.0:
            short_w /= short_sum

        return (long_w + short_w).astype(np.float32)

    def _get_portfolio_heatmap(self) -> np.ndarray:
        """Return AxAx3 RGB image encoding current portfolio weights.

        Green = long position, intensity proportional to weight magnitude.
        Red   = short position, intensity proportional to weight magnitude.
        Black = no position.
        """
        if not self._symbols:
            return np.zeros(
                (self._img_size, self._img_size, 3), dtype=np.uint8
            )

        weights = (
            self._current_weights
            if self._current_weights is not None
            else np.zeros(len(self._symbols), dtype=np.float32)
        )

        clipped = np.clip(weights / _MAX_WEIGHT_CLIP, -1.0, 1.0)

        n_pixels = self._img_size * self._img_size
        padded = np.zeros(n_pixels, dtype=np.float32)
        n = min(len(clipped), n_pixels)
        padded[:n] = clipped[:n]

        image = np.zeros((n_pixels, 3), dtype=np.uint8)
        pos_mask = padded > 0.0
        neg_mask = padded < 0.0
        image[pos_mask, 1] = (padded[pos_mask] * 255).astype(np.uint8)  # green
        image[neg_mask, 0] = (-padded[neg_mask] * 255).astype(np.uint8)  # red

        return image.reshape(self._img_size, self._img_size, 3)
