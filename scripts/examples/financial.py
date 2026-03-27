"""Smoke-test the STONK financial environment with a random policy.

Generates a dummy OHLCV dataset, rolls out a random policy across
4 parallel environments, saves per-step images to ./output/, records
a video, and prints a full metrics report at the end.

Usage:
    python scripts/examples/financial.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import stable_worldmodel as swm
from stable_worldmodel.envs.dataset_registry import register_financial_dataset
from stable_worldmodel.envs.fin_env.metrics import (
    calculate_annualized_return,
    calculate_annualized_volatility,
    calculate_ic_series,
    calculate_information_ratio,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
)
from stable_worldmodel.policy import RandomPolicy

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

N_STOCKS = 50
START_DATE = '2018-01-01'
END_DATE = '2020-12-31'
NUM_ENVS = 4
IMAGE_SHAPE = (224, 224)
MAX_EPISODE_STEPS = 252  # roughly one trading year
OUTPUT_DIR = Path('output/financial')

# --------------------------------------------------------------------------- #
# Dummy OHLCV loader
# --------------------------------------------------------------------------- #

_RNG = np.random.default_rng(42)


def _generate_dummy_ohlcv(
    n_stocks: int, start_date: str, end_date: str
) -> pd.DataFrame:
    """Generate synthetic daily OHLCV data with a random-walk close price."""
    dates = (
        pd.bdate_range(start=start_date, end=end_date)
        .strftime('%Y-%m-%d')
        .tolist()
    )
    symbols = [f'STOCK_{i:03d}' for i in range(n_stocks)]

    records = []
    for sym in symbols:
        # Random-walk log-prices seeded per symbol
        log_returns = _RNG.normal(0.0003, 0.015, size=len(dates))
        log_prices = np.cumsum(log_returns) + np.log(100.0)
        close = np.exp(log_prices).astype(np.float32)
        open_ = close * (1.0 + _RNG.uniform(-0.005, 0.005, size=len(dates)))
        high = np.maximum(close, open_) * (
            1.0 + _RNG.uniform(0.0, 0.01, size=len(dates))
        )
        low = np.minimum(close, open_) * (
            1.0 - _RNG.uniform(0.0, 0.01, size=len(dates))
        )
        volume = _RNG.integers(100_000, 10_000_000, size=len(dates)).astype(
            np.float32
        )

        for i, date in enumerate(dates):
            records.append(
                (date, sym, open_[i], high[i], low[i], close[i], volume[i])
            )

    df = pd.DataFrame(
        records,
        columns=['date', 'symbol', 'open', 'high', 'low', 'close', 'volume'],
    )
    df = df.set_index(['date', 'symbol']).sort_index()
    return df


def my_loader(
    symbols=None,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    universe: str = 'default',
) -> pd.DataFrame:
    return _generate_dummy_ohlcv(N_STOCKS, start_date, end_date)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _save_observation_image(
    obs: np.ndarray, path: Path, env_idx: int, step: int
) -> None:
    """Save the 6-channel (A,A,6) observation as two side-by-side RGB images."""
    market_rgb = obs[:, :, :3]
    portfolio_rgb = obs[:, :, 3:]

    a = obs.shape[0]
    canvas = np.zeros((a, a * 2, 3), dtype=np.uint8)
    canvas[:, :a, :] = market_rgb
    canvas[:, a:, :] = portfolio_rgb

    img = Image.fromarray(canvas, mode='RGB').resize((448, 224), Image.NEAREST)
    img.save(path / f'env{env_idx}_step{step:04d}.png')


def _weights_from_action(
    action: np.ndarray, n_quintiles: int = 5
) -> np.ndarray:
    """Map quintile action vector → raw weights for IC calculation."""
    # 0=−1, 1=−0.5, 2=0, 3=+0.5, 4=+1
    quintile_map = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    return quintile_map[action]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    img_dir = OUTPUT_DIR / 'frames'
    img_dir.mkdir(exist_ok=True)

    # ── 1. Register dataset ──────────────────────────────────────────────────
    print('Registering dummy dataset …')
    register_financial_dataset(
        my_loader,
        name='default',
        start_date=START_DATE,
        end_date=END_DATE,
        max_stocks=N_STOCKS,
    )

    # ── 2. Build world ───────────────────────────────────────────────────────
    world = swm.World(
        'swm/STONK-v0',
        num_envs=NUM_ENVS,
        image_shape=IMAGE_SHAPE,
        max_episode_steps=MAX_EPISODE_STEPS,
        goal_conditioned=False,
    )

    policy = RandomPolicy(seed=0)
    world.set_policy(policy)

    # ── 3. Record a video ────────────────────────────────────────────────────
    video_path = OUTPUT_DIR / 'videos'
    video_path.mkdir(exist_ok=True)
    print(f'Recording rollout video → {video_path}')
    world.record_video(str(video_path), max_steps=MAX_EPISODE_STEPS, fps=5)

    # ── 4. Manual rollout for metrics + frame saving ─────────────────────────
    print('\nRunning manual rollout for metrics …')
    world.reset(seed=1)

    # Per-env accumulators
    daily_returns = [[] for _ in range(NUM_ENVS)]
    bench_returns = [[] for _ in range(NUM_ENVS)]
    portfolio_vals = [[] for _ in range(NUM_ENVS)]
    predicted_lists = [[] for _ in range(NUM_ENVS)]
    realized_lists = [[] for _ in range(NUM_ENVS)]

    # Record the initial frame
    for env_idx in range(NUM_ENVS):
        # pixels live in world.infos, not world.states
        world.infos.get('pixels', None)

    step = 0
    while True:
        # Sample actions from each environment's action space individually
        actions = world.envs.action_space.sample()

        # Save a frame every 10 steps from the raw observation channels
        if step % 10 == 0:
            raw_pixels = world.infos.get('pixels')
            if raw_pixels is not None:
                for env_idx in range(NUM_ENVS):
                    frame = raw_pixels[env_idx]
                    # frame may have a history dim: (T, H, W, C) → take last
                    if frame.ndim == 4:
                        frame = frame[-1]
                    # If it's already (H, W, C) with C=6 it came from the env directly;
                    # if it's (H, W, 3) it's the rendered pixel — save as-is
                    if frame.shape[-1] == 6:
                        _save_observation_image(frame, img_dir, env_idx, step)
                    else:
                        img = Image.fromarray(frame, mode='RGB')
                        img.save(img_dir / f'env{env_idx}_step{step:04d}.png')

        world.step()
        step += 1

        # Accumulate per-env metrics from infos
        for env_idx in range(NUM_ENVS):
            dr = world.infos.get('daily_return', [0.0] * NUM_ENVS)[env_idx]
            br = world.infos.get('benchmark_return', [0.0] * NUM_ENVS)[env_idx]
            pv = world.infos.get('portfolio_value', [100000.0] * NUM_ENVS)[
                env_idx
            ]
            daily_returns[env_idx].append(float(dr))
            bench_returns[env_idx].append(float(br))
            portfolio_vals[env_idx].append(float(pv))

            # Store predicted (action-weights) and realized (daily returns per stock)
            # for cross-sectional IC metrics — use the vectorised action for this env
            if actions is not None and hasattr(actions, '__len__'):
                n_stocks_env = world.envs.single_action_space.nvec.shape[0]
                env_action = (
                    actions[
                        env_idx * n_stocks_env : (env_idx + 1) * n_stocks_env
                    ]
                    if actions.ndim == 1
                    else actions[env_idx]
                )
                predicted_lists[env_idx].append(
                    _weights_from_action(env_action)
                )
                realized_lists[env_idx].append(
                    np.array([float(dr)] * len(env_action), dtype=np.float32)
                )

        if np.all(world.terminateds) or np.all(world.truncateds):
            break

    # ── 5. Print metrics ─────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('  METRICS REPORT')
    print('=' * 60)

    for env_idx in range(NUM_ENVS):
        ret = np.array(daily_returns[env_idx], dtype=np.float32)
        bench = np.array(bench_returns[env_idx], dtype=np.float32)
        pvals = np.array(portfolio_vals[env_idx], dtype=np.float32)
        n = len(ret)
        if n == 0:
            continue

        total_return = (
            float(pvals[-1] / pvals[0] - 1.0) if pvals[0] > 0 else 0.0
        )
        ann_ret = calculate_annualized_return(total_return, n)
        ann_vol = calculate_annualized_volatility(ret)
        sharpe = calculate_sharpe_ratio(ret)
        sortino = calculate_sortino_ratio(ret)
        mdd = calculate_max_drawdown(pvals)

        bench_total = float(np.prod(1.0 + bench) - 1.0)
        bench_ann = calculate_annualized_return(bench_total, n)
        ir, tracking_err = calculate_information_ratio(
            ret, bench, ann_ret, bench_ann
        )

        ic_metrics: dict = {}
        if predicted_lists[env_idx] and realized_lists[env_idx]:
            ic_metrics = calculate_ic_series(
                predicted_lists[env_idx], realized_lists[env_idx]
            )

        print(f'\n  Env {env_idx}')
        print(f'    Steps          : {n}')
        print(f'    Final PV       : {pvals[-1]:>12.2f}')
        print(f'    Total return   : {total_return * 100:>+8.2f}%')
        print(f'    Ann. return    : {ann_ret * 100:>+8.2f}%')
        print(f'    Ann. vol       : {ann_vol * 100:>8.2f}%')
        print(f'    Sharpe         : {sharpe:>8.3f}')
        print(f'    Sortino        : {sortino:>8.3f}')
        print(f'    Max drawdown   : {mdd * 100:>8.2f}%')
        print(
            f'    Info. ratio    : {ir:>8.3f}  (tracking err {tracking_err * 100:.2f}%)'
        )
        if ic_metrics:
            print(
                f'    IC mean        : {ic_metrics.get("ic_mean", float("nan")):>8.4f}'
            )
            print(
                f'    ICIR           : {ic_metrics.get("icir", float("nan")):>8.4f}'
            )
            print(
                f'    Rank IC mean   : {ic_metrics.get("rank_ic_mean", float("nan")):>8.4f}'
            )
            print(
                f'    Rank ICIR      : {ic_metrics.get("rank_icir", float("nan")):>8.4f}'
            )

    print('\n' + '=' * 60)
    print(f'  Frame images saved to : {img_dir}')
    print(f'  Videos saved to       : {video_path}')
    print('=' * 60)

    world.close()


if __name__ == '__main__':
    main()
