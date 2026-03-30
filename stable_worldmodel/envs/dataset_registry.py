"""Registry for pluggable financial dataset loaders.

Usage::

    from stable_worldmodel.envs.dataset_registry import register_financial_dataset

    def my_loader(symbols, start_date, end_date):
        # Return a pd.DataFrame with MultiIndex (date, symbol) and
        # columns: open, high, low, close, volume  (daily frequency)
        ...

    register_financial_dataset(my_loader)
"""

from __future__ import annotations

from collections.abc import Callable

_REGISTRY: dict[str, dict] = {}

# Used when no dataset with date-range metadata has been registered yet.
_FALLBACK_START_DATES = [
    '2015-01-01',
    '2016-01-01',
    '2017-01-01',
    '2018-01-01',
    '2019-01-01',
]


def register_financial_dataset(
    loader: Callable,
    name: str = 'default',
    start_date: str | None = None,
    end_date: str | None = None,
    max_stocks: int | None = None,
) -> None:
    """Register a dataset loader function under *name*.

    Args:
        loader: A callable with signature
            ``loader(symbols, start_date, end_date, universe) -> pd.DataFrame``.
            The returned DataFrame must have a MultiIndex ``(date, symbol)``
            sorted chronologically and columns
            ``open``, ``high``, ``low``, ``close``, ``volume``.
        name: Registry key (default ``'default'``).
        start_date: Earliest date the dataset covers (``'YYYY-MM-DD'``).  Used
            to auto-derive episode start dates in :class:`FinancialEnvironment`.
        end_date: Latest date the dataset covers (``'YYYY-MM-DD'``).
        max_stocks: Maximum number of stocks this dataset can contain across all
            universes.  Used by :class:`FinancialEnvironment` to compute the
            smallest square observation grid that fits all stocks.
    """
    _REGISTRY[name] = {
        'loader': loader,
        'start_date': start_date,
        'end_date': end_date,
        'max_stocks': max_stocks,
    }


def get_registered_dataset(name: str = 'default') -> Callable | None:
    """Return the loader registered under *name*, or ``None`` if absent."""
    entry = _REGISTRY.get(name)
    return entry['loader'] if entry else None


def get_registered_max_stocks() -> int | None:
    """Return the maximum ``max_stocks`` value across all registered datasets.

    Returns ``None`` when no dataset has been registered with an explicit
    ``max_stocks`` value.
    """
    values = [
        e['max_stocks']
        for e in _REGISTRY.values()
        if e.get('max_stocks') is not None
    ]
    return max(values) if values else None


def get_registered_universes() -> list[str]:
    """Return the names of all currently registered dataset loaders.

    These names are used as the universe list in :class:`FinancialEnvironment`.
    """
    return list(_REGISTRY.keys())


def get_registered_start_dates(reserve_years: int = 1) -> list[str]:
    """Derive annual episode start dates from registered dataset metadata.

    Generates one ``'YYYY-01-01'`` entry per year starting from the earliest
    registered ``start_date`` up to ``latest end_date - reserve_years``,
    leaving at least *reserve_years* of data for each episode.

    Falls back to :data:`_FALLBACK_START_DATES` when no dataset has been
    registered with explicit date-range metadata.
    """
    starts = [
        e['start_date'] for e in _REGISTRY.values() if e.get('start_date')
    ]
    ends = [e['end_date'] for e in _REGISTRY.values() if e.get('end_date')]

    if not starts or not ends:
        return list(_FALLBACK_START_DATES)

    start_year = int(min(starts)[:4])
    end_year = int(max(ends)[:4]) - reserve_years

    if end_year < start_year:
        return list(_FALLBACK_START_DATES)

    return [f'{year}-01-01' for year in range(start_year, end_year + 1)]
