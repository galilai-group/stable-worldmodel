"""CLI to convert stable-worldmodel HDF5 datasets into LanceDB tables."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from loguru import logger as logging

from stable_worldmodel.data import convert_hdf5_to_lance
from stable_worldmodel.data.utils import get_cache_dir, load_dataset

DATASETS: dict[str, dict[str, Any]] = {
    'reacher': {
        'source': 'quentinll/lewm-reacher',
        'table_name': 'lewm_reacher',
        'columns': ['pixels', 'action', 'observation'],
    },
    'cube': {
        'source': 'quentinll/lewm-cube',
        'table_name': 'lewm_cube',
        'columns': ['pixels', 'action', 'observation'],
    },
    'pusht': {
        'source': 'quentinll/lewm-pusht',
        'table_name': 'lewm_pusht',
        'columns': ['pixels', 'action', 'proprio', 'state'],
    },
    'tworoom': {
        'source': 'quentinll/lewm-tworooms',
        'table_name': 'lewm_tworoom',
        'columns': ['pixels', 'action', 'proprio'],
    },
}


def _resolve_h5_path(source: str, cache_dir: str | None) -> Path:
    path = Path(source)
    if path.is_file() and path.suffix in ('.h5', '.hdf5'):
        return path
    if path.is_dir():
        matches = list(path.glob('*.h5')) + list(path.glob('*.hdf5'))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one .h5/.hdf5 file inside {path}, found {matches}"
            )
        return matches[0]

    dataset = load_dataset(source, cache_dir=cache_dir)
    h5_path = Path(dataset.h5_path)
    logging.info(f"Resolved '{source}' → {h5_path}")
    return h5_path


def _parse_connect_options(options: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for opt in options:
        if '=' not in opt:
            raise ValueError(
                f"Connect option '{opt}' must be in the form key=value"
            )
        key, value = opt.split('=', 1)
        result[key] = value
    return result


def _dataset_iterator(selection: str | None) -> list[str]:
    if selection is None or selection == 'all':
        return list(DATASETS.keys())
    if selection not in DATASETS:
        raise ValueError(
            f"Unknown dataset '{selection}'. Pick from {list(DATASETS)} or 'all'."
        )
    return [selection]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Convert HDF5 datasets to LanceDB tables.'
    )
    parser.add_argument(
        '--dataset',
        choices=list(DATASETS.keys()) + ['all'],
        default='all',
        help='Predefined dataset to convert (default: all official datasets).',
    )
    parser.add_argument(
        '--source',
        help='Override dataset source: path or HF repo id. Requires --table-name '
        'and --columns when set.',
    )
    parser.add_argument(
        '--columns', nargs='+', help='Columns to copy into Lance (required for --source).'
    )
    parser.add_argument(
        '--table-name',
        help='Destination table name. Defaults to the predefined name for '
        'built-in datasets.',
    )
    parser.add_argument('--lance-uri', required=True, help='LanceDB URI (local folder or s3://).')
    parser.add_argument(
        '--cache-dir',
        help='Cache directory for resolving datasets (defaults to STABLEWM_HOME).',
    )
    parser.add_argument(
        '--overwrite', action='store_true', help='Drop existing tables before writing.'
    )
    parser.add_argument('--max-episodes', type=int, help='Limit the number of episodes to convert.')
    parser.add_argument('--max-steps', type=int, help='Limit the number of steps to convert.')
    parser.add_argument(
        '--batch-rows', type=int, default=4096, help='Rows per record batch (default: 4096).'
    )
    parser.add_argument(
        '--jpeg-quality', type=int, default=95, help='JPEG quality for pixel encoding.'
    )
    parser.add_argument(
        '--connect-option',
        action='append',
        default=[],
        help='Key=Value option forwarded to lancedb.connect (e.g. endpoint_url=...).',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = None
    if args.cache_dir:
        cache_dir = str(get_cache_dir(Path(args.cache_dir)))

    connect_kwargs = _parse_connect_options(args.connect_option)

    if args.source:
        if not args.table_name or not args.columns:
            raise ValueError('--source requires --table-name and --columns')
        dataset_jobs = [
            {
                'source': args.source,
                'table_name': args.table_name,
                'columns': args.columns,
            }
        ]
    else:
        dataset_jobs = [
            {
                'source': DATASETS[name]['source'],
                'table_name': DATASETS[name]['table_name']
                if args.dataset in (name, 'all')
                else args.table_name,
                'columns': DATASETS[name]['columns'],
                'label': name,
            }
            for name in _dataset_iterator(args.dataset)
        ]
        if args.table_name and len(dataset_jobs) > 1:
            raise ValueError('--table-name can only be overridden when converting a single dataset')
        if args.table_name and len(dataset_jobs) == 1:
            dataset_jobs[0]['table_name'] = args.table_name

    for job in dataset_jobs:
        label = job.get('label', 'custom')
        logging.info(f"Converting dataset: {label}")
        h5_path = _resolve_h5_path(job['source'], cache_dir)
        summary = convert_hdf5_to_lance(
            h5_path=h5_path,
            lance_uri=args.lance_uri,
            table_name=job['table_name'],
            columns=job['columns'],
            jpeg_quality=args.jpeg_quality,
            batch_rows=args.batch_rows,
            overwrite=args.overwrite,
            max_episodes=args.max_episodes,
            max_steps=args.max_steps,
            connect_kwargs=connect_kwargs,
        )
        logging.info(
            "Wrote {rows} rows ({episodes} episodes) to {uri}/{table}".format(
                **summary
            )
        )


if __name__ == '__main__':
    main()
