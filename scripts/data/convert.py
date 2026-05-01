"""Convert a dataset from one registered format to another.

Examples::

    # HDF5 → Lance (default destination format)
    python scripts/data/convert.py --source data.h5 --dest data.lance

    # HuggingFace HDF5 → local Lance table
    python scripts/data/convert.py --source quentinll/lewm-pusht \
        --dest /scratch/lewm_pusht.lance

    # Lance → folder (e.g. for inspection)
    python scripts/data/convert.py --source data.lance --dest data/ \
        --dest-format folder
"""

from __future__ import annotations

import argparse

from stable_worldmodel.data import convert


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--source',
        required=True,
        help='Path, HF repo id, or scheme-prefixed identifier '
        '(e.g. lerobot://lerobot/pusht).',
    )
    parser.add_argument(
        '--dest',
        required=True,
        help='Output path for the destination writer.',
    )
    parser.add_argument(
        '--source-format',
        default=None,
        help='Force the source format (skips autodetect).',
    )
    parser.add_argument(
        '--dest-format',
        default='lance',
        help='Registered writer name (default: lance).',
    )
    parser.add_argument(
        '--cache-dir',
        default=None,
        help='Override the dataset cache root.',
    )
    parser.add_argument(
        '--mode',
        choices=('append', 'overwrite', 'error'),
        default='append',
        help='Destination writer mode (default: append).',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert(
        args.source,
        args.dest,
        source_format=args.source_format,
        dest_format=args.dest_format,
        cache_dir=args.cache_dir,
        mode=args.mode,
    )


if __name__ == '__main__':
    main()
