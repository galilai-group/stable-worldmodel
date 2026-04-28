"""Format registry — declarative read/write abstraction over the dataset
backends shipped with stable-worldmodel.

A `Format` ties together three concerns for a given on-disk layout:
  - `name`  — the registry key, also used as the explicit `format=` kwarg.
  - `detect(path)` — does this path look like our format?
  - `open_reader(path, **kw)` / `open_writer(path, **kw)` — entry points.

To add a new format, drop a file under `data/formats/`, subclass `Format`,
implement the methods you need, and decorate with `@register_format`.
Read-only formats simply omit `open_writer`; write-only formats omit
`open_reader`. Both calls raise a clear error by default.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


FORMATS: dict[str, type[Format]] = {}

WRITE_MODES = ('append', 'overwrite', 'error')
"""Standard writer modes shared by all formats.

- ``append``: extend the dataset if it exists; create otherwise. Default.
- ``overwrite``: drop any existing dataset and start fresh.
- ``error``: raise :class:`FileExistsError` if the dataset already exists.
"""


def validate_write_mode(mode: str) -> str:
    if mode not in WRITE_MODES:
        raise ValueError(
            f'write mode must be one of {WRITE_MODES}, got {mode!r}'
        )
    return mode


def register_format(cls: type[Format]) -> type[Format]:
    name = getattr(cls, 'name', None)
    if not name:
        raise ValueError(
            f'{cls.__name__} must set a non-empty `name` class attribute'
        )
    if name in FORMATS:
        raise ValueError(f'format {name!r} is already registered')
    FORMATS[name] = cls
    return cls


def list_formats() -> list[str]:
    return list(FORMATS)


def get_format(name: str) -> type[Format]:
    try:
        return FORMATS[name]
    except KeyError:
        raise ValueError(
            f'unknown format {name!r}; available: {list_formats()}'
        ) from None


def detect_format(path) -> type[Format] | None:
    """Return the first registered format whose `detect()` matches, else None."""
    for fmt in FORMATS.values():
        if fmt.detect(path):
            return fmt
    return None


class Format:
    """Declarative format spec.

    Subclasses set `name` and implement `detect`. They typically also
    implement `open_reader` and/or `open_writer`.
    """

    name: str = ''

    @classmethod
    def detect(cls, path) -> bool:
        raise NotImplementedError(f'{cls.__name__}.detect must be implemented')

    @classmethod
    def open_reader(cls, path, **kwargs):
        raise NotImplementedError(
            f'format {cls.name or cls.__name__!r} does not support reading'
        )

    @classmethod
    def open_writer(cls, path, **kwargs) -> Writer:
        """Return a streaming :class:`Writer` for this format.

        All built-in writers accept a standard ``mode`` kwarg with values
        from :data:`WRITE_MODES` (``'append'`` | ``'overwrite'`` |
        ``'error'``); the default is ``'append'``. Schema mismatches in
        append mode raise a clear exception before any data is written.
        """
        raise NotImplementedError(
            f'format {cls.name or cls.__name__!r} does not support writing (read-only)'
        )


@runtime_checkable
class Writer(Protocol):
    """Streaming writer protocol — append episodes, close on exit."""

    def __enter__(self) -> Writer: ...
    def __exit__(self, *exc) -> None: ...
    def write_episode(self, ep_data: dict) -> None: ...


__all__ = [
    'FORMATS',
    'Format',
    'Writer',
    'detect_format',
    'get_format',
    'list_formats',
    'register_format',
]
