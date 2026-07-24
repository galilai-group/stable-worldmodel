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

from collections.abc import Iterable
from typing import Protocol, runtime_checkable


FORMATS: dict[str, type[Format]] = {}

WRITE_MODES = ('append', 'overwrite', 'error')
"""Standard writer modes shared by all formats.

- ``append``: extend the dataset if it exists; create otherwise. Default.
- ``overwrite``: drop any existing dataset and start fresh.
- ``error``: raise :class:`FileExistsError` if the dataset already exists.
"""

EPISODE_DATA_KEY = '_episode_data'
"""Reserved episode-dict key carrying episode-scoped data.

Episode-scoped values are constant within an episode but vary across
episodes (e.g. a scene XML, per-episode metadata, a full simulator state
whose dimension differs per scene). They ride inside the episode dict as a
plain ``{name: value}`` mapping under this key, next to the per-step
columns; writers that support them (``Format.supports_episode_data``)
split them out with :func:`split_episode_data` and store one value per
episode. The leading underscore keeps the key out of the per-step column
namespace — ``World.collect`` never emits ``_``-prefixed columns.
"""


def split_episode_data(ep_data: dict) -> tuple[dict, dict]:
    """Split an episode dict into ``(per_step_cols, episode_data)``.

    Non-mutating: when :data:`EPISODE_DATA_KEY` is present, returns a
    shallow copy of ``ep_data`` without it plus a copy of the episode-data
    mapping; otherwise returns ``ep_data`` itself and an empty dict.
    """
    if EPISODE_DATA_KEY not in ep_data:
        return ep_data, {}
    per_step = {k: v for k, v in ep_data.items() if k != EPISODE_DATA_KEY}
    return per_step, dict(ep_data[EPISODE_DATA_KEY] or {})


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

    #: Whether this format's writer persists episode-scoped data (the
    #: :data:`EPISODE_DATA_KEY` entry of an episode dict). Routing code
    #: (``convert``/``merge``/``ReplayBuffer.dump``) strips episode data
    #: with a warning before handing episodes to writers of formats that
    #: leave this ``False``.
    supports_episode_data: bool = False

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
    """Streaming writer protocol — append episodes, close on exit.

    Two write paths are supported:

    * :meth:`write_episode` — push one episode at a time.
    * :meth:`write_episodes` — pull from a caller-provided iterable. Formats
      that benefit from a single bulk write (e.g. Lance, where each
      ``table.add`` produces a new dataset version) override this with a
      streaming implementation; everyone else can fall back to looping over
      :meth:`write_episode`.

    Episode dicts may carry episode-scoped data under
    :data:`EPISODE_DATA_KEY`. Writers of formats declaring
    ``supports_episode_data = True`` split it out internally with
    :func:`split_episode_data` and persist one value per episode; other
    writers must never receive the key (routing code strips it first).
    """

    def __enter__(self) -> Writer: ...
    def __exit__(self, *exc) -> None: ...
    def write_episode(self, ep_data: dict) -> None: ...
    def write_episodes(self, episodes: Iterable[dict]) -> None: ...


__all__ = [
    'EPISODE_DATA_KEY',
    'FORMATS',
    'Format',
    'Writer',
    'detect_format',
    'get_format',
    'list_formats',
    'register_format',
    'split_episode_data',
]
