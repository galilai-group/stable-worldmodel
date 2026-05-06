"""WORKAROUND module for LanceDB's incomplete fork-safety.

Everything in this file exists to work around `lancedb` < TBD's
fork-mode DataLoader crash (the ``free(): invalid pointer`` SIGABRT we
hit when workers fork while the parent's tokio runtime has work in
flight). See ``lancedb_fork_segfault_ticket.md`` at the repo root for
the full upstream report.

Until the lance/lancedb runtime fully quiesces around ``fork()``, we:

  1. Force the multiprocessing start method to ``'spawn'`` on Linux
     so workers re-import everything from a clean state instead of
     inheriting the parent's mid-flight async heap.
  2. Switch PyTorch's tensor-sharing strategy to ``'file_system'`` —
     spawn workers IPC tensors via ``/dev/shm`` fds, and cloud
     instances often reject the matching ``ftruncate`` even with
     plenty of free shm.
  3. Wrap user-supplied ``forward=`` callables in a ``__name__``-bearing
     class so spt.Module's bound-method reducer survives pickling
     under spawn (``functools.partial`` lacks ``__name__``).

**Drop this module and its imports** once the upstream fix lands —
LanceDataset can stop forcing spawn, ``forward=`` can take
``functools.partial`` directly, and the file_system strategy switch
becomes unnecessary.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import sys

import torch


_SPAWN_FORCED = False


def force_spawn_for_lancedb() -> None:
    """Idempotently switch multiprocessing to ``spawn`` on Linux.

    Called from ``LanceDataset.__init__``. No-op on non-Linux (macOS
    already defaults to spawn; Python 3.14+ POSIX too) and on processes
    where the start method has already been set to something other
    than fork.

    Also flips ``torch.multiprocessing`` to the ``file_system`` sharing
    strategy because the default ``file_descriptor`` strategy mmap()s
    files under ``/dev/shm`` and cloud instances misbehave there.
    """
    global _SPAWN_FORCED
    if _SPAWN_FORCED:
        return
    _SPAWN_FORCED = True

    if sys.platform != 'linux':
        return

    current = mp.get_start_method(allow_none=True)
    if current not in (None, 'fork'):
        return

    try:
        mp.set_start_method('spawn', force=True)
        logging.info(
            "LanceDataset: multiprocessing start method set to 'spawn' "
            '(was %s) — workaround for lancedb fork-unsafety.',
            current or 'default (fork)',
        )
    except RuntimeError as exc:
        logging.warning(
            "LanceDataset could not switch multiprocessing to 'spawn' "
            '(%s); DataLoader workers may crash.',
            exc,
        )

    try:
        torch.multiprocessing.set_sharing_strategy('file_system')
    except RuntimeError as exc:
        logging.warning(
            'LanceDataset could not switch torch sharing strategy to '
            "'file_system' (%s); workers may crash on shm IPC.",
            exc,
        )


class ForwardWithCfg:
    """Picklable wrapper for ``forward=partial(fn, cfg=cfg)``.

    spt.Module binds the user-supplied ``forward`` arg as an instance
    method (``types.MethodType(forward, self)``). When DataLoader
    workers spawn, multiprocessing's bound-method reducer fetches
    ``m.__func__.__name__`` to round-trip the method via
    ``getattr(module, 'forward')`` on the worker side.
    ``functools.partial`` has no ``__name__`` and crashes the reducer.

    Setting ``__name__ = 'forward'`` at the class level makes any
    instance of this class look like a plain method to the reducer,
    so pickle round-trips cleanly. Use as a drop-in for ``partial``::

        world_model = spt.Module(
            ...,
            forward=ForwardWithCfg(forward_fn, cfg),
        )
    """

    __name__ = 'forward'

    def __init__(self, fn, cfg):
        self.fn = fn
        self.cfg = cfg

    def __call__(self, module, batch, stage):
        return self.fn(module, batch, stage, self.cfg)


__all__ = ['ForwardWithCfg', 'force_spawn_for_lancedb']
