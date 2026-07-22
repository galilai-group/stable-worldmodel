"""Regression repro: reading a long episode of a Lance table hangs forever.

WHAT THE BUG LOOKS LIKE
-----------------------
Take a Lance (or lance_video) dataset written in a *single* streaming session
whose episodes have very different lengths — a short episode followed by a much
longer one — and whose schema has one or more *very wide* fixed-size-list
columns (e.g. a flattened depth image, ~200k floats per row). A bulk read of
the long episode never returns::

    ds = LanceDataset(path=merged)          # 2 episodes: 68 steps, then 300
    ds.load_episode(1)                      # <- hangs forever

The read is pinned inside Lance's own take/scan path
(``LanceDataset._fetch_rows`` -> ``lancedb...Permutation.__getitems__`` ->
lance ``take_offsets``); it is not a stable-worldmodel loop. A plain
``lance.dataset(table).scanner(offset=68, limit=300).to_table()`` hangs the
same way, so the defect is below our reader.

HOW WE HIT IT
-------------
``swm merge`` of two single-episode shards produced the first *multi-episode*
Lance table anyone had read back, and it hung in ``swm convert``. The source
shards each have one episode, so they never triggered it — which is exactly why
the shards read fine but the merged output did not. It is a latent property of
the writers, not of merge.

WHERE IT COMES FROM
-------------------
``LanceWriter._consume_episodes`` (and ``LanceVideoWriter.close``) build the
frames table by handing Lance a ``pa.RecordBatchReader`` with **one RecordBatch
per episode**::

    def batch_gen():
        yield self._batch_from_episode(first_ep)     # 68 rows
        for ep in iterator:
            yield self._batch_from_episode(ep)        # 300 rows
    reader = pa.RecordBatchReader.from_batches(self._schema, batch_gen())
    self._db.create_table(self.table_name, data=reader, schema=self._schema)

Lance's v2.1 writer sizes a wide column's on-disk pages from the batches it is
fed. A short first batch followed by a much longer one yields a page layout
that deadlocks the reader on a bulk read spanning the long episode. Empirically
(see the sweep that accompanied this test):

* a **single-episode** table (one batch) reads fine — the source shards;
* the **same rows written in uniform, modest-sized batches** read fine;
* only the **short-then-long, one-batch-per-episode** stream hangs;
* it needs the columns to be genuinely wide *and* the long episode large
  (~hundreds of MB); shrinking either makes it read normally again.

HOW WE COULD SOLVE IT
---------------------
Normalize the batch stream before handing it to Lance so no single (wide)
batch reaches the writer. A streaming, low-memory ``rechunk_batches`` helper
that splits every ``RecordBatch`` to a uniform cap (~128 rows) and passes it
through ``RecordBatchReader.from_batches(...)`` in both writers fixes it: the
merged dataset then reads and converts normally, at negligible cost. (That fix
was prototyped and reverted; this test is left to pin the bug until we decide
where to address it — here or by upgrading/patching Lance itself.)

RUNNING IT
----------
Marked ``xfail`` (the read currently hangs) and gated behind
``lancedb``/``numpy``. It writes ~280 MB and, while the bug is present, waits
out a read timeout, so it is *slow* — set ``SWM_SKIP_LANCE_HANG_REPRO=1`` to
skip it in fast local runs. The read runs in a spawned subprocess that is
killed on timeout, so a hang here never wedges the test session.
"""

from __future__ import annotations

import multiprocessing as mp
import os

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get('SWM_SKIP_LANCE_HANG_REPRO') == '1',
    reason='SWM_SKIP_LANCE_HANG_REPRO set',
)

# Trigger geometry (from the reproduction sweep): one ~200k-float column plus a
# couple more wide ones and a spread of small ones, a short first episode, and
# a long second episode of a few hundred rows. Shrinking the widths or the long
# length below roughly these values makes the read return normally.
_WIDE_DIMS = (219648, 8192, 4096)
_N_SMALL = 30
_SHORT_LEN = 68
_LONG_LEN = 300
_READ_TIMEOUT_S = 25.0


def _write_short_then_long(path: str) -> None:
    """Write a 2-episode Lance table (short then long) with wide columns,
    via a single ``write_episodes`` call — the exact shape ``swm merge``
    produces and the layout that hangs the reader."""
    import numpy as np

    from stable_worldmodel.data import LanceWriter

    rng = np.random.default_rng(0)

    def episode(n_steps: int) -> dict:
        ep = {
            'action': [
                rng.standard_normal(2).astype(np.float32)
                for _ in range(n_steps)
            ]
        }
        for i, dim in enumerate(_WIDE_DIMS):
            ep[f'wide_{i}'] = [
                rng.standard_normal(dim).astype(np.float32)
                for _ in range(n_steps)
            ]
        for j in range(_N_SMALL):
            ep[f'small_{j}'] = [
                rng.standard_normal(3).astype(np.float32)
                for _ in range(n_steps)
            ]
        return ep

    with LanceWriter(path) as w:
        w.write_episodes([episode(_SHORT_LEN), episode(_LONG_LEN)])


def _read_long_episode(path: str) -> None:
    """Subprocess entrypoint: bulk-read the long episode (this is what hangs)."""
    from stable_worldmodel.data import LanceDataset

    LanceDataset(path=path).load_episode(1)


@pytest.mark.xfail(
    reason='Lance read hangs on a long episode when a wide-column table was '
    'written as a short-then-long per-episode batch stream; see module '
    'docstring for root cause and fix.',
    strict=False,
    raises=TimeoutError,
)
def test_read_long_episode_of_wide_column_table_hangs(tmp_path):
    pytest.importorskip('lancedb')
    pytest.importorskip('numpy')

    table = str(tmp_path / 'short_then_long.lance')
    _write_short_then_long(table)

    # Read in a spawned (clean-runtime) subprocess so a hang is bounded: if the
    # bug is present the process never finishes and we kill it after the
    # timeout, raising TimeoutError (the xfail condition). When the layout is
    # eventually fixed the read returns in ~2s and the test xpasses.
    ctx = mp.get_context('spawn')
    proc = ctx.Process(target=_read_long_episode, args=(table,))
    proc.start()
    proc.join(_READ_TIMEOUT_S)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        raise TimeoutError(
            f'Reading the long episode did not finish within '
            f'{_READ_TIMEOUT_S}s — the Lance wide-column read hang.'
        )

    assert proc.exitcode == 0, f'reader subprocess exited with {proc.exitcode}'
