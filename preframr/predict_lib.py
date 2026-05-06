"""Pure-logic helpers for predict.py.

Extracted out so they can be unit-tested without standing up a full
checkpoint + dataset + GPU. predict.py is the CLI orchestrator that
imports from this module; integration tests cover the orchestrator,
unit tests cover this module.
"""

import glob
import os
from pathlib import Path

from preframr.sidwav import sidq


def describe_cycles(cycles):
    """Format a cycle count as ``"<int> cycles <float> seconds"``.

    Used for logging prompt / generated lengths in the predict path.
    Seconds is computed via ``sidq()`` (PAL clock period).
    """
    return f"{int(cycles)} cycles {cycles*sidq():.2f} seconds"


def add_ext(path, p):
    """Return ``path`` with ``.{p}`` inserted before the suffix when
    ``p > 0``; pass through unchanged otherwise.

    The predict CLI runs ``args.predictions`` separate predicts on the
    same prompt; each gets its own .wav / .csv via this helper so the
    files don't clobber each other.
    """
    if p > 0:
        path = Path(path)
        path = str(path.parent / (path.stem + f".{p}" + path.suffix))
        return path
    return path


def get_ckpt(ckpt, tb_logs):
    """Pick the checkpoint file to load. Returns ``ckpt`` if explicitly
    supplied; otherwise globs ``tb_logs`` for the most recent
    ``best-*.ckpt`` (val-loss best) and falls back to the latest
    per-epoch ``*.ckpt``.

    Generalisation runs train two checkpointers in parallel: the
    per-epoch saver and the best-val saver. The best-val one is the
    right model state to evaluate on held-out songs. Memorise runs
    have no val data and write only per-epoch ckpts, so the second
    glob picks the latest of those instead.

    Raises ``IndexError`` if nothing matches.
    """
    if ckpt:
        return ckpt
    for pattern in (f"{tb_logs}/**/best-*.ckpt", f"{tb_logs}/**/*.ckpt"):
        ckpts = sorted(
            [(os.path.getmtime(p), p) for p in glob.glob(pattern, recursive=True)]
        )
        if ckpts:
            return ckpts[-1][1]
    raise IndexError("no checkpoint")
