"""Pure-logic helpers for predict.py."""

import glob
import os
from pathlib import Path


def add_ext(path, p):
    """Return ``path`` with ``.{p}`` inserted before the suffix when
    ``p > 0``; pass through unchanged otherwise.
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
