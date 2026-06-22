"""BACC corpus builder: a (.sid, subtune) manifest -> per-tune model-id block arrays.

The codec recovers a program directly from a ``.sid`` (the generic sidtrace path,
``recover_from_sid``) -- no pre-rendered ``.dump.parquet``. Each manifest entry is
``relpath<TAB>subtune`` (HVSC-relative ``.sid`` + 1-based subtune); the frame
budget per subtune comes from the HVSC Songlengths. Tunes the codec cannot recover
(or that fail) are logged and skipped. Each kept tune is serialized to the fixed
BACC alphabet, windowed into ``seq_len`` blocks (PAD-padded), and written next to
the ``.sid`` as ``<sid_base>.<subtune>.blocks.npy``."""

import logging
import os
from dataclasses import dataclass

import numpy as np
from preframr_tokens import CPF, program_to_ids
from preframr_tokens.bacc.generic import recover_from_sid

from preframr.songlengths import subtune_frames
from preframr.tokenizer import PAD_ID, BaccTokenizer

BLOCKS_SUFFIX = ".blocks.npy"


@dataclass
class SeqMeta:
    """Per-tune metadata the predict/logging path reads off a block set: source .sid path, frame clock (cycles/frame, carried into render timing), and 1-based subtune index."""

    df_file: str
    irq: float
    subtune: int


def _blocks_path(sid_path, subtune):
    """``<sid_base>.<subtune>.blocks.npy`` next to the .sid (subtune-distinct)."""
    base = sid_path[: -len(".sid")] if sid_path.endswith(".sid") else sid_path
    return f"{base}.{subtune}{BLOCKS_SUFFIX}"


def _windows(ids, seq_len, stride):
    """Right-padded ``seq_len + 1``-wide windows (x = block[:-1], y = block[1:]); the extra +1 gives ``BlockMapper`` its shifted target. A tune shorter than a window yields a single PAD-padded block (PAD is masked out of the loss)."""
    width = seq_len + 1
    blocks = []
    for start in range(0, max(1, len(ids) - 1), stride):
        win = ids[start : start + width]
        if len(win) < 2:
            continue
        if len(win) < width:
            win = win + [PAD_ID] * (width - len(win))
        blocks.append(win)
        if start + width >= len(ids):
            break
    return blocks


def _load_cached_blocks(blocks_path, seq_len):
    """Return an existing ``.blocks.npy`` iff its width matches ``seq_len + 1`` (so a stale array built at a different seq_len is rebuilt, not silently reused), else None."""
    if not os.path.exists(blocks_path):
        return None
    try:
        arr = np.load(blocks_path, mmap_mode="r")
    except (ValueError, OSError):
        return None
    if arr.ndim != 2 or arr.shape[1] != seq_len + 1 or arr.shape[0] == 0:
        return None
    return arr


def read_manifest(path):
    """Parse a ``relpath<TAB>subtune`` manifest -> [(relpath, subtune)]. A bare
    relpath (no tab) defaults to subtune 1; ``#`` comments + blanks ignored."""
    out = []
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            subtune = int(parts[1]) if len(parts) > 1 else 1
            out.append((parts[0], subtune))
    return out


def parse_eval_manifests(spec):
    """``name=path;name=path`` -> [(name, path)] for named eval subsets."""
    out = []
    for chunk in (spec or "").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, _, path = chunk.partition("=")
        out.append((name.strip(), path.strip()))
    return out


class Corpus:
    """Owns the BACC tokenizer + the manifest -> block-array orchestration."""

    def __init__(self, args, logger=logging):
        self.args = args
        self.logger = logger
        self.tokenizer = BaccTokenizer()
        self.reg_widths = {}

    @property
    def n_vocab(self):
        return self.tokenizer.n_vocab

    @property
    def n_words(self):
        return self.tokenizer.n_words

    def preload(self, tokens=None, tkmodel=None):  # pylint: disable=unused-argument
        """No-op: the BACC alphabet is fixed, so there is nothing to pre-train; kept for signature compatibility with the old corpus (predict passes the checkpoint's tokens/tkmodel, both fixed/None here)."""

    def _sid_path(self, relpath):
        root = getattr(self.args, "sid_root", "") or ""
        return os.path.join(root, relpath) if root else relpath

    def _build_blocks(self, relpath, subtune):
        """Recover + tokenize + window one (sid, subtune); write the blocks array. Returns ``(blocks_path, seq_meta)`` or ``None`` if skipped (missing .sid, no Songlengths entry, no matching driver, or a parse failure). An existing ``.blocks.npy`` of the right width is reused, so a prior parse run (or a restored cache) skips the codec recovery."""
        sid = self._sid_path(relpath)
        seq_len = self.args.seq_len
        blocks_path = _blocks_path(sid, subtune)
        cached = _load_cached_blocks(blocks_path, seq_len)
        if cached is not None:
            self.logger.info(
                "%s.%u: reuse %s (%u blocks)",
                os.path.basename(relpath),
                subtune,
                os.path.basename(blocks_path),
                cached.shape[0],
            )
            return blocks_path, SeqMeta(df_file=sid, irq=CPF, subtune=subtune)
        if not os.path.exists(sid):
            self.logger.info("skip %s: no .sid", sid)
            return None
        try:
            nframes = subtune_frames(sid, subtune, self.args.songlengths)
            program, _resid, _dump = recover_from_sid(
                sid, subtune=subtune, nframes=nframes
            )
            ids = [i + 1 for i in program_to_ids(program)]
        except Exception as err:  # pylint: disable=broad-except
            self.logger.info("skip %s.%u: %s", relpath, subtune, err)
            return None
        stride = getattr(self.args, "block_stride", None) or seq_len
        blocks = _windows(ids, seq_len, stride)
        if not blocks:
            self.logger.info(
                "skip %s.%u: too short (%u ids)", relpath, subtune, len(ids)
            )
            return None
        arr = np.asarray(blocks, dtype=np.int16)
        np.save(blocks_path, arr)
        self.logger.info(
            "%s.%u -> %s (%u ids, %u blocks of %u)",
            os.path.basename(relpath),
            subtune,
            os.path.basename(blocks_path),
            len(ids),
            arr.shape[0],
            arr.shape[1],
        )
        return blocks_path, SeqMeta(df_file=sid, irq=CPF, subtune=subtune)

    def _iter(self, kind, manifest_path):
        """Yield ``(kind, blocks_path, seq_meta)`` for each kept entry of a manifest."""
        if not manifest_path or not os.path.exists(manifest_path):
            return
        entries = read_manifest(manifest_path)
        max_files = getattr(self.args, "max_files", 0)
        if max_files:
            entries = entries[:max_files]
        kept = skipped = 0
        for relpath, subtune in entries:
            built = self._build_blocks(relpath, subtune)
            if built is None:
                skipped += 1
                continue
            kept += 1
            yield (kind, *built)
        self.logger.info("%s corpus: kept %u, skipped %u", kind, kept, skipped)

    def iter_block_seqs(self):
        """Training (+ optional named eval) block sets."""
        yield from self._iter("train", self.args.manifest)
        for name, path in parse_eval_manifests(getattr(self.args, "eval_manifest", "")):
            yield from self._iter(name, path)

    def iter_predict_block_seqs(self):
        """Block sets for predict: the positional ``manifest_arg`` else ``--manifest``."""
        path = getattr(self.args, "manifest_arg", "") or self.args.manifest
        yield from self._iter("train", path)


def parse_corpus(args, logger):
    """CLI body: build (and cache) every tune's block array. Returns kept count."""
    corpus = Corpus(args, logger)
    kept = sum(1 for _ in corpus.iter_block_seqs())
    logger.info("parsed %u tune(s)", kept)
    return kept
