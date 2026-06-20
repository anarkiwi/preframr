"""BACC corpus builder: (.sid + .dump) pairs -> per-tune model-id block arrays. Replaces the deleted ``preframr_tokens.Corpus`` / ``parse_corpus``: each register ``.dump.parquet`` is paired with its sibling ``.sid`` (the codec recovers a program by running the playroutine white-box), only the Hubbard driver backends match, and tunes whose driver no backend recognises (or that fail to parse) are logged and skipped. Each kept tune is serialized to the fixed BACC alphabet, windowed into ``seq_len`` blocks (PAD-padded), and written next to the dump as ``<base>.blocks.npy``."""

import glob
import logging
import os
import re
from dataclasses import dataclass

import numpy as np
from preframr_tokens import cpf_from_meta, program_to_ids, recover_program

from preframr.tokenizer import PAD_ID, BaccTokenizer

DUMP_SUFFIX = ".dump.parquet"
BLOCKS_SUFFIX = ".blocks.npy"


@dataclass
class SeqMeta:
    """Per-tune metadata the predict/logging path reads off a block set: source dump path, frame clock (cycles/frame, carried into render timing), and subtune index."""

    df_file: str
    irq: float
    subtune: int


def _resolve_paths(dump):
    """``<name>.<subtune>.dump.parquet`` -> (sid, subtune_index, base_prefix). The dump filename is 1-indexed per subtune while ``recover_program``'s subtune is 0-indexed; the ``.sid`` carries no subtune component; ``base`` (dump minus ``.dump.parquet``) is the meta-sidecar prefix + blocks-path stem."""
    base = dump[: -len(DUMP_SUFFIX)] if dump.endswith(DUMP_SUFFIX) else dump
    match = re.match(r"^(.*)\.(\d+)$", base)
    if match:
        name, sub = match.group(1), int(match.group(2))
    else:
        name, sub = base, 1
    return name + ".sid", max(sub - 1, 0), base


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


class Corpus:
    """Owns the BACC tokenizer + the dump-glob -> block-array orchestration."""

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

    def _build_blocks(self, dump):
        """Recover + tokenize + window one tune; write ``<base>.blocks.npy``. Returns ``(blocks_path, seq_meta)`` or ``None`` if skipped (missing ``.sid``, no matching driver backend, or a parse failure). An existing ``.blocks.npy`` of the right width is reused, so a prior parse run (or a restored cache) skips the py65 recovery."""
        sid, subtune, base = _resolve_paths(dump)
        cpf = cpf_from_meta(base)
        seq_len = self.args.seq_len
        blocks_path = base + BLOCKS_SUFFIX
        cached = _load_cached_blocks(blocks_path, seq_len)
        if cached is not None:
            self.logger.info(
                "%s: reuse %s (%u blocks)",
                os.path.basename(dump),
                os.path.basename(blocks_path),
                cached.shape[0],
            )
            return blocks_path, SeqMeta(df_file=dump, irq=cpf, subtune=subtune)
        if not os.path.exists(sid):
            self.logger.info("skip %s: no sibling .sid (%s)", dump, sid)
            return None
        try:
            program = recover_program(sid, dump, cpf, subtune)
            ids = [i + 1 for i in program_to_ids(program)]
        except Exception as err:  # pylint: disable=broad-except
            self.logger.info("skip %s: %s", dump, err)
            return None
        stride = getattr(self.args, "block_stride", None) or seq_len
        blocks = _windows(ids, seq_len, stride)
        if not blocks:
            self.logger.info("skip %s: too short (%u ids)", dump, len(ids))
            return None
        arr = np.asarray(blocks, dtype=np.int16)
        np.save(blocks_path, arr)
        self.logger.info(
            "%s -> %s (%u ids, %u blocks of %u)",
            os.path.basename(dump),
            os.path.basename(blocks_path),
            len(ids),
            arr.shape[0],
            arr.shape[1],
        )
        return blocks_path, SeqMeta(df_file=dump, irq=cpf, subtune=subtune)

    def _iter(self, kind, pattern):
        """Yield ``(kind, blocks_path, seq_meta)`` for each kept dump in glob."""
        if not pattern:
            return
        dumps = sorted(glob.glob(pattern, recursive=True))
        max_files = getattr(self.args, "max_files", 0)
        if max_files:
            dumps = dumps[:max_files]
        kept = skipped = 0
        for dump in dumps:
            built = self._build_blocks(dump)
            if built is None:
                skipped += 1
                continue
            kept += 1
            yield (kind, *built)
        self.logger.info("%s corpus: kept %u, skipped %u", kind, kept, skipped)

    def iter_block_seqs(self):
        """Training (+ optional eval) block sets."""
        yield from self._iter("train", self.args.reglogs)
        eval_pat = getattr(self.args, "eval_reglogs", "")
        if eval_pat:
            yield from self._iter("val", eval_pat)

    def iter_predict_block_seqs(self):
        """Block sets for predict: the positional ``reglog`` else ``--reglogs``."""
        pattern = getattr(self.args, "reglog", "") or self.args.reglogs
        yield from self._iter("train", pattern)


def parse_corpus(args, logger):
    """CLI body: build (and cache) every tune's block array. Returns kept count."""
    corpus = Corpus(args, logger)
    kept = sum(1 for _ in corpus.iter_block_seqs())
    logger.info("parsed %u tune(s)", kept)
    return kept
