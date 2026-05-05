#!/usr/bin/env python
"""Measurement study for the proposed FuzzyLoopPass.

For a sample of Goto80 dumps, computes per-frame "musical fingerprints"
(quantized note + waveform + gate per voice, plus filter cutoff + volume
buckets), then finds candidate fingerprint matches, then measures the
byte-level overlay size needed to actually replay each candidate's
body. Reports the distribution of (match length, overlay size) so we
can decide whether FuzzyLoopPass would pay off or whether
per-pattern state drift dominates.

Run after parsing with ``--no-loop-pass`` so we see what LoopPass
itself missed.
"""
import argparse, glob, os, random, sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from preframr.args import add_args
from preframr.macros import _FastRow, _df_arrays_and_frames, DECODERS, DecodeState
from preframr.reglogparser import RegLogParser
from preframr.stfconstants import (
    DELAY_REG,
    FRAME_REG,
    MIN_DIFF,
    SET_OP,
    VOICE_REG_SIZE,
    VOICES,
)


def compute_fingerprints(df):
    """Walk df via ``_simulate_palette``-style dispatch, snapshot
    state.last_val at each frame's end, return a 64-bit musical
    fingerprint per logical frame. The walk mirrors the encoder's so
    gate/instrument palettes evolve identically and dispatches don't
    crash on cross-frame palette refs.

    Fingerprint fields: per voice (note bucket, waveform high nibble,
    gate bit) + global filter-cutoff high nibble + volume low nibble.
    """
    from preframr.macros import expand_loops, _build_decode_state
    literal_df = expand_loops(df.copy())
    state = _build_decode_state(literal_df)
    if state is None:
        return []
    state.instrument_palette_frozen = False
    if df.attrs.get("instrument_palette"):
        state.instrument_palette = list(df.attrs["instrument_palette"])
        state.instrument_palette_frozen = True

    arrs, frame_starts = _df_arrays_and_frames(literal_df)
    regs = arrs["reg"]
    vals = arrs["val"]
    ops = arrs["op"]
    subregs = arrs["subreg"]
    diffs = arrs["diff"]
    descs = arrs["description"]
    indices = arrs["Index"]
    n_total = len(literal_df)
    n_frames = len(frame_starts)
    description_default = 0
    fps = []
    out_frame_idx = 0
    for fi in range(n_frames):
        start = int(frame_starts[fi])
        end = int(frame_starts[fi + 1]) if fi + 1 < n_frames else n_total
        f_writes = []
        marker_reg = int(regs[start])
        marker_val = int(vals[start])
        marker_diff = int(diffs[start])
        if marker_reg == FRAME_REG:
            f_writes.append((marker_reg, marker_val, marker_diff, description_default))
        elif marker_reg == DELAY_REG:
            for _ in range(marker_val - 1):
                delay_writes = [
                    (FRAME_REG, 0, state.frame_diff, description_default)
                ]
                delay_writes.extend(state.tick_frame())
                state.observe_frame(delay_writes, frame_idx=out_frame_idx)
            f_writes.append((FRAME_REG, 0, state.frame_diff, description_default))
        for i in range(start + 1, end):
            reg = int(regs[i])
            if reg < 0:
                continue
            op = int(ops[i])
            decoder = DECODERS.get(op)
            if decoder is None:
                continue
            row = _FastRow(
                reg=reg, val=int(vals[i]), op=op, subreg=int(subregs[i]),
                diff=int(diffs[i]), description=int(descs[i]), Index=int(indices[i]),
            )
            writes = decoder.expand(row, state)
            if writes:
                f_writes.extend(writes)
        f_writes.extend(state.tick_frame())
        state.observe_frame(f_writes, frame_idx=out_frame_idx)
        out_frame_idx += 1
        # Build fingerprint from end-of-frame state.last_val.
        fp = 0
        for v in range(VOICES):
            base = v * VOICE_REG_SIZE
            ctrl = int(state.last_val.get(base + 4, 0))
            freq = int(state.last_val.get(base + 0, 0))
            gate = ctrl & 0x01
            wave = (ctrl & 0xF0) >> 4
            note = freq & 0xFF if gate else 0xFF
            fp = (fp << 13) | (note << 5) | (wave << 1) | gate
        cutoff = (int(state.last_val.get(22, 0)) >> 4) & 0x0F
        modevol = int(state.last_val.get(24, 0)) & 0x0F
        fp = (fp << 8) | (cutoff << 4) | modevol
        fps.append(fp)
    return fps


def compute_overlay(df_arrs, frame_starts, src_idx, dst_idx, length):
    """Count byte-level mismatches between two frame ranges of equal length.

    Returns (overlay_count, body_size).
    """
    regs = df_arrs["reg"]
    vals = df_arrs["val"]
    ops = df_arrs["op"]
    subregs = df_arrs["subreg"]
    overlay = 0
    body_size = 0
    n_frames = len(frame_starts)
    for k in range(length):
        s_lo = int(frame_starts[src_idx + k])
        s_hi = int(frame_starts[src_idx + k + 1]) if src_idx + k + 1 < n_frames else len(regs)
        d_lo = int(frame_starts[dst_idx + k])
        d_hi = int(frame_starts[dst_idx + k + 1]) if dst_idx + k + 1 < n_frames else len(regs)
        # Compare row-by-row up to common length; mismatched rows + length-diff = overlay.
        s_len = s_hi - s_lo
        d_len = d_hi - d_lo
        body_size += d_len
        common = min(s_len, d_len)
        for r in range(common):
            if (
                regs[s_lo + r] != regs[d_lo + r]
                or vals[s_lo + r] != vals[d_lo + r]
                or ops[s_lo + r] != ops[d_lo + r]
                or subregs[s_lo + r] != subregs[d_lo + r]
            ):
                overlay += 1
        overlay += abs(s_len - d_len)
    return overlay, body_size


def study_song(df, max_match=16, max_candidates_per_pos=8):
    """For each frame i, find the longest fingerprint match against any
    earlier i, compute its overlay, return (length, overlay, body_size)
    triples."""
    fps = compute_fingerprints(df)
    arrs, frame_starts = _df_arrays_and_frames(df)
    n_frames = len(frame_starts)
    if n_frames < 2:
        return []
    seed = defaultdict(list)
    out = []
    for i in range(n_frames - 1):
        key = (fps[i], fps[i + 1])
        cands = seed.get(key, [])
        # Find longest match among candidates.
        best_length = 0
        best_cand = -1
        for cand in cands[-max_candidates_per_pos:]:
            if cand >= i:
                continue
            length = 0
            while (
                length < max_match
                and i + length < n_frames
                and cand + length < i
                and fps[cand + length] == fps[i + length]
            ):
                length += 1
            if length > best_length:
                best_length, best_cand = length, cand
        if best_length >= 2:
            overlay, body_size = compute_overlay(
                arrs, frame_starts, best_cand, i, best_length
            )
            out.append((best_length, overlay, body_size))
        seed[key].append(i)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_args(ap)
    ap.add_argument("--dump-dir", default="/scratch/preframr/training-dumps/MUSICIANS/G/Goto80/")
    ap.add_argument("--sample", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.loop_pass = False  # study what fuzzy could catch BEYOND existing loop-pass

    files = sorted(glob.glob(os.path.join(args.dump_dir, "*.dump.parquet")))
    files = random.Random(args.seed).sample(files, min(args.sample, len(files)))

    parser = RegLogParser(args)
    all_triples = []
    per_file_stats = []
    for f in files:
        try:
            dfs = list(parser.parse(f, max_perm=1, require_pq=False, reparse=True))
        except Exception as e:
            print(f"{os.path.basename(f)}: parse failed: {e}", flush=True)
            continue
        if not dfs:
            print(f"{os.path.basename(f)}: filtered", flush=True)
            continue
        df = dfs[0]
        # Strip voice-rotation markers so absolute regs are restored;
        # decoders assume that form, and our fingerprint state tracking
        # depends on it.
        df, _ = parser._remove_voice_reg(df, {})
        try:
            triples = study_song(df)
        except Exception as e:
            print(f"{os.path.basename(f)}: study failed: {e}", flush=True)
            continue
        if not triples:
            print(f"{os.path.basename(f)}: no matches", flush=True)
            continue
        # Aggregate stats: of the matches, how many would pay off as
        # FuzzyLoopPass? cost = 1 + overlay; save = body_size - cost.
        saves = [(b - 1 - o) for (_, o, b) in triples]
        positive = [s for s in saves if s > 0]
        total_save = sum(positive)
        print(
            f"{os.path.basename(f)}: matches={len(triples)} "
            f"positive_save={len(positive)}/{len(triples)} "
            f"total_save={total_save} "
            f"max_save={max(saves) if saves else 0}",
            flush=True,
        )
        per_file_stats.append((os.path.basename(f), len(triples), len(positive), total_save))
        all_triples.extend(triples)

    print()
    print("=== Aggregate ===")
    print(f"total candidate matches: {len(all_triples)}")
    if all_triples:
        # Distribution of overlay/body ratio.
        ratios = [o / max(1, b) for (_, o, b) in all_triples]
        for thresh in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
            cnt = sum(1 for r in ratios if r <= thresh)
            print(
                f"  overlay/body <= {thresh}: {cnt} ({100*cnt/len(ratios):.1f}%)"
            )
        # Length distribution.
        lengths = [l for (l, _, _) in all_triples]
        print(f"  match length avg={sum(lengths)/len(lengths):.1f} max={max(lengths)}")
        # Total potential save.
        total_save = sum(max(0, b - 1 - o) for (_, o, b) in all_triples)
        total_body = sum(b for (_, _, b) in all_triples)
        print(f"  total potential save: {total_save} / total body {total_body} = {100*total_save/max(1,total_body):.1f}%")


if __name__ == "__main__":
    sys.exit(main())
