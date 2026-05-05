#!/usr/bin/env python
"""Identify tokens whose decoded expansion writes a register's current value.

For each *.dump.parquet under the requested directory:

1. Run the file through ``RegLogParser.parse`` (one rotation, no PQ cache)
   to obtain the encoded token stream the LM would see.
2. Walk that stream through the same per-row dispatch as
   ``RegLogParser._expand_ops`` -- including DO_LOOP / BACK_REF expansion,
   ``DecodeState.tick_frame`` calls, and DELAY_REG-induced empty frames --
   while tracking the running value of every SID register.
3. For every emitted ``(reg, val)`` write, report the source token index
   when ``val`` already equals the register's tracked value.

Tick-driven writes (REPEAT/FLIP/PWM/FILTER_SWEEP/FLIP2/INTERVAL bursts and
deferred subreg flushes) are attributed to the frame-marker row whose
``tick_frame()`` produced them; the original scheduling token's index is
also recorded where unambiguous.
"""

import argparse
import glob
import os
import sys
from collections import defaultdict

import pandas as pd

from preframr.args import add_args
from preframr import macros
from preframr.macros import (
    DECODERS,
    DecodeState,
    _FastRow,
    _deserialize_gate_palette,
    _df_arrays_and_frames,
)
from preframr.reglogparser import RegLogParser
from preframr.stfconstants import (
    DELAY_REG,
    FRAME_REG,
    MIN_DIFF,
    MODEL_PDTYPE,
    SET_OP,
)


def _bootstrap_last_diff(df):
    last_diff = {}
    for reg in df["reg"].unique():
        reg_df = df[(df["reg"] == reg) & (df["op"] == SET_OP)]["diff"]
        last_diff[reg] = int(reg_df.iloc[0]) if len(reg_df) else MIN_DIFF
    return last_diff


def find_redundant_writes(token_df):
    """Return a list of redundancy event dicts for ``token_df``."""
    loader = RegLogParser()
    df = token_df.copy()
    # The encoded form carries VOICE_REG rows; the decoders expect per-row
    # absolute regs, so undo that the same way prepare_df_for_audio does.
    df, _ = loader._remove_voice_reg(df, {})
    df = macros.expand_loops(df)
    if df.empty:
        return []

    last_diff = _bootstrap_last_diff(df)
    frame_diff_rows = df[df["reg"] == FRAME_REG]["diff"]
    if frame_diff_rows.empty:
        return []
    frame_diff = int(frame_diff_rows.iloc[0])

    state = DecodeState(
        frame_diff,
        last_diff=last_diff,
        strict=False,
        frozen_instrument_palette=token_df.attrs.get("instrument_palette"),
        frozen_gate_palette=_deserialize_gate_palette(
            token_df.attrs.get("gate_palette")
        ),
    )

    df = df.copy()
    if "description" not in df.columns:
        df["description"] = 0

    cur_reg_val = {}
    redundancies = []

    def record(writes, src_idx, src_label):
        if not writes:
            return
        for w in writes:
            wreg = int(w[0])
            wval = int(w[1])
            if wreg < 0:
                continue
            if wreg in cur_reg_val and cur_reg_val[wreg] == wval:
                redundancies.append(
                    {
                        "token_idx": src_idx,
                        "source": src_label,
                        "write_reg": wreg,
                        "write_val": wval,
                    }
                )
            cur_reg_val[wreg] = wval

    arrs, frame_starts = _df_arrays_and_frames(df)
    regs = arrs["reg"]
    vals = arrs["val"]
    ops = arrs["op"]
    subregs = arrs["subreg"]
    diffs = arrs["diff"]
    descs = arrs["description"]
    indices = arrs["Index"]
    n_total = len(df)
    n_frames = len(frame_starts)

    for fi in range(n_frames):
        start = int(frame_starts[fi])
        end = int(frame_starts[fi + 1]) if fi + 1 < n_frames else n_total
        cur_frame = fi
        f_writes_for_observe = []
        # First row of the frame is the FRAME_REG/DELAY_REG marker.
        marker_reg = int(regs[start])
        marker_val = int(vals[start])
        marker_idx = int(indices[start])
        if marker_reg == FRAME_REG:
            pass
        elif marker_reg == DELAY_REG:
            for _ in range(marker_val - 1):
                tick = state.tick_frame()
                record(tick, marker_idx, "delay-tick")
                state.observe_frame(tick, frame_idx=cur_frame)
        else:
            raise AssertionError(f"unknown negative reg {marker_reg}")
        for i in range(start + 1, end):
            reg = int(regs[i])
            assert reg >= 0, (i, reg)
            op = int(ops[i])
            decoder = DECODERS.get(op)
            assert decoder is not None, f"unknown op {op}"
            row = _FastRow(
                reg=reg,
                val=int(vals[i]),
                op=op,
                subreg=int(subregs[i]),
                diff=int(diffs[i]),
                description=int(descs[i]),
                Index=int(indices[i]),
            )
            writes = decoder.expand(row, state)
            label = f"op={op} reg={reg} val={int(vals[i])}"
            record(writes, int(indices[i]), label)
            if writes:
                f_writes_for_observe.extend(writes)
        tick = state.tick_frame()
        record(tick, None, "frame-tick")
        if tick:
            f_writes_for_observe.extend(tick)
        state.observe_frame(f_writes_for_observe, frame_idx=cur_frame)

    return redundancies


def parse_one(parser, dump_path, reparse=False):
    """Parse one dump file and return its rotation DataFrames.

    By default this uses the pre-parsed ``*.N.parquet`` cache that
    ``preframr.parse`` writes alongside each dump (much faster than a fresh
    parse). Pass ``reparse=True`` to force the full pipeline.
    """
    dfs = list(parser.parse(dump_path, max_perm=1, require_pq=False, reparse=reparse))
    return dfs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_args(ap)
    ap.add_argument(
        "--dump-dir",
        default="/scratch/preframr/training-dumps/MUSICIANS/J/Jammer",
        help="Directory containing *.dump.parquet files to scan.",
    )
    ap.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print per-file totals, not individual events.",
    )
    ap.add_argument(
        "--reparse",
        action="store_true",
        help=(
            "Re-run the full parse pipeline from each dump (slow). Default "
            "uses the cached pre-parsed *.N.parquet siblings."
        ),
    )
    ap.add_argument(
        "--sample",
        type=int,
        default=0,
        help=(
            "Sample N random files instead of scanning the whole directory. "
            "0 (default) means scan everything."
        ),
    )
    ap.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help="Random seed for --sample selection.",
    )
    ap.add_argument(
        "--report-every",
        type=int,
        default=10,
        help="Print rolling totals every N files.",
    )
    args = ap.parse_args()

    import random
    dump_files = sorted(glob.glob(os.path.join(args.dump_dir, "*.dump.parquet")))
    if not dump_files:
        print(f"no dump files under {args.dump_dir}", file=sys.stderr)
        return 1

    if args.sample and args.sample < len(dump_files):
        rng = random.Random(args.sample_seed)
        dump_files = rng.sample(dump_files, args.sample)
        print(
            f"[sample] {args.sample} of {len(dump_files)} files "
            f"(seed={args.sample_seed})",
            flush=True,
        )

    parser = RegLogParser(args)
    grand_total = 0
    files_with_redundancy = 0
    files_scanned = 0
    files_filtered = 0
    total_tokens = 0

    def report_rolling():
        usable = files_scanned - files_filtered
        print(
            f"[rolling] scanned={files_scanned} usable={usable} "
            f"redundant_files={files_with_redundancy} "
            f"redundant_writes={grand_total} total_tokens={total_tokens}",
            flush=True,
        )

    for dump in dump_files:
        files_scanned += 1
        try:
            dfs = parse_one(parser, dump, reparse=args.reparse)
        except Exception as e:
            print(f"{dump}: parse failed: {e}", file=sys.stderr, flush=True)
            files_filtered += 1
            if files_scanned % args.report_every == 0:
                report_rolling()
            continue
        if not dfs:
            print(f"{dump}: filtered out by parser (no usable rotations)", flush=True)
            files_filtered += 1
            if files_scanned % args.report_every == 0:
                report_rolling()
            continue
        token_df = dfs[0]
        try:
            events = find_redundant_writes(token_df)
        except Exception as e:
            print(f"{dump}: scan failed: {e}", file=sys.stderr, flush=True)
            files_filtered += 1
            if files_scanned % args.report_every == 0:
                report_rolling()
            continue
        total_tokens += len(token_df)
        if not events:
            print(f"{dump}: 0 redundant writes  (tokens={len(token_df)})", flush=True)
            if files_scanned % args.report_every == 0:
                report_rolling()
            continue
        files_with_redundancy += 1
        grand_total += len(events)
        print(
            f"{dump}: {len(events)} redundant writes  "
            f"(tokens={len(token_df)})",
            flush=True,
        )
        if files_scanned % args.report_every == 0:
            report_rolling()
        if args.summary_only:
            continue
        per_reg = defaultdict(int)
        per_source = defaultdict(int)
        for ev in events:
            per_reg[ev["write_reg"]] += 1
            per_source[ev["source"].split()[0]] += 1
        print("  by reg : " + ", ".join(
            f"{r}={n}" for r, n in sorted(per_reg.items())
        ))
        print("  by src : " + ", ".join(
            f"{s}={n}" for s, n in sorted(per_source.items())
        ))
        head = events[:10]
        for ev in head:
            print(
                f"  token_idx={ev['token_idx']} write_reg={ev['write_reg']} "
                f"write_val={ev['write_val']} source=<{ev['source']}>"
            )
        if len(events) > len(head):
            print(f"  ... ({len(events) - len(head)} more)")

    print()
    print(
        f"TOTAL: {grand_total} redundant writes across "
        f"{files_with_redundancy}/{len(dump_files)} files"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
