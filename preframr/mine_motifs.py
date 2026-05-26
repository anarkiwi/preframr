#!/usr/bin/env python
"""Thin CLI shim around preframr_tokens.mine_dict_from_dumps. Parses the corpus
with the run's pipeline/macro args (motif pass forced off) and writes a
motif_dict.json artifact for a later --motif-pass tokenize/train. Reuses add_args
so the parse exactly matches training; only --motif-out and the mining knobs are
mine-specific."""

import argparse

from preframr_tokens import mine_dict_from_dumps

from preframr.args import add_args, apply_pipeline_spec_to_args
from preframr.utils import get_logger


def main():
    parser = add_args(argparse.ArgumentParser())
    parser.add_argument("--motif-out", type=str, required=True)
    parser.add_argument("--motif-k", type=int, default=256)
    parser.add_argument("--motif-min-count", type=int, default=3)
    parser.add_argument("--motif-min-composers", type=int, default=3)
    args = parser.parse_args()
    apply_pipeline_spec_to_args(args)
    logger = get_logger("INFO")
    motif_dict = mine_dict_from_dumps(
        args,
        args.reglogs,
        max_files=args.max_files,
        k=args.motif_k,
        min_count=args.motif_min_count,
        min_composers=args.motif_min_composers,
        logger=logger,
    )
    with open(args.motif_out, "w") as f:
        f.write(motif_dict.to_json())
    logger.info("wrote motif dict (%u motifs) to %s", len(motif_dict), args.motif_out)


if __name__ == "__main__":
    main()
