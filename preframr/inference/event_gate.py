#!/usr/bin/env python3
"""BACC generate+decode gate: load a checkpoint, continue token prompts from
``.blocks.npy``, and report mean greedy token-accuracy + the fraction of
generations that decode to a valid BACC program (``ids_to_program`` round-trip).
Exit non-zero below threshold. The BACC replacement for the old event grammar
gate."""

import argparse
import glob
import json
import sys

import numpy as np
import pyarrow

pyarrow.PyExtensionType = pyarrow.ExtensionType
import torch
from preframr_tokens import VOCAB, ids_to_program

from preframr.args import add_args
from preframr.inference.predict import Predictor, load_model
from preframr.utils import get_logger


def add_gate_args(parser):
    parser.add_argument(
        "--blocks-glob",
        type=str,
        default="/scratch/preframr/train/**/*.blocks.npy",
    )
    parser.add_argument("--n-prompts", type=int, default=8)
    parser.add_argument("--gen-tokens", type=int, default=256)
    parser.add_argument("--grammar-min", type=float, default=0.5)
    return parser


def load_prompts(blocks_glob, n_prompts, prompt_seq_len, gen_tokens, logger):
    """First qualifying block per file, up to ``n_prompts``: the first
    ``prompt_seq_len`` non-PAD ids as the prompt, the next ``gen_tokens`` as the
    ground-truth continuation."""
    prompts = []
    for path in sorted(glob.glob(blocks_glob, recursive=True)):
        arr = np.load(path)
        for row in np.atleast_2d(arr):
            nonzero = row[row > 0]
            if len(nonzero) < prompt_seq_len + 1:
                continue
            prompt = nonzero[:prompt_seq_len].astype(np.int64)
            truth = nonzero[prompt_seq_len : prompt_seq_len + gen_tokens].astype(
                np.int64
            )
            if len(truth):
                prompts.append((path, prompt, truth))
                break
        if len(prompts) >= n_prompts:
            break
    logger.info("loaded %u prompts from %s", len(prompts), blocks_glob)
    return prompts


def decodes_to_program(ids):
    """True iff ``ids`` (model-space) round-trips to a valid BACC program."""
    prog_ids = [int(i) - 1 for i in ids if 1 <= int(i) <= VOCAB]
    try:
        ids_to_program(prog_ids)
        return True
    except Exception:  # pylint: disable=broad-except
        return False


def run_gate(args, logger):
    dataset, model, device, _ = load_model(args, logger)
    model = model.to(device)
    predictor = Predictor(args, dataset, model, device, logger=logger)
    prompts = load_prompts(
        args.blocks_glob, args.n_prompts, args.prompt_seq_len, args.gen_tokens, logger
    )
    if not prompts:
        print(
            "EVENT_GATE_RESULT " + json.dumps({"passed": False, "reason": "no prompts"})
        )
        return 1

    per_prompt = []
    for path, prompt_ids, truth in prompts:
        if model.model.caches_are_enabled():
            model.model.reset_caches()
        prompt = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0)
        gen = (
            predictor.predict(prompt, len(truth), temperature=1.0, top_k=1)
            .cpu()
            .numpy()
            .astype(np.int64)
        )
        overlap = min(len(gen), len(truth))
        acc = float((gen[:overlap] == truth[:overlap]).mean()) if overlap else 0.0
        full = np.concatenate([prompt_ids, gen[: len(truth)]]).tolist()
        decodes = decodes_to_program(full)
        logger.info("prompt=%s greedy_acc=%.3f decodes=%s", path, acc, decodes)
        per_prompt.append({"path": path, "acc": acc, "decodes": decodes})

    mean_acc = sum(p["acc"] for p in per_prompt) / len(per_prompt)
    decode_rate = sum(1 for p in per_prompt if p["decodes"]) / len(per_prompt)
    passed = mean_acc >= args.min_acc and decode_rate >= args.grammar_min
    result = {
        "passed": passed,
        "mean_greedy_acc": mean_acc,
        "decode_rate": decode_rate,
        "n_prompts": len(per_prompt),
        "min_acc": args.min_acc,
        "grammar_min": args.grammar_min,
        "per_prompt": per_prompt,
    }
    print("EVENT_GATE_RESULT " + json.dumps(result))
    logger.info(
        "gate %s: mean_greedy_acc=%.3f decode_rate=%.3f",
        "PASS" if passed else "FAIL",
        mean_acc,
        decode_rate,
    )
    return 0 if passed else 1


def main():
    parser = add_gate_args(add_args(argparse.ArgumentParser()))
    args = parser.parse_args()
    logger = get_logger("INFO")
    sys.exit(run_gate(args, logger))


if __name__ == "__main__":
    main()
