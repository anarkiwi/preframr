#!/usr/bin/env python3
"""Event-native generate+decode gate: load a checkpoint, greedily continue
event-token prompts from ``.blocks.npy``, decode generated ids through
``preframr_tokens.events.generate`` (strict grammar parser), report mean
greedy token-accuracy + grammar-clean rate, exit non-zero below threshold.
The event-model replacement for ``predict.py``'s old get_prompt/_state_df."""

import argparse
import glob
import json
import sys

import numpy as np
import pyarrow

pyarrow.PyExtensionType = pyarrow.ExtensionType
import torch

from preframr.args import add_args
from preframr.inference.predict import Predictor, load_model
from preframr.utils import get_logger
from preframr_tokens.events.generate import tokens_to_writes


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
    """First qualifying block per file (deduped songs), up to ``n_prompts``."""
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


def decode_tolerant(tokenizer, ids, min_keep):
    """Largest frame-aligned decodable prefix of ``ids`` that yields non-empty
    writes. A fixed-length generation ends mid-frame, and stream boundaries are
    sparse, so trim back to the last whole frame; ``stream.decode`` raises or
    returns empty on a mid-frame cut, so skip both. Returns (writes, trimmed,
    err); writes is None if nothing past ``min_keep`` decodes."""
    last_err = ""
    for trim in range(len(ids) - min_keep + 1):
        cut = ids if trim == 0 else ids[:-trim]
        try:
            writes = tokens_to_writes(tokenizer, cut)
        except Exception as exc:  # pylint: disable=broad-except
            last_err = f"{type(exc).__name__}: {exc}"
            continue
        if writes:
            return writes, trim, ""
    return None, len(ids) - min_keep, last_err


def run_gate(args, logger):
    dataset, model, device, _ = load_model(args, logger)
    model = model.to(device)
    predictor = Predictor(
        args, dataset, model, device, vocab_arrays=None, logger=logger
    )
    tokenizer = dataset.tokenizer
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
        writes, trim, err = decode_tolerant(tokenizer, full, len(prompt_ids))
        grammar_ok = writes is not None and len(writes) > 0
        decoded_gen = max(len(full) - trim - len(prompt_ids), 0)
        gen_frac = decoded_gen / max(len(gen[: len(truth)]), 1)
        logger.info(
            "prompt=%s greedy_acc=%.3f grammar_ok=%s decoded_gen_frac=%.3f trim=%u %s",
            path,
            acc,
            grammar_ok,
            gen_frac,
            trim,
            err,
        )
        per_prompt.append(
            {
                "path": path,
                "acc": acc,
                "grammar_ok": grammar_ok,
                "decoded_gen_frac": gen_frac,
                "trim": trim,
                "err": err,
            }
        )

    mean_acc = sum(p["acc"] for p in per_prompt) / len(per_prompt)
    fully_clean_rate = sum(
        1 for p in per_prompt if p["decoded_gen_frac"] >= 0.999
    ) / len(per_prompt)
    mean_decoded_gen_frac = sum(p["decoded_gen_frac"] for p in per_prompt) / len(
        per_prompt
    )
    passed = mean_acc >= args.min_acc and mean_decoded_gen_frac >= args.grammar_min
    result = {
        "passed": passed,
        "mean_greedy_acc": mean_acc,
        "mean_decoded_gen_frac": mean_decoded_gen_frac,
        "fully_clean_rate": fully_clean_rate,
        "n_prompts": len(per_prompt),
        "min_acc": args.min_acc,
        "grammar_min": args.grammar_min,
        "per_prompt": per_prompt,
    }
    print("EVENT_GATE_RESULT " + json.dumps(result))
    logger.info(
        "gate %s: mean_greedy_acc=%.3f mean_decoded_gen_frac=%.3f fully_clean_rate=%.3f",
        "PASS" if passed else "FAIL",
        mean_acc,
        mean_decoded_gen_frac,
        fully_clean_rate,
    )
    return 0 if passed else 1


def main():
    parser = add_gate_args(add_args(argparse.ArgumentParser()))
    args = parser.parse_args()
    logger = get_logger("INFO")
    sys.exit(run_gate(args, logger))


if __name__ == "__main__":
    main()
