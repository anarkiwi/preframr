#!/usr/bin/env python3
"""Export an inference-only checkpoint: drop optimizer/training state and cast
float weights to bf16 (the precision predict runs at), ~6x smaller and lossless
for inference. Usage: ``python3 -m preframr.inference.export_weights IN OUT``."""

import argparse
import sys

import torch

_KEEP = (
    "state_dict",
    "hyper_parameters",
    "hparams_name",
    "pytorch-lightning_version",
    "epoch",
    "global_step",
)


def export_weights(in_path, out_path):
    with torch.serialization.safe_globals([argparse.Namespace]):
        ck = torch.load(in_path, weights_only=False, map_location="cpu")
    out = {k: ck[k] for k in _KEEP if k in ck}
    out["state_dict"] = {
        k: (v.to(torch.bfloat16) if torch.is_tensor(v) and v.is_floating_point() else v)
        for k, v in out.get("state_dict", {}).items()
    }
    torch.save(out, out_path)


def main():
    if len(sys.argv) != 3:
        print("usage: export_weights IN.ckpt OUT.ckpt", file=sys.stderr)
        sys.exit(2)
    export_weights(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
