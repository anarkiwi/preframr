#!/usr/bin/env python3

import argparse
import random
import pandas as pd
from torchtune.utils import get_logger
from regdataset import RegDataset, MODEL_PDTYPE, state_df, get_prompt
from args import add_args
from preframr.stfconstants import FRAME_REG
from sidwav import write_samples


def main():
    parser = add_args(argparse.ArgumentParser())
    args = parser.parse_args()
    if not args.reglog:
        raise ValueError("--reglog required")
    logger = get_logger("INFO")
    dataset = RegDataset(args, logger=logger)
    dataset.load(tokens=None, tkmodel=None)
    irq, _n, prompt, _prompt_compare, reg_start = get_prompt(args, dataset, logger)
    states = prompt.squeeze(0).tolist()
    prompt_df = state_df(dataset.decode(states), dataset, irq)
    if args.csv:
        prompt_df.astype(MODEL_PDTYPE).to_csv(args.csv, index=False)
    write_samples(
        prompt_df,
        args.wav,
        dataset.reg_widths,
        asid=args.asid,
        sysex_delay=args.sysex_delay,
        reg_start=reg_start,
    )


if __name__ == "__main__":
    main()
