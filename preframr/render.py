#!/usr/bin/env python3

import argparse
import random
import pandas as pd
from torchtune.utils import get_logger
from regdataset import RegDataset, MODEL_PDTYPE
from args import add_args
from preframr.stfconstants import FRAME_REG
from sidwav import write_samples


def state_df(states, dataset, irq):
    df = pd.DataFrame(states, columns=["n"]).merge(dataset.tokens, on="n", how="left")
    return df


def get_prompt(args, dataset, logger):
    seq = dataset.getseq(args.start_seq)
    if args.start_n is None:
        start = random.randint(0, len(seq))
    else:
        start = args.start_n
    logger.info("starting at %u / %u", start, len(seq))
    n = args.max_seq_len - args.prompt_seq_len
    if n <= 0:
        raise ValueError("max seq length too short")
    prompt = seq[start:][: args.prompt_seq_len].unsqueeze(0)
    irq = int(dataset.dfs[args.start_seq].iloc[start]["irq"])
    return irq, n, prompt


def main():
    parser = add_args(argparse.ArgumentParser())
    args = parser.parse_args()
    logger = get_logger("INFO")
    dataset = RegDataset(args, logger=logger)
    dataset.load(train=False)
    irq, _n, prompt = get_prompt(args, dataset, logger)
    states = prompt.squeeze(0).tolist()
    decoded_prompt = dataset.decode(states)
    prompt_df = state_df(decoded_prompt, dataset, irq)
    if args.csv:
        prompt_df.astype(MODEL_PDTYPE).to_csv(args.csv, index=False)
    write_samples(prompt_df, args.wav, dataset.reg_widths)


if __name__ == "__main__":
    main()
