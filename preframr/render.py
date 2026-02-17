#!/usr/bin/env python3

import argparse
from preframr.args import add_args
from preframr.regdataset import RegDataset, get_prompt
from preframr.reglogparser import state_df, prepare_df_for_audio
from preframr.stfconstants import MODEL_PDTYPE
from preframr.sidwav import default_sid, sidq, write_samples
from preframr.utils import get_logger


def main():
    parser = add_args(argparse.ArgumentParser())
    args = parser.parse_args()
    if not args.reglog:
        raise ValueError("--reglog required")
    logger = get_logger("INFO")
    dataset = RegDataset(args, logger=logger)
    dataset.make_tokens(args.reglog)
    dataset.load()
    irq, _n, prompt, _prompt_compare, reg_start = get_prompt(args, dataset, logger)
    states = prompt.squeeze(0).tolist()
    prompt_df = state_df(dataset.tokenizer.decode(states), dataset, irq)
    if args.csv:
        prompt_df.astype(MODEL_PDTYPE).drop("n", axis=1).to_csv(args.csv, index=False)
    prompt_df, reg_widths = prepare_df_for_audio(
        prompt_df, dataset.reg_widths, irq, sidq()
    )
    write_samples(
        prompt_df,
        args.wav,
        reg_widths=reg_widths,
        asid=args.asid,
        sysex_delay=args.sysex_delay,
        reg_start=reg_start,
    )


if __name__ == "__main__":
    main()
