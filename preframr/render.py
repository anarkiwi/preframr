#!/usr/bin/env python3

import argparse
from torchtune.utils import get_logger
from regdataset import RegDataset, get_prompt
from reglogparser import state_df, prepare_df_for_audio
from args import add_args
from preframr.stfconstants import MODEL_PDTYPE
from sidwav import default_sid, sidq, write_samples


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
        prompt_df.astype(MODEL_PDTYPE).to_csv(args.csv, index=False)
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
