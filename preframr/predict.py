#!/usr/bin/env python3

import argparse
import glob
import os
import random

import pandas as pd
from torchtune.utils import get_logger
from torchtune.generation import generate
import torch
import torchmetrics

from args import add_args, MODEL_PRECISION
from model import get_device, Model
from regdataset import RegDataset, MODEL_PDTYPE
from sidwav import write_samples, sidq
from preframr.stfconstants import FRAME_REG, PAD_ID


class Predictor:
    def __init__(self, args, dataset, model, device):
        self.args = args
        self.dataset = dataset
        self.model = model
        self.device = device
        self.rng = torch.Generator(device=self.device)
        self.rng.seed()

    @torch.inference_mode()
    def predict(self, prompt, n, temperature=1.0, top_k=None):
        output, _logits = generate(
            self.model.model,
            prompt.clone().to(self.device),
            max_generated_tokens=n,
            pad_id=PAD_ID,
            top_k=top_k,
            temperature=temperature,
            rng=self.rng,
        )
        return output.squeeze(0)[-n:]


def state_df(states, dataset, irq):
    tokens = dataset.tokens.copy()
    tokens.loc[tokens["reg"] == FRAME_REG, "diff"] = irq
    df = pd.DataFrame(states, columns=["n"]).merge(tokens, on="n", how="left")
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
    prompt_compare = seq[start:][: args.max_seq_len]
    irq = int(dataset.dfs[args.start_seq]["irq"].iat[start])
    preamble_df = state_df(dataset.decode(seq[:start]), dataset, irq)
    reg_start = {
        r: preamble_df[preamble_df["reg"] == r]["val"].iat[-1]
        for r in preamble_df["reg"].unique()
        if r >= 0
    }
    return irq, n, prompt, prompt_compare, reg_start


def generate_sequence(args, logger, dataset, model, predictor):
    irq, n, prompt, prompt_compare, reg_start = get_prompt(args, dataset, logger)
    states = prompt.squeeze(0).tolist()
    decoded_prompt = dataset.decode(states)
    prompt_df = state_df(decoded_prompt, dataset, irq)
    prompt_cycles = prompt_df["diff"].sum()
    logger.info(
        "prompt lasts %u cycles %.2f seconds %u tokens (%u decoded tokens), predicting %u tokens",
        prompt_cycles,
        prompt_cycles * sidq(),
        args.prompt_seq_len,
        len(decoded_prompt),
        n,
    )

    predict_states = predictor.predict(
        prompt, n, temperature=args.temperature, top_k=args.top_k
    )
    states.extend(predict_states.tolist())
    df = state_df(dataset.decode(states), dataset, irq)
    predicted_compare = prompt_compare[args.prompt_seq_len :]
    acc = pd.NA
    if len(predicted_compare):
        acc = torchmetrics.functional.classification.multiclass_accuracy(
            predict_states[: len(predicted_compare)].to("cpu"),
            predicted_compare.to("cpu"),
            dataset.n_vocab,
        )
        acc = "%3.3f" % acc
    cycles = df["diff"].sum() - prompt_cycles
    logger.info(
        "generated %9.u cycles %6.2f seconds accuracy %s",
        cycles,
        cycles * sidq(),
        acc,
    )
    cycles = df["diff"].sum()
    logger.info(
        "finalized %9.u total cycles %6.2f total seconds", cycles, cycles * sidq()
    )
    if args.csv:
        out_df = df.join(
            state_df(dataset.decode(prompt_compare.numpy()), dataset, irq),
            how="left",
            rsuffix="_p",
        )
        out_df["p_n"] = out_df["n"] == out_df["n_p"]
        out_df.astype(MODEL_PDTYPE).to_csv(args.csv, index=False)
    write_samples(
        df,
        args.wav,
        dataset.reg_widths,
        reg_start=reg_start,
        asid=args.asid,
        sysex_delay=args.sysex_delay,
    )


def get_ckpt(ckpt, tb_logs):
    if ckpt:
        return ckpt
    ckpts = sorted(
        [
            (os.path.getmtime(p), p)
            for p in glob.glob(f"{tb_logs}/**/*ckpt", recursive=True)
        ]
    )
    try:
        return ckpts[-1][1]
    except IndexError:
        raise IndexError("no checkpoint")


def load_model(args, logger):
    ckpt = get_ckpt(args.model_state, args.tb_logs)
    logger.info("loading %s", ckpt)
    dataset = RegDataset(args, logger=logger)
    dataset.load(train=False)
    device, model_compiler = get_device(args, logger)
    model = Model.load_from_checkpoint(ckpt)  # pylint: disable=no-value-for-parameter
    predict_precision = MODEL_PRECISION[args.model_precision]
    model = model.to(predict_precision)
    model.eval()
    model.model.eval()
    with device:
        model.model.setup_caches(
            batch_size=1,
            dtype=predict_precision,
            decoder_max_seq_len=args.max_seq_len,
        )
    model = model_compiler(args, model)
    model = model.to(device)
    predictor = model_compiler(args, Predictor)(
        args,
        dataset,
        model,
        device,
    )
    return dataset, model, predictor


def main():
    parser = add_args(argparse.ArgumentParser())
    args = parser.parse_args()
    logger = get_logger("INFO")
    dataset, model, predictor = load_model(args, logger)
    generate_sequence(args, logger, dataset, model, predictor)


if __name__ == "__main__":
    main()
