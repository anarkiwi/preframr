#!/usr/bin/env python3

import argparse
import glob
import os
import sys

import pandas as pd
from torchtune.utils import get_logger
from torchtune.generation import generate
import torch
import torchmetrics

from args import add_args, MODEL_PRECISION
from model import get_device, Model
from regdataset import RegDataset, get_prompt
from reglogparser import state_df, prepare_df_for_audio
from sidwav import write_samples, sidq
from preframr.stfconstants import MODEL_PDTYPE, PAD_ID


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


def describe_cycles(cycles):
    return f"{int(cycles)} cycles {cycles*sidq():.2f} seconds"


def generate_sequence(args, logger, dataset, predictor):
    irq, n, prompt, prompt_compare, reg_start = get_prompt(args, dataset, logger)
    states = prompt.squeeze(0).tolist()
    decoded_prompt = dataset.tokenizer.decode(states)
    prompt_df = state_df(decoded_prompt, dataset, irq)
    prompt_df_audio, _reg_widths = prepare_df_for_audio(
        prompt_df, dataset.reg_widths, irq, sidq()
    )
    prompt_cycles = prompt_df_audio["diff"].sum()
    logger.info(
        "prompt lasts %s %u tokens (%u decoded tokens), predicting %u tokens",
        describe_cycles(prompt_cycles),
        args.prompt_seq_len,
        len(decoded_prompt),
        n,
    )

    predict_states = predictor.predict(
        prompt, n, temperature=args.temperature, top_k=args.top_k
    )
    states.extend(predict_states.tolist())
    df = state_df(dataset.tokenizer.decode(states), dataset, irq)
    predicted_compare = prompt_compare[args.prompt_seq_len :]
    f_acc = pd.NA
    if len(predicted_compare):
        acc = torchmetrics.functional.classification.multiclass_accuracy(
            predict_states[: len(predicted_compare)].to("cpu"),
            predicted_compare.to("cpu"),
            dataset.n_vocab,
        )
        acc = acc.item()
        f_acc = "%3.3f" % acc
    if args.csv:
        out_df = df.join(
            state_df(dataset.tokenizer.decode(prompt_compare.numpy()), dataset, irq),
            how="left",
            rsuffix="_p",
        )
        out_df["p_n"] = out_df["n"] == out_df["n_p"]
        out_df.astype(MODEL_PDTYPE).to_csv(args.csv, index=False)
    if args.min_acc:
        if acc is pd.NA or acc < args.min_acc:
            logger.error(f"{acc} below min_acc {args.min_acc}")
            sys.exit(-1)
    df, reg_widths = prepare_df_for_audio(df, dataset.reg_widths, irq, sidq())
    total_cycles = df["diff"].sum()
    generated_cycles = total_cycles - prompt_cycles
    logger.info(
        "generated %s accuracy %s, total %s",
        describe_cycles(generated_cycles),
        f_acc,
        describe_cycles(total_cycles),
    )
    write_samples(
        df,
        args.wav,
        reg_widths,
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
    # pylint: disable=no-value-for-parameter
    with torch.serialization.safe_globals([argparse.Namespace]):
        model = Model.load_from_checkpoint(ckpt, weights_only=False)
    dataset = RegDataset(args, logger=logger)
    dataset.preload(tokens=model.tokens, tkmodel=model.tkmodel)
    dataset.load()
    device, model_compiler = get_device(args, logger)
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
    dataset, _model, predictor = load_model(args, logger)
    generate_sequence(args, logger, dataset, predictor)


if __name__ == "__main__":
    main()
