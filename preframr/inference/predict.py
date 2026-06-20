#!/usr/bin/env python3
"""Generate a BACC continuation from a checkpoint and render it to a WAV. A prompt is the first ``prompt_seq_len`` model ids of a selected block; the model continues it. BACC ids carry a program header up front, so a generated stream is not guaranteed to be a decodable program: the render tries the full generated stream first and falls back to the ground-truth tune so the pipeline always produces audio, and token accuracy vs the ground-truth tail is reported either way."""

import argparse
import copy
import csv as _csv
import sys

import pyarrow

pyarrow.PyExtensionType = pyarrow.ExtensionType
from torchtune.generation import generate
import torch
import torchmetrics

from preframr.args import add_args, MODEL_PRECISION
from preframr.inference.event_render import render_ids_to_wav, state_to_dump_df
from preframr.inference.predict_lib import add_ext, get_ckpt
from preframr.train.model import get_device, Model
from preframr.train.regdataset import RegDataset, get_prompt
from preframr.tokenizer import PAD_ID
from preframr.utils import get_logger


class Predictor:
    def __init__(self, args, dataset, model, device, logger=None):
        self.args = args
        self.dataset = dataset
        self.model = model
        self.device = device
        self.rng = torch.Generator(device=self.device)
        self.rng.seed()
        self.logger = logger

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


def _accuracy(predict_states, predicted_compare, n_vocab):
    if not len(predicted_compare):
        return None
    acc = torchmetrics.functional.classification.multiclass_accuracy(
        predict_states[: len(predicted_compare)].to("cpu"),
        predicted_compare.to("cpu"),
        n_vocab,
    )
    return float(acc)


def _render_truth(seq_meta, irq, wav, chip):
    """Render the source tune (re-recovered from its dump) to ``wav``; a training block is a windowed slice of the program id stream and not itself a decodable program, so the always-available ground truth is the tune's own (.sid, .dump) pair."""
    from preframr_tokens import (  # pylint: disable=import-outside-toplevel
        recover_program,
        render_program,
    )

    from preframr.corpus import (
        _resolve_paths,
    )  # pylint: disable=import-outside-toplevel
    from preframr.inference.event_render import (  # pylint: disable=import-outside-toplevel
        render_state_to_wav,
    )

    sid, subtune, _base = _resolve_paths(seq_meta.df_file)
    program = recover_program(sid, seq_meta.df_file, irq, subtune)
    return render_state_to_wav(render_program(program), irq, wav, chip)


def generate_sequence(args, logger, dataset, predictor, p):
    irq, n, prompt, prompt_compare, seq_meta = get_prompt(args, dataset, logger)
    predict_states = predictor.predict(
        prompt, n, temperature=args.temperature, top_k=args.top_k
    )
    prompt_ids = prompt.squeeze(0).tolist()
    full_ids = prompt_ids + predict_states.tolist()

    predicted_compare = prompt_compare[args.prompt_seq_len :]
    acc = _accuracy(predict_states, predicted_compare, dataset.n_vocab)
    logger.info(
        "predicted %u tokens (prompt %u), accuracy %s",
        n,
        args.prompt_seq_len,
        "n/a" if acc is None else f"{acc:.3f}",
    )
    if args.min_acc and (acc is None or acc < args.min_acc):
        logger.error("%s below min_acc %s", acc, args.min_acc)
        sys.exit(-1)

    if args.csv:
        with open(add_ext(args.csv, p), "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["i", "gt", "pred"])
            gt = prompt_compare.tolist()
            for i, pid in enumerate(full_ids):
                w.writerow([i, gt[i] if i < len(gt) else "", pid])

    wav = add_ext(args.wav, p)
    try:
        samples = render_ids_to_wav(full_ids, irq, wav, _chip(args))
        logger.info("rendered generation -> %s (%u samples)", wav, samples)
    except Exception as err:  # pylint: disable=broad-except
        logger.error("generation not a decodable program (%s); rendering truth", err)
        samples = _render_truth(seq_meta, irq, wav, _chip(args))
        logger.info("rendered ground-truth tune -> %s (%u samples)", wav, samples)

    if getattr(args, "predict_dump", None):
        _write_predict_dump(args, logger, full_ids, irq, p)
    if getattr(args, "play", False):
        logger.warning(
            "--play is not supported in the BACC render path; wrote wav only"
        )


def _chip(args):
    return getattr(args, "chip_model", "MOS8580")


def _write_predict_dump(args, logger, full_ids, irq, p):
    from preframr_tokens import (  # pylint: disable=import-outside-toplevel
        VOCAB,
        ids_to_program,
        render_program,
    )

    try:
        prog_ids = [int(i) - 1 for i in full_ids if 1 <= int(i) <= VOCAB]
        state = render_program(ids_to_program(prog_ids))
    except Exception as err:  # pylint: disable=broad-except
        logger.error("predict-dump skipped (undecodable generation): %s", err)
        return
    dump_path = add_ext(args.predict_dump, p)
    df = state_to_dump_df(state, irq)
    df.attrs["irq"] = int(irq)
    df.to_parquet(dump_path)
    logger.info("wrote prediction dump (%u rows) to %s", len(df), dump_path)


def load_model(args, logger):
    ckpt = get_ckpt(args.model_state, args.tb_logs)
    logger.info("loading %s", ckpt)
    with torch.serialization.safe_globals([argparse.Namespace]):
        model = Model.load_from_checkpoint(ckpt, weights_only=False, map_location="cpu")
    dataset = RegDataset(args, logger=logger)
    dataset.preload(tokens=model.tokens, tkmodel=model.tkmodel)
    dataset.predict_load()
    device, model_compiler = get_device(args, logger)
    model.eval()
    model.model.eval()
    predict_precision = MODEL_PRECISION[args.model_precision]
    model = model.to(predict_precision)
    with device:
        model.model.setup_caches(
            batch_size=1,
            dtype=predict_precision,
            decoder_max_seq_len=args.max_seq_len,
        )
    if getattr(args, "compile", True) and device.type == "cuda":
        model.model = torch.compile(
            model.model,
            options={"epilogue_fusion": True, "triton.cudagraphs": True},
        )
    else:
        model = model_compiler(args, model)
    return dataset, model, device, model_compiler


def run_predict(args, logger, dataset, model, device, model_compiler, p):
    model = model.to(device)
    predictor = model_compiler(args, Predictor)(
        args, dataset, model, device, logger=logger
    )
    generate_sequence(args, logger, dataset, predictor, p)
    model.cpu()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = add_args(argparse.ArgumentParser())
    args = parser.parse_args()
    logger = get_logger("INFO")
    dataset, model, device, model_compiler = load_model(args, logger)
    for p in range(args.predictions):
        m = copy.deepcopy(model) if args.predictions > 1 else model
        run_predict(args, logger, dataset, m, device, model_compiler, p)


if __name__ == "__main__":
    main()
