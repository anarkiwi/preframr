#!/usr/bin/env python3

import argparse
import copy
import glob
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow

pyarrow.PyExtensionType = pyarrow.ExtensionType
from torchtune.generation import generate, sample as _tt_sample
import torch
import torchmetrics

from preframr.args import add_args, MODEL_PRECISION
from preframr.constrained_decode import (
    StreamState,
    _frame_marker_count,
    precompute_vocab_arrays,
)
from preframr.macros import validate_back_refs, validate_gate_replays
from preframr.model import get_device, Model
from preframr.regdataset import RegDataset, get_prompt
from preframr.reglogparser import RegLogParser, prepare_df_for_audio
from preframr.sidwav import write_samples, sidq
from preframr.stfconstants import MIN_DIFF, MODEL_PDTYPE, PAD_ID
from preframr.utils import get_logger


class Predictor:
    def __init__(self, args, dataset, model, device, vocab_arrays=None, logger=None):
        self.args = args
        self.dataset = dataset
        self.model = model
        self.device = device
        self.rng = torch.Generator(device=self.device)
        self.rng.seed()
        self.vocab_arrays = vocab_arrays
        self.logger = logger

    @torch.inference_mode()
    def predict(self, prompt, n, temperature=1.0, top_k=None, irq=None):
        if self.vocab_arrays is None:
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
        return self._predict_constrained(prompt, n, temperature, top_k, irq)

    @torch.inference_mode()
    def _predict_constrained(self, prompt, n, temperature, top_k, irq):
        # Forked from torchtune.generation.generate so we can mask logits
        # uniformly on every step (including the first), not just inside
        # the custom_generate_next_token loop. KV cache + causal mask
        # bookkeeping mirrors the upstream incremental-decoding path.
        model = self.model.model
        prompt = prompt.clone().to(self.device)
        prompt = prompt.view(1, -1) if prompt.ndim == 1 else prompt
        bsz, prompt_length = prompt.size()
        total_len = prompt_length + n
        max_seq_len = (
            total_len
            if not model.caches_are_enabled()
            else model.decoder_max_cache_seq_len
        )

        masks = torch.tril(
            torch.ones(total_len, max_seq_len, dtype=torch.bool, device=self.device)
        ).unsqueeze(0)
        input_pos = torch.arange(0, total_len, device=self.device).unsqueeze(0)

        prompt_ids = prompt.squeeze(0).tolist()
        is_frame_marker_np = self.vocab_arrays["is_frame_marker"].cpu().numpy()
        is_real_reg_np = self.vocab_arrays["is_real_reg"].cpu().numpy()
        prompt_frames = _frame_marker_count(prompt_ids, is_frame_marker_np)
        # Compute the budget remaining at the end of the prompt by walking
        # back from the last frame marker, charging MIN_DIFF per real-reg
        # row. Falls back to the full IRQ window if the prompt has no
        # marker (very short prompt).
        init_budget = irq
        prompt_arr = np.asarray(prompt_ids, dtype=np.int64)
        marker_positions = np.nonzero(is_frame_marker_np[prompt_arr])[0]
        if marker_positions.size:
            tail = prompt_arr[marker_positions[-1] + 1 :]
            charged = int(is_real_reg_np[tail].sum() * MIN_DIFF)
            init_budget = max(irq - charged, 0)
        state = StreamState(
            self.vocab_arrays,
            init_frame_count=prompt_frames,
            irq=irq,
            init_budget=init_budget,
            logger=self.logger,
        )

        def _q():
            return torch.empty(
                (bsz, model.tok_embeddings.num_embeddings), device=self.device
            ).exponential_(1, generator=self.rng)

        def _step(curr_input_pos, curr_masks, x):
            logits = model(x, input_pos=curr_input_pos, mask=curr_masks)[:, -1]
            masked = state.mask_logits(logits)
            return _tt_sample(
                masked.clone(), temperature=temperature, top_k=top_k, q=_q()
            )

        # Step 0: forward the whole prompt, sample token after it.
        if model.caches_are_enabled():
            curr_masks = masks[:, :prompt_length]
        else:
            curr_masks = masks[:, :prompt_length, :prompt_length]
        tok = _step(input_pos[:, :prompt_length].squeeze(), curr_masks, prompt)
        state.update(tok.item())
        generated = torch.cat([prompt, tok], dim=-1)

        # Steps 1..n-1: incremental decode with KV cache.
        # ``curr_pos`` is the sequence index of the token we just sampled
        # (which becomes the input for the next forward pass), matching
        # upstream torchtune.generate's bookkeeping.
        curr_pos = prompt_length
        for _ in range(n - 1):
            if model.caches_are_enabled():
                curr_input_pos = input_pos[:, curr_pos].contiguous()
                curr_masks = masks[:, curr_pos, None, :].contiguous()
                x = tok.clone()
            else:
                curr_input_pos = input_pos[:, : curr_pos + 1]
                curr_masks = masks[:, : curr_pos + 1, : curr_pos + 1]
                x = generated.clone()
            tok = _step(curr_input_pos, curr_masks, x)
            state.update(tok.item())
            generated = torch.cat([generated, tok], dim=-1)
            curr_pos += 1

        return generated.squeeze(0)[-n:]


def describe_cycles(cycles):
    return f"{int(cycles)} cycles {cycles*sidq():.2f} seconds"


def add_ext(path, p):
    if p > 0:
        path = Path(path)
        path = str(path.parent / (path.stem + f".{p}" + path.suffix))
        return path
    return path


def generate_sequence(args, logger, dataset, predictor, p):
    (
        irq,
        n,
        prompt,
        prompt_compare,
        reg_start,
        prompt_df,
    ) = get_prompt(args, dataset, logger)
    states = prompt.squeeze(0).tolist()
    loader = RegLogParser(args)
    # prompt_df came back from get_prompt with BACK_REF / GATE_REPLAY rows
    # whose targets fell before the slice already materialised, so the
    # decoder can expand it without the preamble in scope.
    prompt_df_audio, _reg_widths = prepare_df_for_audio(
        prompt_df, dataset.reg_widths, irq, sidq(), strict=False
    )
    prompt_cycles = prompt_df_audio["diff"].sum()
    logger.info(
        "prompt lasts %s %u tokens (%u decoded rows), predicting %u tokens",
        describe_cycles(prompt_cycles),
        args.prompt_seq_len,
        len(prompt_df),
        n,
    )

    predict_states = predictor.predict(
        prompt, n, temperature=args.temperature, top_k=args.top_k, irq=irq
    )
    states.extend(predict_states.tolist())
    completion_df = loader._state_df(
        dataset.tokenizer.decode(predict_states.tolist()), dataset, irq
    )
    # ``prompt_df`` is already post-``_remove_voice_reg`` (absolute regs); put
    # the completion in the same coordinate system before concat so the
    # combined stream is decoder-ready in one form. Subsequent
    # ``prepare_df_for_audio`` will see no VOICE_REG rows and skip the
    # internal call (it's a no-op when ``len(df[df["reg"] == VOICE_REG]) == 0``).
    completion_df, _ = loader._remove_voice_reg(completion_df, dataset.reg_widths)
    df = pd.concat([prompt_df, completion_df], ignore_index=True)
    # Case-B safety net: scan the full prompt+generated row stream for any
    # BACK_REF or GATE_REPLAY that escapes its bounds. The LM should
    # predict only in-bounds refs, but this catches the bug rather than
    # letting it corrupt the audio render. (A proper sampling-time logit
    # guard belongs in predictor.predict; tracked as a follow-up.)
    try:
        validate_back_refs(df, prompt_frame_count=0)
        validate_gate_replays(df)
    except (AssertionError, ValueError) as e:
        # AssertionError: an escaped macro ref the LM predicted that
        # reaches before the start of the generated frames.
        # ValueError: a malformed row (e.g. NaN in description) that
        # the simulator can't dispatch -- usually means concat shapes
        # mismatched between prompt_df and completion_df.
        logger.error("generated stream rejected by safety net: %s", e)
        if args.min_acc:
            sys.exit(-1)
    predicted_compare = prompt_compare[args.prompt_seq_len :]
    f_acc = pd.NA
    acc = pd.NA
    if len(predicted_compare):
        acc = torchmetrics.functional.classification.multiclass_accuracy(
            predict_states[: len(predicted_compare)].to("cpu"),
            predicted_compare.to("cpu"),
            dataset.n_vocab,
        )
        acc = acc.item()
        f_acc = "%3.3f" % acc
    out_df = df.join(
        loader._state_df(
            dataset.tokenizer.decode(prompt_compare.numpy()), dataset, irq
        ),
        how="left",
        rsuffix="_p",
    )
    out_df["p_n"] = out_df["n"] == out_df["n_p"]
    if args.csv:
        csv = add_ext(args.csv, p)
        out_df.astype(MODEL_PDTYPE).to_csv(csv, index=False)
    if args.min_acc:
        if acc is pd.NA or acc < args.min_acc:
            logger.error(f"{acc} below min_acc {args.min_acc}")
            sys.exit(-1)
    df, reg_widths = prepare_df_for_audio(
        df, dataset.reg_widths, irq, sidq(), prompt_len=len(prompt_df)
    )
    total_cycles = df["diff"].sum()
    generated_cycles = total_cycles - prompt_cycles
    logger.info(
        "generated %s accuracy %s, total %s",
        describe_cycles(generated_cycles),
        f_acc,
        describe_cycles(total_cycles),
    )
    wav = add_ext(args.wav, p)
    write_samples(
        df,
        wav,
        reg_widths,
        cents=args.cents,
        reg_start=reg_start,
        asid=args.asid,
        sysex_delay=args.sysex_delay,
        descriptions=["prompt", "predictions"],
    )


def get_ckpt(ckpt, tb_logs):
    if ckpt:
        return ckpt
    # Prefer best-by-val_loss checkpoints written by
    # ``ModelCheckpoint(monitor="val_loss")`` (filename starts
    # ``best-``). Generalisation runs train two checkpointers in
    # parallel: the per-epoch saver and the best-val saver. The
    # best-val one is the right model state to evaluate on held-out
    # songs. Memorise runs have no val data and write only per-epoch
    # ckpts, so the second glob picks the latest of those instead.
    for pattern in (f"{tb_logs}/**/best-*.ckpt", f"{tb_logs}/**/*.ckpt"):
        ckpts = sorted(
            [(os.path.getmtime(p), p) for p in glob.glob(pattern, recursive=True)]
        )
        if ckpts:
            return ckpts[-1][1]
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
    model = model_compiler(args, model)
    return dataset, model, device, model_compiler


def run_predict(args, logger, dataset, model, device, model_compiler, p):
    model = model.to(device)
    vocab_arrays = None
    if getattr(args, "constrained_decode", False):
        vocab_arrays = precompute_vocab_arrays(dataset.tokenizer.tokens, device)
        logger.info("constrained decode enabled (vocab=%u)", vocab_arrays["n_vocab"])
    predictor = model_compiler(args, Predictor)(
        args,
        dataset,
        model,
        device,
        vocab_arrays=vocab_arrays,
        logger=logger,
    )
    generate_sequence(args, logger, dataset, predictor, p)
    model.cpu()
    del model
    torch.cuda.empty_cache()


def main():
    parser = add_args(argparse.ArgumentParser())
    args = parser.parse_args()
    logger = get_logger("INFO")
    dataset, model, device, model_compiler = load_model(args, logger)
    for p in range(args.predictions):
        run_predict(
            args,
            logger,
            dataset,
            copy.deepcopy(model),
            device,
            model_compiler,
            p,
        )


if __name__ == "__main__":
    main()
