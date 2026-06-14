#!/usr/bin/env python3

import argparse
import copy
import sys

import numpy as np
import pandas as pd
import pyarrow

pyarrow.PyExtensionType = pyarrow.ExtensionType
from torchtune.generation import generate, sample as _tt_sample
from torchtune.modules import RMSNorm
import torch
import torchmetrics

from preframr.args import add_args, MODEL_PRECISION
from preframr_tokens import (
    StreamState,
    frame_marker_count,
    precompute_subtoken_arrays,
    precompute_vocab_arrays,
    tail_charge_for_prompt,
)
from preframr_tokens import (
    validate_back_refs,
    validate_pattern_overlays,
)
from preframr.train.model import get_device, Model
from preframr.inference.predict_lib import add_ext, describe_cycles, get_ckpt
from preframr.train.regdataset import RegDataset, get_prompt
from preframr_tokens import (
    RegLogParser,
    prepare_df_for_audio,
    remove_voice_reg,
)
from preframr_audio.audio_driver import play_samples, render_to_wav
from preframr_audio.sidwav import sidq
from preframr_tokens.stfconstants import MODEL_PDTYPE, PAD_ID
from preframr.utils import get_logger


def _last_token_logits(model_out):
    """Extract last-token logits from a model forward result."""
    if isinstance(model_out, list):
        model_out = model_out[-1]
    return model_out[:, -1]


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
    def predict(
        self,
        prompt,
        n,
        temperature=1.0,
        top_k=None,
        irq=None,
        event_constraint=None,
    ):
        if event_constraint is not None:
            return self._predict_constrained(
                prompt, n, temperature, top_k, irq, state=event_constraint
            )
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
        return self._predict_constrained(
            prompt,
            n,
            temperature,
            top_k,
            irq,
        )

    @torch.inference_mode()
    def _predict_constrained(
        self,
        prompt,
        n,
        temperature,
        top_k,
        irq,
        state=None,
    ):
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

        prompt_mask = torch.tril(
            torch.ones(prompt_length, max_seq_len, dtype=torch.bool, device=self.device)
        ).unsqueeze(0)
        caches_on = model.caches_are_enabled()
        step_mask: torch.Tensor | None = None
        full_mask: torch.Tensor | None = None
        if caches_on:
            step_mask = torch.zeros(
                (1, 1, max_seq_len), dtype=torch.bool, device=self.device
            )
        else:
            full_mask = torch.tril(
                torch.ones(total_len, max_seq_len, dtype=torch.bool, device=self.device)
            ).unsqueeze(0)
        input_pos = torch.arange(0, total_len, device=self.device).unsqueeze(0)

        prompt_ids = prompt.squeeze(0).tolist()
        if state is not None:
            pass
        elif self.vocab_arrays.get("subtoken_mode"):
            seed = StreamState(
                self.vocab_arrays,
                init_frame_count=0,
                irq=irq,
                init_budget=irq,
                logger=self.logger,
            )
            for tid in prompt_ids:
                seed.update(int(tid))
            state = StreamState(
                self.vocab_arrays,
                init_frame_count=seed.frame_count,
                irq=irq,
                init_budget=seed.frame_budget,
                init_sval=seed.current_sval,
                init_fn=seed.current_fn,
                remaining_steps=n,
                logger=self.logger,
            )
            state.pending_overlays = seed.pending_overlays
        else:
            is_frame_marker_np = self.vocab_arrays["is_frame_marker"]
            is_frame_reg_strict_np = self.vocab_arrays["is_frame_reg_strict"]
            is_voice_reg_np = self.vocab_arrays["is_voice_reg"]
            frame_sval_np = self.vocab_arrays["frame_sval"]
            prompt_frames = frame_marker_count(prompt_ids, is_frame_marker_np)
            init_budget = max(
                irq - tail_charge_for_prompt(prompt_ids, self.vocab_arrays), 0
            )
            prompt_arr = np.asarray(prompt_ids, dtype=np.int64)
            init_sval = 0
            init_fn = 0
            frame_strict_positions = np.nonzero(is_frame_reg_strict_np[prompt_arr])[0]
            if frame_strict_positions.size:
                last_fr = int(frame_strict_positions[-1])
                init_sval = int(frame_sval_np[prompt_arr[last_fr]])
                init_fn = int(is_voice_reg_np[prompt_arr[last_fr + 1 :]].sum())
            state = StreamState(
                self.vocab_arrays,
                init_frame_count=prompt_frames,
                irq=irq,
                init_budget=init_budget,
                init_sval=init_sval,
                init_fn=init_fn,
                remaining_steps=n,
                logger=self.logger,
            )

        def _q():
            return torch.empty(
                (bsz, model.tok_embeddings.num_embeddings), device=self.device
            ).exponential_(1, generator=self.rng)

        rep_penalty = float(getattr(self.args, "repetition_penalty", 1.0) or 1.0)
        no_repeat_n = int(getattr(self.args, "no_repeat_ngram_size", 0) or 0)
        pen_window = int(getattr(self.args, "decode_penalty_window", 128) or 128)

        def _penalize(masked, recent):
            """Tier-1 anti-collapse: repetition penalty + no-repeat-n-gram on the recent window, applied
            after the grammar mask. Reverts the n-gram ban if it would leave no grammatically-valid token.
            """
            if not recent:
                return masked
            if rep_penalty != 1.0:
                idx = torch.tensor(
                    sorted(set(recent)), device=masked.device, dtype=torch.long
                )
                v = masked.index_select(-1, idx)
                masked.index_copy_(
                    -1, idx, torch.where(v > 0, v / rep_penalty, v * rep_penalty)
                )
            if no_repeat_n >= 2 and len(recent) >= no_repeat_n:
                prefix = tuple(recent[-(no_repeat_n - 1) :])
                banned = {
                    recent[i + no_repeat_n - 1]
                    for i in range(len(recent) - no_repeat_n + 1)
                    if tuple(recent[i : i + no_repeat_n - 1]) == prefix
                }
                if banned:
                    saved = masked.clone()
                    for t in banned:
                        masked[..., t] = float("-inf")
                    if bool(torch.isinf(masked).all()):
                        masked = saved
            return masked

        def _step(curr_input_pos, curr_masks, x, recent):
            logits = _last_token_logits(
                model(x, input_pos=curr_input_pos, mask=curr_masks)
            )
            masked = _penalize(state.mask_logits(logits), recent)
            return _tt_sample(masked, temperature=temperature, top_k=top_k, q=_q())

        if caches_on:
            curr_masks = prompt_mask
        else:
            assert full_mask is not None
            curr_masks = full_mask[:, :prompt_length, :prompt_length]
        tok = _step(
            input_pos[:, :prompt_length].squeeze(),
            curr_masks,
            prompt,
            prompt.reshape(-1).tolist()[-pen_window:],
        )
        state.update(tok.item())
        generated = torch.empty(
            (bsz, total_len), dtype=prompt.dtype, device=self.device
        )
        generated[:, :prompt_length] = prompt
        curr_pos = prompt_length
        generated[:, curr_pos] = tok[:, 0]

        for _ in range(n - 1):
            if caches_on:
                assert step_mask is not None
                curr_input_pos = input_pos[:, curr_pos].contiguous()
                step_mask[..., : curr_pos + 1] = True
                curr_masks = step_mask
                x = tok
            else:
                assert full_mask is not None
                curr_input_pos = input_pos[:, : curr_pos + 1]
                curr_masks = full_mask[:, : curr_pos + 1, : curr_pos + 1]
                x = generated[:, : curr_pos + 1].clone()
            recent = generated[
                0, max(0, curr_pos + 1 - pen_window) : curr_pos + 1
            ].tolist()
            tok = _step(curr_input_pos, curr_masks, x, recent)
            state.update(tok.item())
            curr_pos += 1
            generated[:, curr_pos] = tok[:, 0]

        return generated.squeeze(0)[-n:]


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
        prompt,
        n,
        temperature=args.temperature,
        top_k=args.top_k,
        irq=irq,
    )
    states.extend(predict_states.tolist())
    completion_df = loader._state_df(
        dataset.tokenizer.decode(predict_states.tolist()), dataset, irq
    )
    completion_df, _ = remove_voice_reg(completion_df, dataset.reg_widths)
    df = pd.concat([prompt_df, completion_df], ignore_index=True)
    try:
        validate_pattern_overlays(df)
        validate_back_refs(df, prompt_frame_count=0)
    except (AssertionError, ValueError) as e:
        logger.error("generated stream rejected by safety net: %s", e)
        if args.min_acc:
            sys.exit(-1)
    predicted_compare = prompt_compare[args.prompt_seq_len :]
    f_acc = pd.NA
    acc = pd.NA
    _ACC_VOCAB_CAP = 16384
    if len(predicted_compare) and dataset.n_vocab <= _ACC_VOCAB_CAP:
        acc = torchmetrics.functional.classification.multiclass_accuracy(
            predict_states[: len(predicted_compare)].to("cpu"),
            predicted_compare.to("cpu"),
            dataset.n_vocab,
        )
        acc = acc.item()
        f_acc = "%3.3f" % acc
    elif len(predicted_compare):
        logger.info(
            "accuracy compute skipped (n_vocab=%u > %u cap)",
            dataset.n_vocab,
            _ACC_VOCAB_CAP,
        )
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
    prompt_rows = len(prompt_df)
    completion_rows = len(df) - prompt_rows
    audio_df = None
    audio_reg_widths = None
    try:
        audio_df, audio_reg_widths = prepare_df_for_audio(
            df, dataset.reg_widths, irq, sidq(), prompt_len=prompt_rows
        )
    except (AssertionError, ValueError) as e:
        logger.error("prepare_df_for_audio rejected full stream: %s", e)
        lo, hi = 0, completion_rows
        good_n = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            trial = df.iloc[: prompt_rows + mid].copy()
            try:
                audio_df, audio_reg_widths = prepare_df_for_audio(
                    trial, dataset.reg_widths, irq, sidq(), prompt_len=prompt_rows
                )
                good_n = mid
                lo = mid + 1
            except (AssertionError, ValueError):
                hi = mid - 1
        if audio_df is None:
            logger.error("no renderable prefix found; skipping wav")
            sys.exit(2)
        logger.error(
            "rendered partial: %u/%u completion rows valid",
            good_n,
            completion_rows,
        )
    reg_widths = audio_reg_widths
    df = audio_df
    total_cycles = df["diff"].sum()
    generated_cycles = total_cycles - prompt_cycles
    logger.info(
        "generated %s accuracy %s, total %s",
        describe_cycles(generated_cycles),
        f_acc,
        describe_cycles(total_cycles),
    )
    wav = add_ext(args.wav, p)
    render_to_wav(
        df,
        wav,
        reg_widths=reg_widths,
        irq=irq,
        cents=args.cents,
        reg_start=reg_start,
        descriptions=["prompt", "predictions"],
    )
    if getattr(args, "predict_dump", None):
        dump_path = add_ext(args.predict_dump, p)
        pred_only = df[df["description"] == 1].reset_index(drop=True)
        pred_only.attrs["irq"] = int(irq) if irq is not None else 0
        pred_only.to_parquet(dump_path)
        logger.info("wrote prediction dump (%u rows) to %s", len(pred_only), dump_path)
    if getattr(args, "play", False):
        try:
            logger.info("starting real-time playback...")
            play_samples(df, reg_widths, irq=irq, cents=args.cents, reg_start=reg_start)
            logger.info("playback complete")
        except RuntimeError as e:
            logger.error("--play unavailable: %s", e)


def _patch_unembed_keep_dtype(decoder):
    """Override torchtune's ``TransformerDecoder.unembed`` so the output
    projection stays in the model's compute dtype instead of being cast
    to fp32. The upstream version does ``self.output(h).float()`` for
    training-time numerical stability, but at inference that cast
    materialises a ``(B, prompt_len, vocab)`` fp32 tensor -- 1.16 GB
    """
    import types

    def unembed(self, h):
        h = self.norm(h)
        if self.num_output_chunks > 0:
            return self.chunked_output(h)
        return self.output(h)

    decoder.unembed = types.MethodType(unembed, decoder)


def _keep_norms_fp32(model):
    """Keep RMSNorm scale params fp32 after a bf16 model cast so the fp32-input
    fused ``F.rms_norm`` kernel dispatches; a bf16 weight against the fp32
    activation forces the slow non-fused path plus extra dtype copies."""
    for module in model.modules():
        if isinstance(module, RMSNorm):
            module.scale.data = module.scale.data.float()


def load_model(args, logger):
    from preframr.args import apply_macro_flags_to_args

    ckpt = get_ckpt(args.model_state, args.tb_logs)
    logger.info("loading %s", ckpt)
    # pylint: disable=no-value-for-parameter
    with torch.serialization.safe_globals([argparse.Namespace]):
        model = Model.load_from_checkpoint(ckpt, weights_only=False, map_location="cpu")
    if not getattr(model, "per_tier_heads_on", False):
        _patch_unembed_keep_dtype(model.model)
    if not getattr(args, "macro_flags", "") and getattr(
        getattr(model, "hparams", {}), "args", None
    ):
        recovered = getattr(model.hparams.args, "macro_flags", "")
        if recovered:
            args.macro_flags = recovered
            logger.info("recovered macro_flags from checkpoint: %s", recovered)
    apply_macro_flags_to_args(args)
    dataset = RegDataset(args, logger=logger)
    recovered_reg_widths = getattr(model, "reg_widths", {}) or {}
    if recovered_reg_widths:
        dataset.reg_widths = dict(recovered_reg_widths)
        logger.info(
            "recovered reg_widths from checkpoint (%u regs)",
            len(recovered_reg_widths),
        )
    dataset.preload(tokens=model.tokens, tkmodel=model.tkmodel)
    dataset.predict_load()
    device, model_compiler = get_device(args, logger)
    model.eval()
    model.model.eval()
    predict_precision = MODEL_PRECISION[args.model_precision]
    model = model.to(predict_precision)
    _keep_norms_fp32(model)
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
    vocab_arrays = None
    if getattr(args, "constrained_decode", False):
        if getattr(args, "tkvocab", 0) and dataset.tokenizer.tkmodel is not None:
            vocab_arrays = precompute_subtoken_arrays(
                dataset.tokenizer.tokens, dataset.tokenizer
            )
            logger.info(
                "constrained decode enabled (sub-token mode, vocab=%u)",
                vocab_arrays["n_vocab"],
            )
        else:
            vocab_arrays = precompute_vocab_arrays(dataset.tokenizer.tokens)
            logger.info(
                "constrained decode enabled (vocab=%u)", vocab_arrays["n_vocab"]
            )
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
    from preframr.args import apply_macro_flags_to_args

    parser = add_args(argparse.ArgumentParser())
    args = parser.parse_args()
    apply_macro_flags_to_args(args)
    logger = get_logger("INFO")
    dataset, model, device, model_compiler = load_model(args, logger)
    for p in range(args.predictions):
        m = copy.deepcopy(model) if args.predictions > 1 else model
        run_predict(
            args,
            logger,
            dataset,
            m,
            device,
            model_compiler,
            p,
        )


if __name__ == "__main__":
    main()
