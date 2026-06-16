"""Event-native render-to-WAV: decoded canonical writes -> a frame-paced raw dump -> the existing
RegLogParser-reparse + prepare_df_for_audio + render_to_wav chain. predict.py's _state_df path is
old-substrate and cannot decode event atoms; this is the replacement. writes_to_timed_dump_df adds
the one missing piece -- absolute cycle timing (clock = frame * frame_cycles, the player frame
period) -- since the decode emits a surrogate clock with no frame structure."""

from __future__ import annotations

import logging
import os
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd

from preframr_audio.audio_driver import render_to_wav
from preframr_audio.sidwav import sidq
from preframr_tokens import prepare_df_for_audio, read_initial_irq
from preframr_tokens.events.constrained import EventStreamState
from preframr_tokens.events.generate import tokens_to_writes, writes_to_dump_df
from preframr_tokens.reglogparser import RegLogParser
from preframr_tokens.stfconstants import DUMP_SUFFIX
from preframr_tokens.tokenizer_config import named_config

_INTRA_FRAME_GAP = 8
PAL_FRAME_CYCLES = 19656


class EventConstraint:
    """Event-grammar logit mask for constrained generation, PRIMED with the prompt so the mask is
    correct at the prompt/generation boundary. The old ``StreamState`` mask keys on FRAME_REG, absent
    from the event alphabet, so it is a no-op here and the model drifts off-grammar. Atom space ``a``
    maps to model n-space id ``a + 1`` (id 0 = PAD)."""

    def __init__(self, prompt_ids, n_vocab):
        self.state = EventStreamState()
        for tok in prompt_ids:
            atom = int(tok) - 1
            if atom >= 0:
                self.state.push(atom)
        self.n_vocab = int(n_vocab)

    def allowed_nspace(self) -> np.ndarray:
        """Boolean over model n-space ids: id a+1 allowed iff atom a is grammar-valid next; PAD never."""
        vm = self.state.valid_mask()
        nmask = np.zeros(self.n_vocab, dtype=bool)
        k = min(len(vm), self.n_vocab - 1)
        nmask[1 : 1 + k] = vm[:k]
        return nmask

    def mask_logits(self, logits):
        import torch  # pylint: disable=import-outside-toplevel

        allow = torch.as_tensor(self.allowed_nspace(), device=logits.device)
        return logits.masked_fill(~allow, float("-inf"))

    def update(self, tok) -> None:
        atom = int(tok) - 1
        if atom >= 0:
            self.state.push(atom)


def writes_to_timed_dump_df(writes, frame_cycles, chipno: int = 0) -> pd.DataFrame:
    """(frame, reg, val) writes -> a raw dump with absolute cycle timing (clock = frame*frame_cycles
    + intra-frame offset; frame_cycles = the player frame period, PAL ~19656). Skipped frames widen
    the clock gap so the SID holds state (sustained audio). Reuses writes_to_dump_df for (reg, val).
    """
    base = writes_to_dump_df(writes, chipno=chipno)
    if base.empty:
        return base
    frame = base["irq"].to_numpy().astype(np.int64)
    intra = base.groupby("irq").cumcount().to_numpy().astype(np.int64)
    clock = frame * int(frame_cycles) + intra * _INTRA_FRAME_GAP
    base["clock"] = clock
    base["irq"] = frame * int(frame_cycles)
    return base


def render_writes_to_wav(
    writes, frame_cycles=PAL_FRAME_CYCLES, wav_path=None, cents: int = 50, logger=None
) -> int:
    """Frame-resolution event writes -> WAV. Writes a temp frame-paced dump, reparses it to the
    (op,reg,subreg,val) substrate, then runs the standard ``prepare_df_for_audio`` + ``render_to_wav``
    chain. Returns the sample count."""
    logger = logger or logging.getLogger("event_render")
    timed = writes_to_timed_dump_df(writes, frame_cycles)
    tmp = tempfile.NamedTemporaryFile(suffix=DUMP_SUFFIX, delete=False)
    tmp.close()
    try:
        timed.to_parquet(tmp.name)
        args = SimpleNamespace(**vars(named_config("baseline", cents=cents)))
        args.reglog = tmp.name
        args.config = "baseline"
        args.cents = cents
        parser = RegLogParser(args, logger=logger)
        rotations = list(
            parser.parse(tmp.name, max_perm=1, require_pq=False, reparse=True)
        )
        if not rotations:
            raise ValueError("frame-paced dump reparsed to zero rotations")
        df = rotations[0]
        irq = read_initial_irq(df)
        df, reg_widths = prepare_df_for_audio(
            df, {}, irq, sidq(), strict=False, cents=cents
        )
        return render_to_wav(df, wav_path, reg_widths, irq, cents=cents)
    finally:
        os.unlink(tmp.name)


def render_tokens_to_wav(
    tokenizer,
    bpe_ids,
    wav_path,
    frame_cycles=PAL_FRAME_CYCLES,
    cents: int = 50,
    logger=None,
) -> int:
    """Generated model token ids -> WAV (decode + render). ``frame_cycles`` is the player frame period
    (carry the prompt tune's ``irq``)."""
    writes = tokens_to_writes(tokenizer, bpe_ids)
    return render_writes_to_wav(
        writes, frame_cycles, wav_path, cents=cents, logger=logger
    )


def run_render(args, logger):
    """Audition CLI body: load a ckpt, generate continuations from .blocks.npy prompts (reusing the
    event_gate machinery), constrained-decode, then decode + render each to a WAV under --wav-dir.
    Returns 0 if any WAV was written; the generate step is unchanged, only render is added.
    """
    import torch as _torch  # pylint: disable=import-outside-toplevel

    from preframr.inference.event_gate import (  # pylint: disable=import-outside-toplevel
        decode_tolerant,
        load_prompts,
    )
    from preframr.inference.predict import (  # pylint: disable=import-outside-toplevel
        Predictor,
        load_model,
    )

    dataset, model, device, _ = load_model(args, logger)
    model = model.to(device)
    predictor = Predictor(
        args, dataset, model, device, vocab_arrays=None, logger=logger
    )
    n_vocab = model.model.tok_embeddings.num_embeddings
    tokenizer = dataset.tokenizer
    prompts = load_prompts(
        args.blocks_glob, args.n_prompts, args.prompt_seq_len, args.gen_tokens, logger
    )
    os.makedirs(args.wav_dir, exist_ok=True)
    temperature = getattr(args, "temperature", 1.0)
    top_k = getattr(args, "top_k", 1)
    rendered = 0
    for i, (path, prompt_ids, _truth) in enumerate(prompts):
        if model.model.caches_are_enabled():
            model.model.reset_caches()
        prompt = _torch.tensor(prompt_ids, dtype=_torch.long).unsqueeze(0)
        gen = (
            predictor.predict(
                prompt,
                args.gen_tokens,
                temperature=temperature,
                top_k=top_k,
                irq=args.frame_cycles,
                event_constraint=EventConstraint(prompt_ids, n_vocab),
            )
            .cpu()
            .numpy()
            .astype(np.int64)
        )
        full = np.concatenate([prompt_ids, gen]).tolist()
        writes, trim, err = decode_tolerant(tokenizer, full, len(prompt_ids))
        if not writes:
            logger.error("prompt %s: no decodable writes (%s)", path, err)
            continue
        wav = os.path.join(args.wav_dir, f"audition_{i:02d}.wav")
        n = render_writes_to_wav(
            writes, args.frame_cycles, wav, cents=args.cents, logger=logger
        )
        logger.info(
            "rendered %s -> %s (%u samples, %.1fs, trim=%u)",
            os.path.basename(path),
            wav,
            n,
            n / 48000,
            trim,
        )
        rendered += 1
    logger.info("rendered %u/%u auditions to %s", rendered, len(prompts), args.wav_dir)
    return 0 if rendered else 1


def main():
    import argparse  # pylint: disable=import-outside-toplevel
    import sys  # pylint: disable=import-outside-toplevel

    from preframr.args import add_args  # pylint: disable=import-outside-toplevel
    from preframr.inference.event_gate import (  # pylint: disable=import-outside-toplevel
        add_gate_args,
    )
    from preframr.utils import get_logger  # pylint: disable=import-outside-toplevel

    parser = add_gate_args(add_args(argparse.ArgumentParser()))
    parser.add_argument("--wav-dir", default="/scratch/tmp/auditions")
    parser.add_argument("--frame-cycles", type=int, default=PAL_FRAME_CYCLES)
    parser.set_defaults(repetition_penalty=1.3, no_repeat_ngram_size=4, top_k=8)
    args = parser.parse_args()
    sys.exit(run_render(args, get_logger("INFO")))


if __name__ == "__main__":
    main()
