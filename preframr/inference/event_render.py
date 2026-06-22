"""Render a BACC token stream to a WAV. The codec yields a per-frame, 25-register raw SID state (``render_program``), exactly what a resid chip consumes, so the render path is direct: replay each frame's changed registers into a pyresidfp ``SoundInterfaceDevice`` and clock it one frame period. This sidesteps the deleted ``prepare_df_for_audio`` logical-register substrate; the only render deps are pyresidfp + scipy + numpy."""

from __future__ import annotations

import logging
from datetime import timedelta

import numpy as np
from preframr_tokens import CPF, ids_to_program, render_program

NREG = 25


def render_state_to_wav(state, cpf=CPF, wav_path=None, chip_model="MOS8580"):
    """Per-frame ``(nframes, 25)`` raw register state -> WAV; returns #samples. Each frame writes only the registers that changed since the previous frame (the dump's own semantics), then clocks the chip one frame period (``cpf`` cycles); ``wav_path=None`` renders without writing a file."""
    from pyresidfp import (  # pylint: disable=import-outside-toplevel
        SoundInterfaceDevice,
    )
    from pyresidfp.sound_interface_device import (  # pylint: disable=import-outside-toplevel
        ChipModel,
    )

    state = np.asarray(state, dtype=int)
    sid = SoundInterfaceDevice(model=getattr(ChipModel, chip_model))
    seconds_per_frame = float(cpf) / float(sid.clock_frequency)
    prev = [None] * NREG
    chunks = []
    for frame in state:
        for reg in range(NREG):
            val = int(frame[reg])
            if val != prev[reg]:
                sid.write_register(reg, val)
                prev[reg] = val
        samples = np.asarray(
            sid.clock(timedelta(seconds=seconds_per_frame)), dtype=np.int16
        )
        if len(samples):
            chunks.append(samples)
    out = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)
    if wav_path is not None:
        from scipy.io import wavfile  # pylint: disable=import-outside-toplevel

        wavfile.write(wav_path, int(sid.sampling_frequency), out)
    return len(out)


def state_to_dump_df(state, cpf=CPF, chipno=0):
    """Per-frame ``(nframes, 25)`` state -> a register-dump DataFrame: one ``(clock, reg, val, chipno)`` row per changed register per frame (``clock = frame * cpf``), the same schema as the input ``.dump.parquet`` so a generated tune can be re-read by ``per_frame_state`` for downstream scoring."""
    import pandas as pd  # pylint: disable=import-outside-toplevel

    state = np.asarray(state, dtype=int)
    rows = []
    prev = [None] * NREG
    for frame_i, frame in enumerate(state):
        clock = int(frame_i * cpf)
        for reg in range(NREG):
            val = int(frame[reg])
            if val != prev[reg]:
                rows.append((clock, reg, val, chipno))
                prev[reg] = val
    return pd.DataFrame(rows, columns=["clock", "reg", "val", "chipno"])


def render_ids_to_wav(
    ids, cpf=CPF, wav_path=None, chip_model="MOS8580", driver="generic"
):
    """Model-space id stream -> WAV: shift ids back to BACC program space (dropping PAD / out-of-range), recover the program (``driver`` defaults to ``generic`` = the trained sid-only stream), render to per-frame state, then to audio. Raises if the id stream is not a complete valid BACC program (``ids_to_program`` indexes past a truncated stream), so callers feeding generated output wrap this and fall back."""
    from preframr_tokens import VOCAB  # pylint: disable=import-outside-toplevel

    prog_ids = [int(i) - 1 for i in ids if 1 <= int(i) <= VOCAB]
    program = ids_to_program(prog_ids, driver=driver)
    state = render_program(program)
    return render_state_to_wav(state, cpf, wav_path, chip_model)


def render_program_to_wav(program, cpf=CPF, wav_path=None, chip_model="MOS8580"):
    """Recovered ``BaccProgram`` -> WAV (skips the id round-trip)."""
    return render_state_to_wav(render_program(program), cpf, wav_path, chip_model)


def run_render(args, logger=None):
    """Audition CLI body: generate continuations from ``.blocks.npy`` prompts and render each decodable result to a WAV under ``--wav-dir``; the ground-truth tune is always rendered (pipeline smoke) and the generated continuation too when it decodes to a valid program."""
    import os  # pylint: disable=import-outside-toplevel

    import torch as _torch  # pylint: disable=import-outside-toplevel

    from preframr.inference.event_gate import (  # pylint: disable=import-outside-toplevel
        load_prompts,
    )
    from preframr.inference.predict import (  # pylint: disable=import-outside-toplevel
        Predictor,
        load_model,
    )

    logger = logger or logging.getLogger("event_render")
    dataset, model, device, _ = load_model(args, logger)
    model = model.to(device)
    predictor = Predictor(args, dataset, model, device, logger=logger)
    prompts = load_prompts(
        args.blocks_glob, args.n_prompts, args.prompt_seq_len, args.gen_tokens, logger
    )
    os.makedirs(args.wav_dir, exist_ok=True)
    cpf = float(getattr(args, "frame_cycles", CPF))
    rendered = 0
    for i, (path, prompt_ids, truth) in enumerate(prompts):
        if model.model.caches_are_enabled():
            model.model.reset_caches()
        prompt = _torch.tensor(prompt_ids, dtype=_torch.long).unsqueeze(0)
        gen = (
            predictor.predict(
                prompt,
                args.gen_tokens,
                temperature=getattr(args, "temperature", 1.0),
                top_k=getattr(args, "top_k", None),
            )
            .cpu()
            .numpy()
            .astype(np.int64)
        )
        full = np.concatenate([prompt_ids, gen]).tolist()
        truth_wav = os.path.join(args.wav_dir, f"audition_{i:02d}.truth.wav")
        try:
            n = render_ids_to_wav(truth, cpf, truth_wav)
            logger.info("rendered truth %s -> %s (%u samples)", path, truth_wav, n)
            rendered += 1
        except Exception as err:  # pylint: disable=broad-except
            logger.error("truth %s did not render: %s", path, err)
        gen_wav = os.path.join(args.wav_dir, f"audition_{i:02d}.gen.wav")
        try:
            n = render_ids_to_wav(full, cpf, gen_wav)
            logger.info("rendered generation -> %s (%u samples)", gen_wav, n)
            rendered += 1
        except Exception as err:  # pylint: disable=broad-except
            logger.info("generation %s not a valid program (expected): %s", path, err)
    logger.info("rendered %u wav(s) to %s", rendered, args.wav_dir)
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
    parser.add_argument("--frame-cycles", type=int, default=int(CPF))
    parser.set_defaults(repetition_penalty=1.3, no_repeat_ngram_size=4, top_k=8)
    args = parser.parse_args()
    sys.exit(run_render(args, get_logger("INFO")))


if __name__ == "__main__":
    main()
