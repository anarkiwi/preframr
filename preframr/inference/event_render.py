"""Event-model render-to-WAV: generated/decoded canonical writes -> a frame-paced raw dump ->
the existing render chain (``RegLogParser`` reparse -> ``prepare_df_for_audio`` -> ``render_to_wav``).

The event codec drops dumps to frame-resolution ``(frame, reg, val)`` writes; ``predict.py``'s
``_state_df`` path is the old (op,reg,subreg,val) substrate and cannot decode event atoms, so this is
the event-native render. The one piece the decode side leaves open is absolute cycle timing:
``events.generate.writes_to_dump_df`` emits a surrogate clock with no frame structure, so the parser
finds no rotations. :func:`writes_to_timed_dump_df` reconstructs an absolute clock from the player's
frame period (``frame_cycles`` = the tune's ``irq``), after which the unchanged render chain produces
a byte-faithful WAV (a frame with no writes renders as held SID state, i.e. sustained sound).
"""

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
from preframr_tokens.events.generate import tokens_to_writes, writes_to_dump_df
from preframr_tokens.reglogparser import RegLogParser
from preframr_tokens.stfconstants import DUMP_SUFFIX
from preframr_tokens.tokenizer_config import named_config

_INTRA_FRAME_GAP = (
    8  # cycles between writes within a frame (<< frame_cycles; sub-frame, ~inaudible)
)
PAL_FRAME_CYCLES = (
    19656  # PAL cycles per player tick; the single-speed-corpus default frame period
)


def writes_to_timed_dump_df(writes, frame_cycles, chipno: int = 0) -> pd.DataFrame:
    """``(frame, reg, val)`` writes -> a raw dump with ABSOLUTE cycle timing the parser/renderer accept.
    ``clock`` = ``frame * frame_cycles`` + an intra-frame offset; ``frame_cycles`` is the player's frame
    period in cycles (PAL ~19656), i.e. the tune's ``irq``. Skipped frames widen the clock gap -> the SID
    holds state -> sustained audio (correct). Reuses :func:`writes_to_dump_df` for the (reg, val) columns.
    """
    base = writes_to_dump_df(writes, chipno=chipno)
    if base.empty:
        return base
    frame = (
        base["irq"].to_numpy().astype(np.int64)
    )  # writes_to_dump_df puts the frame index in irq
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
