"""Event-model render-to-WAV: decoded canonical writes render through the standard chain at the right
duration. Guards the event-native render path (``predict.py``'s ``_state_df`` is old-substrate and can't
decode event atoms; this is the replacement)."""

import wave

import numpy as np
import pandas as pd
import pytest

from preframr_tokens.events import oracle, stream
from preframr_tokens.events.constrained import EventStreamState
from preframr.inference.event_render import (
    EventConstraint,
    writes_to_timed_dump_df,
    render_writes_to_wav,
)

PAL_FRAME_CYCLES = 19656
N_VOCAB = stream.VOCAB_SIZE + 1


def _synth_df(n_frames=12, seed=3):
    """A tiny 3-voice dump (gated note walk per voice) so frames/voices/events all recur."""
    rng = np.random.default_rng(seed)
    writes = []
    for f in range(n_frames):
        for v in range(3):
            base = 7 * v
            fr = 0x1000 + v * 0x400 + int(rng.integers(0, 0x80))
            writes.append((f, base + 0, fr & 0xFF))
            writes.append((f, base + 1, (fr >> 8) & 0xFF))
            writes.append((f, base + 4, 0x41 if (f + v) % 4 else 0x40))
            writes.append((f, base + 5, 0x08))
            writes.append((f, base + 6, 0xA9))
    writes.sort(key=lambda t: t[0])
    return pd.DataFrame(
        {
            "clock": np.arange(len(writes), dtype=np.int64),
            "irq": np.array([w[0] for w in writes], dtype=np.int64),
            "chipno": np.zeros(len(writes), dtype=np.int64),
            "reg": np.array([w[1] for w in writes], dtype=np.int64),
            "val": np.array([w[2] for w in writes], dtype=np.int64),
        }
    )


def _event_nspace(df):
    """A synthetic tune's event atoms in model n-space (atom + 1; the model's vocab)."""
    atoms = stream.encode(oracle.ordered_writes(df))
    return [a + 1 for a in atoms]


def test_primed_constraint_admits_the_true_continuation():
    """The core audition fix: an EventConstraint PRIMED with a prompt prefix admits the real
    continuation at every step (its grammar mask is correct at the prompt/generation boundary), so a
    well-trained model is never masked away from a valid stream."""
    ids = _event_nspace(_synth_df(n_frames=24))
    split = len(ids) // 3
    prompt, cont = ids[:split], ids[split:]
    ec = EventConstraint(prompt, N_VOCAB)
    for j, tok in enumerate(cont):
        allowed = ec.allowed_nspace()
        assert allowed[tok], f"primed mask rejected the true atom at step {j}"
        assert (
            0 < int(allowed.sum()) < N_VOCAB
        ), "mask must be a non-trivial restriction"
        ec.update(tok)


def test_unprimed_state_rejects_a_mid_frame_continuation_that_priming_admits():
    """Control: split mid-frame-group (just before an event-kind atom). A FRESH state expects the
    leading frame-count varint and rejects it immediately; the PRIMED state knows it is mid-frame and
    admits it. This is exactly why unprimed constrained generation drifted off-grammar.
    """
    ids = _event_nspace(_synth_df(n_frames=24))
    fresh_mask = EventStreamState().valid_mask()
    split = next(i for i, t in enumerate(ids) if i > 0 and not fresh_mask[t - 1])
    prompt, nxt = ids[:split], ids[split]
    assert not fresh_mask[nxt - 1], "a fresh state must reject a mid-frame event atom"
    assert EventConstraint(prompt, N_VOCAB).allowed_nspace()[
        nxt
    ], "the primed state must admit it"


def test_mask_logits_blocks_invalid_and_keeps_valid():
    """mask_logits sets every grammar-invalid id to -inf and leaves valid ones finite (PAD always -inf)."""
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "zeros"):
        pytest.skip("torch unavailable (namespace stub)")
    ids = _event_nspace(_synth_df(n_frames=24))
    ec = EventConstraint(ids[: len(ids) // 3], N_VOCAB)
    logits = torch.zeros(1, N_VOCAB)
    masked = ec.mask_logits(logits)[0].numpy()
    allowed = ec.allowed_nspace()
    assert np.isneginf(masked[~allowed]).all(), "invalid ids must be -inf"
    assert np.isfinite(masked[allowed]).all(), "valid ids must stay finite"
    assert np.isneginf(masked[0]), "PAD (id 0) is never valid"


def test_timed_dump_has_absolute_frame_paced_clock():
    """writes_to_timed_dump_df turns the surrogate clock into an absolute, frame-paced one (every frame
    boundary advances frame_cycles), which is what the parser/renderer need."""
    writes = stream.canonical_writes(oracle.ordered_writes(_synth_df()))
    timed = writes_to_timed_dump_df(writes, PAL_FRAME_CYCLES)
    assert list(timed.columns) == ["clock", "irq", "chipno", "reg", "val"]
    assert timed["clock"].is_monotonic_increasing
    frames = sorted({f for f, _, _ in writes})
    for fr in frames:
        first = timed[timed["irq"] == fr * PAL_FRAME_CYCLES]["clock"].iloc[0]
        assert (
            first == fr * PAL_FRAME_CYCLES
        ), "each frame's first write sits at frame*frame_cycles"


def test_decoded_writes_render_to_wav_at_frame_duration(tmp_path):
    """The event-native path: canonical writes -> WAV via the standard render chain, at the duration
    the frame count implies (frames * frame_cycles), proving the timing reconstruction is right.
    """
    df = _synth_df(n_frames=12)
    writes = stream.canonical_writes(oracle.ordered_writes(df))
    n_frames = writes[-1][0] + 1
    wav = str(tmp_path / "ev.wav")
    n = render_writes_to_wav(writes, PAL_FRAME_CYCLES, wav)
    assert n > 0
    with wave.open(wav) as w:
        rate = w.getframerate()
        assert w.getnframes() == n
    secs = n / rate
    assert (
        0.1 < secs < 5.0
    ), f"a {n_frames}-frame synth renders to a short clip, got {secs:.2f}s"
