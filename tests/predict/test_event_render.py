"""BACC render path: ``state_to_dump_df`` turns raw 25-register-per-frame state back into the input dump schema (changed-only writes, frame-paced clock) and ``render_state_to_wav`` replays it through a resid chip (guarded behind pyresidfp so the pure-logic test runs everywhere)."""

import unittest

import numpy as np
import pytest

from preframr.inference.event_render import (
    render_ids_to_wav,
    render_program_to_wav,
    render_state_to_wav,
    state_to_dump_df,
)

CPF = 19656

_GENERIC_PORT_BLOCKED = (
    "TODO(flat-v2): the generic/hubbard_monty backend was removed (flat-v2 GoatTracker"
    " is the live path), so this v1-style NoteOn program no longer renders; re-point at"
    " a GoatTracker flat fixture once the generic flat port lands."
)

_SEED_KEYS = (
    "notenum",
    "instrnr",
    "lnthcc",
    "lenleft",
    "sfl",
    "sfh",
    "porta",
    "vctrl",
    "pdly",
    "pdir",
)


def _synthetic_program():
    from preframr_tokens import BaccProgram, NoteOn

    seed = {k: [0, 0, 0] for k in _SEED_KEYS}
    seed["init_speed"] = 1
    seed["resetspd"] = 1
    return BaccProgram(
        driver="hubbard_monty",
        nframes=8,
        boot=[0] * 25,
        instruments=[[0] * 8 for _ in range(64)],
        score=[NoteOn(frame=0, voice=0, note=60, instr=1, lnth=2, porta=0)],
        seed=seed,
        tables={"static_img": [0] * 256},
    )


def _state(nframes=6):
    """A tiny deterministic state: a couple of registers stepping per frame."""
    state = np.zeros((nframes, 25), dtype=int)
    for f in range(nframes):
        state[f, 0] = f
        state[f, 4] = 0x41 if f % 2 else 0x40
        state[f, 24] = 0x0F
    return state


class TestStateToDumpDf(unittest.TestCase):
    def test_schema_and_changed_only(self):
        df = state_to_dump_df(_state(3), CPF)
        self.assertEqual(list(df.columns), ["clock", "reg", "val", "chipno"])
        self.assertEqual(int((df["reg"] == 24).sum()), 1)
        self.assertTrue((df["chipno"] == 0).all())

    def test_clock_is_frame_paced(self):
        df = state_to_dump_df(_state(4), CPF)
        for frame_i in range(4):
            rows = df[df["clock"] == frame_i * CPF]
            if frame_i == 0:
                self.assertGreater(len(rows), 0)

    def test_empty_state(self):
        df = state_to_dump_df(np.zeros((0, 25), dtype=int), CPF)
        self.assertEqual(len(df), 0)


class TestRenderStateToWav(unittest.TestCase):
    def test_renders_nonempty(self):
        pytest.importorskip("pyresidfp")
        import tempfile
        import wave

        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            n = render_state_to_wav(_state(10), CPF, tmp.name)
            self.assertGreater(n, 0)
            with wave.open(tmp.name) as w:
                self.assertEqual(w.getnframes(), n)


class TestRenderProgram(unittest.TestCase):
    @unittest.skip(_GENERIC_PORT_BLOCKED)
    def test_render_program_and_ids_to_wav(self):
        pytest.importorskip("pyresidfp")
        from preframr.tokenizer import BaccTokenizer

        prog = _synthetic_program()
        n_prog = render_program_to_wav(prog, CPF, None)
        self.assertGreater(n_prog, 0)
        ids = BaccTokenizer().encode(prog)
        n_ids = render_ids_to_wav(ids, CPF, None, driver="hubbard_monty")
        self.assertEqual(n_ids, n_prog)


if __name__ == "__main__":
    unittest.main()
