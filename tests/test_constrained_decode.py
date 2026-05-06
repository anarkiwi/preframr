"""Unit tests for ``preframr.constrained_decode``.

Covers the per-step mask state machine: structural-validity rules
(BACK_REF distance, PATTERN_OVERLAY pairing, GATE_REPLAY/PLAY_INSTRUMENT
palette indexing, DELAY_REG / pad-token suppression), the per-frame
diff budget, voice-rotation tracking, and remaining-step PATTERN_REPLAY
overlay capping.
"""

import unittest

import numpy as np
import pandas as pd
import torch

from preframr.constrained_decode import (
    StreamState,
    _frame_marker_count,
    precompute_vocab_arrays,
)
from preframr.stfconstants import (
    BACK_REF_OP,
    DELAY_REG,
    FRAME_REG,
    GATE_REPLAY_OP,
    MIN_DIFF,
    PAD_REG,
    PATTERN_OVERLAY_OP,
    PATTERN_REPLAY_OP,
    PLAY_INSTRUMENT_OP,
    SET_OP,
    VOICES,
    VOICE_REG,
    VOICE_REG_SIZE,
)


def _build_vocab():
    """Compact vocab covering every class the mask cares about.

    Indices:
      0: PAD_REG (synthetic pad)
      1: SET reg=0 (real-reg)
      2: FRAME_REG val=11 (svt = 0b001011: voice 2 fn=0, voice 1 fn=1, none fn=2)
      3: FRAME_REG val=5  (svt = 0b000101: voice 0 for fn=0..2)
      4: DELAY_REG val=2
      5: VOICE_REG (advances fn)
      6: BACK_REF distance=1 length=1   (val=(1<<8)|1=257)
      7: BACK_REF distance=5 length=1   (val=(5<<8)|1=1281)
      8: PATTERN_REPLAY distance=2 length=1 with subreg=2 overlays
      9: PATTERN_OVERLAY (overlay row)
      10: GATE_REPLAY voice0_ctrl(reg=4) dir=0 slot=0
      11: GATE_REPLAY reg=4 dir=0 slot=2
      12: PLAY_INSTRUMENT reg=4 slot=3
      13: PLAY_INSTRUMENT reg=4 slot=10
    """
    rows = [
        {"op": SET_OP, "reg": PAD_REG, "subreg": -1, "val": 0},
        {"op": SET_OP, "reg": 0, "subreg": -1, "val": 7},
        {"op": SET_OP, "reg": FRAME_REG, "subreg": -1, "val": 11},
        {"op": SET_OP, "reg": FRAME_REG, "subreg": -1, "val": 5},
        {"op": SET_OP, "reg": DELAY_REG, "subreg": -1, "val": 2},
        {"op": SET_OP, "reg": VOICE_REG, "subreg": -1, "val": 0},
        {"op": BACK_REF_OP, "reg": -125, "subreg": -1, "val": (1 << 8) | 1},
        {"op": BACK_REF_OP, "reg": -125, "subreg": -1, "val": (5 << 8) | 1},
        {"op": PATTERN_REPLAY_OP, "reg": -125, "subreg": 2, "val": (2 << 8) | 1},
        {"op": PATTERN_OVERLAY_OP, "reg": -125, "subreg": 0, "val": 0},
        {"op": GATE_REPLAY_OP, "reg": 4, "subreg": 0, "val": 0},
        {"op": GATE_REPLAY_OP, "reg": 4, "subreg": 0, "val": 2},
        {"op": PLAY_INSTRUMENT_OP, "reg": 4, "subreg": -1, "val": 3},
        {"op": PLAY_INSTRUMENT_OP, "reg": 4, "subreg": -1, "val": 10},
    ]
    return pd.DataFrame(rows)


def _mask_bool(state, n_vocab):
    out = state.mask_logits(torch.zeros(n_vocab, dtype=torch.float32))
    return [v == float("-inf") for v in out.tolist()]


class TestFrameMarkerCount(unittest.TestCase):
    def test_counts_frame_and_delay(self):
        # Tiny vocab: token 0 = real reg, 1 = FRAME_REG, 2 = DELAY_REG
        is_frame_marker = np.array([False, True, True], dtype=bool)
        # Sequence: real, FRAME, real, DELAY, FRAME -> 3 markers
        ids = [0, 1, 0, 2, 1]
        self.assertEqual(_frame_marker_count(ids, is_frame_marker), 3)

    def test_empty(self):
        self.assertEqual(_frame_marker_count([], np.zeros(1, dtype=bool)), 0)


class TestPrecomputeVocabArrays(unittest.TestCase):
    def test_keys_and_shapes(self):
        df = _build_vocab()
        arrs = precompute_vocab_arrays(df, torch.device("cpu"))
        for k in (
            "is_frame_marker",
            "is_delay_reg",
            "is_pad",
            "is_real_reg",
            "is_back_ref_or_pattern_replay",
            "is_pattern_replay",
            "is_pattern_overlay",
            "is_gate_replay",
            "is_play_instrument",
            "is_frame_reg_strict",
            "is_voice_reg",
            "frame_sval",
            "gate_dir",
            "gate_slot",
            "instr_slot",
            "distance",
            "overlay_count",
        ):
            self.assertIn(k, arrs, k)
        self.assertEqual(arrs["n_vocab"], len(df))

    def test_frame_sval_extraction(self):
        df = _build_vocab()
        arrs = precompute_vocab_arrays(df, torch.device("cpu"))
        # Token 2 = FRAME_REG val=11 -> frame_sval = 11 & 0x3F = 11.
        self.assertEqual(int(arrs["frame_sval"][2].item()), 11)
        # Token 3 = FRAME_REG val=5 -> 5.
        self.assertEqual(int(arrs["frame_sval"][3].item()), 5)
        # Non-FRAME_REG tokens: frame_sval defaults to 0.
        self.assertEqual(int(arrs["frame_sval"][1].item()), 0)

    def test_distance_and_overlay_count(self):
        df = _build_vocab()
        arrs = precompute_vocab_arrays(df, torch.device("cpu"))
        # Token 6 = BACK_REF distance=1 -> distance[6]=1.
        self.assertEqual(int(arrs["distance"][6].item()), 1)
        self.assertEqual(int(arrs["distance"][7].item()), 5)
        # Token 8 = PATTERN_REPLAY distance=2 -> distance[8]=2; overlay_count=2.
        self.assertEqual(int(arrs["distance"][8].item()), 2)
        self.assertEqual(int(arrs["overlay_count"][8].item()), 2)
        # Non-back-ref/pattern-replay tokens have distance=0, overlay_count=0.
        self.assertEqual(int(arrs["distance"][1].item()), 0)
        self.assertEqual(int(arrs["overlay_count"][1].item()), 0)

    def test_gate_replay_indexing(self):
        df = _build_vocab()
        arrs = precompute_vocab_arrays(df, torch.device("cpu"))
        # Token 10: GATE_REPLAY reg=4 (=> would map to voice 0 if static)
        # subreg=0 -> dir=0; val=0 -> slot=0.
        self.assertEqual(int(arrs["gate_dir"][10].item()), 0)
        self.assertEqual(int(arrs["gate_slot"][10].item()), 0)
        # Token 11: slot=2.
        self.assertEqual(int(arrs["gate_slot"][11].item()), 2)


class TestStreamStateMasking(unittest.TestCase):
    def setUp(self):
        self.df = _build_vocab()
        self.arrs = precompute_vocab_arrays(self.df, torch.device("cpu"))
        self.n = self.arrs["n_vocab"]

    def _state(self, **kw):
        defaults = dict(
            init_frame_count=10,
            irq=19656,
            init_budget=19656,
            instrument_palette_size=4,
        )
        defaults.update(kw)
        return StreamState(self.arrs, **defaults)

    def test_pad_always_masked(self):
        state = self._state()
        m = _mask_bool(state, self.n)
        self.assertTrue(m[0], "pad token (idx 0) must always be masked")

    def test_delay_reg_masked_at_top_level(self):
        state = self._state()
        m = _mask_bool(state, self.n)
        # Token 4 = DELAY_REG: masked at top level.
        self.assertTrue(m[4])

    def test_pattern_overlay_orphan_masked_at_top_level(self):
        state = self._state()
        m = _mask_bool(state, self.n)
        # Token 9 = PATTERN_OVERLAY: orphan at top, must be masked.
        self.assertTrue(m[9])

    def test_back_ref_distance_check(self):
        # Frame count = 0 -> any distance reaches before frame 0.
        state = self._state(init_frame_count=0)
        m = _mask_bool(state, self.n)
        # Token 6 = BACK_REF distance=1: with frame_count=0, masked.
        self.assertTrue(m[6])
        # Token 7 = BACK_REF distance=5: masked too.
        self.assertTrue(m[7])

    def test_back_ref_within_bounds(self):
        # Frame count = 5 -> distance <= 5 valid; distance > 5 masked.
        state = self._state(init_frame_count=5)
        m = _mask_bool(state, self.n)
        self.assertFalse(m[6], "BACK_REF dist=1 valid at frame_count=5")
        # Token 7 = distance=5: 5 > 5 is False, so valid.
        self.assertFalse(m[7])

    def test_pattern_overlay_inside_block_only(self):
        state = self._state()
        # Manually set pending_overlays > 0.
        state.pending_overlays = 2
        m = _mask_bool(state, self.n)
        # Inside overlay block: only PATTERN_OVERLAY (token 9) is valid.
        for i in range(self.n):
            if i == 9:
                self.assertFalse(m[i], f"token {i} should be valid (overlay)")
            else:
                self.assertTrue(m[i], f"token {i} should be masked inside overlay")

    def test_gate_replay_palette_check(self):
        gp = np.zeros((VOICES, 2), dtype=np.int64)
        # voice 0 dir 0 has 2 slots; voice 2 has 5 slots.
        gp[0, 0] = 2
        gp[2, 0] = 5
        state = self._state(gate_palette_sizes=gp)
        # Default voice = 0 (sval=0/fn=0 yields v=-1 -> clamped 0).
        m = _mask_bool(state, self.n)
        # Token 10: slot=0 < 2 -> valid.
        self.assertFalse(m[10])
        # Token 11: slot=2 >= 2 -> masked.
        self.assertTrue(m[11])

    def test_play_instrument_palette_check(self):
        state = self._state(instrument_palette_size=4)
        m = _mask_bool(state, self.n)
        # Token 12: slot=3 < 4 -> valid.
        self.assertFalse(m[12])
        # Token 13: slot=10 >= 4 -> masked.
        self.assertTrue(m[13])

    def test_diff_budget_exhaustion(self):
        # init_budget < MIN_DIFF -> all real-reg tokens masked.
        state = self._state(init_budget=MIN_DIFF - 1)
        m = _mask_bool(state, self.n)
        # Token 1 = SET reg=0 (real-reg): masked.
        self.assertTrue(m[1])
        # Token 2 = FRAME_REG: not real-reg, still allowed.
        self.assertFalse(m[2])

    def test_remaining_steps_overlay_cap(self):
        # remaining_steps small -> PATTERN_REPLAY with overlay_count >
        # remaining-1 gets masked.
        state = self._state(remaining_steps=2)
        m = _mask_bool(state, self.n)
        # Token 8 = PATTERN_REPLAY with overlay_count=2; cap = remaining-1=1.
        # 2 > 1 -> masked.
        self.assertTrue(m[8])

    def test_remaining_steps_unset_no_cap(self):
        # remaining_steps=None disables the overlay cap.
        state = self._state(remaining_steps=None, init_frame_count=10)
        m = _mask_bool(state, self.n)
        # PATTERN_REPLAY distance=2 with frame_count=10 -> within bounds.
        self.assertFalse(m[8])

    def test_all_masked_safety_valve(self):
        # Force all-masked: budget exhausted + frame_count=0 + no slots.
        state = self._state(init_frame_count=0, init_budget=0)
        # The safety valve should unmask the first frame-marker token so
        # generation can advance.
        m = _mask_bool(state, self.n)
        self.assertIn(False, m, "all-masked safety valve should leave one")


class TestStreamStateUpdate(unittest.TestCase):
    def setUp(self):
        self.df = _build_vocab()
        self.arrs = precompute_vocab_arrays(self.df, torch.device("cpu"))

    def test_frame_marker_increments_count_resets_budget(self):
        state = StreamState(
            self.arrs,
            init_frame_count=0,
            irq=100,
            init_budget=10,
        )
        state.update(2)  # FRAME_REG val=11
        self.assertEqual(state.frame_count, 1)
        self.assertEqual(state.frame_budget, 100)

    def test_real_reg_charges_budget(self):
        state = StreamState(
            self.arrs,
            init_frame_count=0,
            irq=100,
            init_budget=64,
        )
        state.update(1)  # SET reg=0 -> -MIN_DIFF.
        self.assertEqual(state.frame_budget, 64 - MIN_DIFF)

    def test_voice_rotation_tracking(self):
        # FRAME_REG val=11 (svt=0b001011): fn=0 -> v=(11&3)-1=2;
        # after VOICE_REG: fn=1 -> v=((11>>2)&3)-1=1; fn=2 -> v=-1 clamped to 0.
        state = StreamState(self.arrs, init_frame_count=0, irq=100)
        state.update(2)  # FRAME_REG val=11
        self.assertEqual(state.current_sval, 11)
        self.assertEqual(state.current_fn, 0)
        self.assertEqual(state._current_voice(), 2)
        state.update(5)  # VOICE_REG -> fn=1
        self.assertEqual(state.current_fn, 1)
        self.assertEqual(state._current_voice(), 1)
        state.update(5)  # VOICE_REG -> fn=2
        self.assertEqual(state._current_voice(), 0)  # clamped from -1

    def test_frame_reg_resets_fn(self):
        state = StreamState(self.arrs, init_frame_count=0, irq=100)
        state.update(2)  # FRAME_REG val=11
        state.update(5)  # VOICE_REG -> fn=1
        state.update(3)  # FRAME_REG val=5
        # New FRAME_REG resets fn=0 and updates sval.
        self.assertEqual(state.current_sval, 5)
        self.assertEqual(state.current_fn, 0)
        # svt=5 (0b000101): fn=0 -> v=(5&3)-1=0.
        self.assertEqual(state._current_voice(), 0)

    def test_pattern_replay_sets_pending_overlays(self):
        state = StreamState(self.arrs, init_frame_count=10, irq=100)
        state.update(8)  # PATTERN_REPLAY with subreg=2 overlays
        self.assertEqual(state.pending_overlays, 2)

    def test_overlay_consumed(self):
        state = StreamState(self.arrs, init_frame_count=10, irq=100)
        state.update(8)  # PATTERN_REPLAY -> pending=2
        state.update(9)  # PATTERN_OVERLAY -> pending=1
        self.assertEqual(state.pending_overlays, 1)
        state.update(9)  # PATTERN_OVERLAY -> pending=0
        self.assertEqual(state.pending_overlays, 0)

    def test_remaining_steps_decrements(self):
        state = StreamState(
            self.arrs,
            init_frame_count=0,
            irq=100,
            remaining_steps=5,
        )
        state.update(1)  # any token
        self.assertEqual(state.remaining_steps, 4)
        state.update(1)
        self.assertEqual(state.remaining_steps, 3)


if __name__ == "__main__":
    unittest.main()
