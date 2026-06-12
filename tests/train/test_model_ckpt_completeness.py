"""Enforce that the Model checkpoint contains everything needed for inference."""

import argparse
import unittest

import pandas as pd

from preframr.train.model import Model
from preframr_tokens.stfconstants import FRAME_REG, MODEL_PDTYPE, SET_OP

_REQUIRED_HPARAMS_FOR_INFERENCE = (
    "args",
    "n_vocab",
    "tokens",
    "tkmodel",
    "metadata",
    "reg_widths",
)


def _tiny_args(**overrides):
    args = argparse.Namespace(
        embed=32,
        heads=4,
        kv_heads=2,
        layers=2,
        intermediate=64,
        max_seq_len=16,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=1e4,
        rope_scale=1.0,
        tie_word_embeddings=True,
        precision="high",
        max_autotune=False,
        accumulate_grad_batches=1,
        learning_rate=1e-4,
        weight_decay=0.01,
        label_smoothing=0.0,
        model="llama3_2",
        macro_flags="",
        macro_config="",
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def _tiny_tokens():
    return pd.DataFrame(
        [
            {"op": SET_OP, "reg": -1, "subreg": -1, "val": 0, "n": 0},
            {"op": SET_OP, "reg": FRAME_REG, "subreg": -1, "val": 1, "n": 1},
            {"op": SET_OP, "reg": 0, "subreg": -1, "val": 5, "n": 2},
        ],
        dtype=MODEL_PDTYPE,
    )


class TestModelCkptCompleteness(unittest.TestCase):
    def _make_model(self, **kw):
        args = _tiny_args()
        tokens = _tiny_tokens()
        reg_widths = {0: 2, 1: 1, 4: 1, 21: 2, 23: 1, 24: 1}
        return Model(
            args,
            n_vocab=3,
            tokens=tokens,
            tkmodel=None,
            metadata=["pad", "frame", "set_reg0_val5"],
            reg_widths=reg_widths,
            **kw,
        )

    def test_all_required_hparams_present_in_memory(self):
        model = self._make_model()
        present = set(model.hparams.keys())
        missing = set(_REQUIRED_HPARAMS_FOR_INFERENCE) - present
        self.assertFalse(
            missing,
            msg=(
                f"Model hparams missing required fields for inference: {missing}. "
                f"Update save_hyperparameters in Model.__init__ AND add to "
                f"_REQUIRED_HPARAMS_FOR_INFERENCE here in lockstep."
            ),
        )

    def test_args_carries_macro_flags(self):
        model = self._make_model()
        self.assertTrue(
            hasattr(model.hparams.args, "macro_flags"),
            msg=(
                "args.macro_flags missing — the resolved macro-flag set MUST be "
                "in args so the inferer can reconstruct the encoder configuration "
                "without external state."
            ),
        )

    def test_macro_flags_is_resolved_csv(self):
        model = self._make_model()
        raw = getattr(model.hparams.args, "macro_flags", "")
        self.assertNotIn("{", raw)
        self.assertNotIn("@", raw)

    def test_apply_macro_flags_sets_booleans(self):
        from preframr.args import apply_macro_flags_to_args

        from preframr_tokens.macros.flag_registry import macro_flag_names

        names = sorted(macro_flag_names())
        if not names:
            self.skipTest("no macro flags in this tokens build")
        flag = names[0]
        ns = argparse.Namespace(macro_flags=flag, macro_config="")
        apply_macro_flags_to_args(ns)
        self.assertTrue(getattr(ns, flag))
        self.assertIn(flag, ns.macro_flags.split(","))

    def test_reg_widths_carried_through(self):
        model = self._make_model()
        self.assertEqual(model.reg_widths[0], 2)
        self.assertEqual(model.reg_widths[24], 1)
        self.assertEqual(model.hparams.reg_widths[0], 2)

    def test_metadata_round_trips(self):
        model = self._make_model()
        self.assertEqual(model.metadata, ["pad", "frame", "set_reg0_val5"])
        self.assertEqual(model.hparams.metadata, ["pad", "frame", "set_reg0_val5"])

    def test_tokens_carries_alphabet(self):
        model = self._make_model()
        self.assertEqual(len(model.tokens), 3)
        self.assertEqual(len(model.hparams.tokens), 3)

    def test_default_reg_widths_is_dict(self):
        args = _tiny_args()
        tokens = _tiny_tokens()
        model = Model(args, n_vocab=3, tokens=tokens, tkmodel=None, metadata=None)
        self.assertEqual(model.reg_widths, {})


if __name__ == "__main__":
    unittest.main()
