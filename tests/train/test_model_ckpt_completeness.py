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
        pipeline_spec='{"transforms":[{"name":"hard_restart"}]}',
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

    def test_args_carries_pipeline_spec(self):
        model = self._make_model()
        self.assertTrue(
            hasattr(model.hparams.args, "pipeline_spec"),
            msg=(
                "args.pipeline_spec missing — pipeline-spec MUST be in args so "
                "the inferer can reconstruct the encoder configuration without "
                "external state."
            ),
        )

    def test_pipeline_spec_is_resolved_json_not_path_reference(self):
        import json

        model = self._make_model()
        raw = getattr(model.hparams.args, "pipeline_spec", "")
        self.assertFalse(
            raw.startswith("@"),
            msg=(
                "args.pipeline_spec stashed as '@path' reference — "
                "apply_pipeline_spec_to_args must resolve @path into JSON "
                "content before training, otherwise the checkpoint depends "
                "on an external file that may be moved/deleted."
            ),
        )
        if raw:
            parsed = json.loads(raw)
            self.assertIn(
                "transforms",
                parsed,
                msg="args.pipeline_spec JSON missing 'transforms' key.",
            )

    def test_apply_pipeline_spec_resolves_at_path(self):
        import json
        import tempfile

        from preframr.args import apply_pipeline_spec_to_args

        spec_dict = {"transforms": [{"name": "hard_restart"}, {"name": "ctrl_bigram"}]}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(spec_dict, f)
            path = f.name
        ns = argparse.Namespace(pipeline_spec=f"@{path}")
        apply_pipeline_spec_to_args(ns)
        self.assertFalse(
            ns.pipeline_spec.startswith("@"),
            msg=(
                f"after apply_pipeline_spec_to_args, args.pipeline_spec is still "
                f"a @path reference: {ns.pipeline_spec}"
            ),
        )
        recovered = json.loads(ns.pipeline_spec)
        self.assertEqual(recovered, spec_dict)

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
