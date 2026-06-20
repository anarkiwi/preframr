"""Enforce that the Model checkpoint contains everything needed for inference."""

import argparse
import unittest

from preframr.train.model import Model

_REQUIRED_HPARAMS_FOR_INFERENCE = (
    "args",
    "n_vocab",
    "tokens",
    "tkmodel",
    "metadata",
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
        log_embeddings=False,
        model="llama3_2",
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def _tiny_tokens():
    return ["PAD", "c0", "t0"]


class TestModelCkptCompleteness(unittest.TestCase):
    def _make_model(self, metadata=None):
        return Model(
            _tiny_args(),
            n_vocab=3,
            tokens=_tiny_tokens(),
            tkmodel=None,
            metadata=metadata if metadata is not None else ["pad", "c0", "t0"],
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

    def test_metadata_round_trips(self):
        model = self._make_model()
        self.assertEqual(model.metadata, ["pad", "c0", "t0"])
        self.assertEqual(model.hparams.metadata, ["pad", "c0", "t0"])

    def test_tokens_carries_alphabet(self):
        model = self._make_model()
        self.assertEqual(len(model.tokens), 3)
        self.assertEqual(len(model.hparams.tokens), 3)

    def test_n_vocab_carried(self):
        model = self._make_model()
        self.assertEqual(model.hparams.n_vocab, 3)


if __name__ == "__main__":
    unittest.main()
