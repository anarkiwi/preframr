"""Smoke test for ``Model.__init__`` + a single training/validation step on a
tiny synthetic vocab. Covers the LightningModule glue + forward path without
standing up a Trainer."""

import argparse
import unittest

import torch

from preframr.train.model import Model


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


def _model(metadata=None):
    return Model(
        _tiny_args(),
        n_vocab=3,
        tokens=_tiny_tokens(),
        tkmodel=None,
        metadata=metadata,
    )


class TestModelInit(unittest.TestCase):
    def test_init_builds_underlying_model(self):
        model = _model()
        self.assertEqual(model.n_vocab, 3)
        self.assertEqual(model.metadata, None)
        self.assertEqual(model.model.tok_embeddings.num_embeddings, 3)

    def test_forward_on_random_input(self):
        model = _model()
        model.eval()
        x = torch.tensor([[0, 1, 2, 1]], dtype=torch.long)
        with torch.no_grad():
            out = model.model(x)
        self.assertEqual(out.shape, torch.Size([1, 4, 3]))

    def test_training_step_returns_scalar_loss(self):
        model = _model()
        x = torch.tensor([[1, 2, 1, 2]], dtype=torch.long)
        y = torch.tensor([[2, 1, 2, 1]], dtype=torch.long)
        loss = model.training_step((x, y), 0)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))

    def test_training_step_all_pad_is_finite(self):
        """All-PAD targets must not divide by zero (clamp(min=1.0))."""
        model = _model()
        x = torch.tensor([[1, 2, 1, 2]], dtype=torch.long)
        y = torch.zeros((1, 4), dtype=torch.long)
        loss = model.training_step((x, y), 0)
        self.assertTrue(torch.isfinite(loss))

    def test_validation_step_logs_acc_and_loss(self):
        model = _model()
        log_calls = []
        model.log = lambda *a, **k: log_calls.append((a, k))
        x = torch.tensor([[1, 2, 1, 2]], dtype=torch.long)
        y = torch.tensor([[2, 1, 2, 1]], dtype=torch.long)
        out = model.validation_step((x, y), 0)
        self.assertIn("loss", out)
        self.assertIn("preds", out)
        self.assertIn("gt", out)
        self.assertTrue(torch.isfinite(out["loss"]))
        self.assertEqual(out["preds"].shape, y.shape)
        names = [a[0] for (a, _) in log_calls]
        self.assertIn("val_loss", names)
        self.assertIn("val_acc", names)

    def test_configure_optimizers(self):
        opt = _model().configure_optimizers()
        self.assertIsNotNone(opt)


class _FakeTokenizer:
    tokens = _tiny_tokens()
    tkmodel = None

    def token_metadata(self):
        return list(self.tokens)


class _FakeDataset:
    n_vocab = 3
    tokenizer = _FakeTokenizer()


class TestGetModel(unittest.TestCase):
    def test_get_model_builds_via_factory(self):
        import logging

        from preframr.train.model import get_model

        model = get_model(_FakeDataset(), _tiny_args(compile=False), logging)
        self.assertEqual(model.n_vocab, 3)
        self.assertEqual(len(model.tokens), 3)


if __name__ == "__main__":
    unittest.main()
