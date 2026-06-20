"""Tests for ``preframr.train.model`` loss-function machinery."""

import unittest

import torch

from preframr.train.model import chunked_cross_entropy


class TestPadMaskedReduction(unittest.TestCase):
    """The training_step reduction is a plain mean over non-PAD targets and
    must not divide by zero when every target is PAD (clamp(min=1.0))."""

    def test_all_nonpad_equals_mean(self):
        per_tok = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mask = torch.ones_like(per_tok)
        loss = (per_tok * mask).sum() / mask.sum().clamp(min=1.0)
        self.assertAlmostEqual(float(loss), float(per_tok.mean()))

    def test_all_pad_is_finite_zero(self):
        per_tok = torch.tensor([1.0, 2.0, 3.0])
        mask = torch.zeros_like(per_tok)
        loss = (per_tok * mask).sum() / mask.sum().clamp(min=1.0)
        self.assertEqual(float(loss), 0.0)


class TestChunkedCrossEntropy(unittest.TestCase):
    """``chunked_cross_entropy`` is a memory-efficient drop-in for
    ``F.cross_entropy(logits.swapaxes(1, 2), target, reduction='none', ...)``.
    Values and gradients must match the eager call to within fp tolerance.
    """

    def _reference(self, logits, target, label_smoothing):
        return torch.nn.functional.cross_entropy(
            logits.swapaxes(1, 2),
            target,
            reduction="none",
            label_smoothing=label_smoothing,
        )

    def test_values_match_reference_no_chunk(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 7, 17, dtype=torch.float32)
        target = torch.randint(0, 17, (2, 7))
        got = chunked_cross_entropy(logits, target, chunk_bytes=1 << 30)
        ref = self._reference(logits, target, 0.0)
        torch.testing.assert_close(got, ref)

    def test_values_match_reference_chunked(self):
        torch.manual_seed(0)
        logits = torch.randn(4, 13, 32, dtype=torch.float32)
        target = torch.randint(0, 32, (4, 13))
        got = chunked_cross_entropy(logits, target, chunk_bytes=256)
        ref = self._reference(logits, target, 0.0)
        torch.testing.assert_close(got, ref)

    def test_values_match_reference_with_label_smoothing(self):
        torch.manual_seed(1)
        logits = torch.randn(3, 11, 23, dtype=torch.float32)
        target = torch.randint(0, 23, (3, 11))
        got = chunked_cross_entropy(
            logits, target, label_smoothing=0.1, chunk_bytes=256
        )
        ref = self._reference(logits, target, 0.1)
        torch.testing.assert_close(got, ref)

    def test_gradients_match_reference_chunked(self):
        torch.manual_seed(2)
        logits = torch.randn(2, 9, 19, dtype=torch.float32, requires_grad=True)
        target = torch.randint(0, 19, (2, 9))
        got = chunked_cross_entropy(logits, target, chunk_bytes=128)
        got.sum().backward()
        got_grad = logits.grad.detach().clone()
        logits.grad = None
        ref = self._reference(logits, target, 0.0)
        ref.sum().backward()
        torch.testing.assert_close(got_grad, logits.grad)

    def test_list_of_chunks_matches_full_tensor(self):
        """A seq-dim-split list of logit chunks (set_num_output_chunks path) must
        match the equivalent single (B, S, V) tensor."""
        torch.manual_seed(3)
        full = torch.randn(2, 8, 11, dtype=torch.float32)
        target = torch.randint(0, 11, (2, 8))
        chunks = list(full.chunk(2, dim=1))
        got = chunked_cross_entropy(chunks, target)
        ref = self._reference(full, target, 0.0)
        torch.testing.assert_close(got, ref)


if __name__ == "__main__":
    unittest.main()
