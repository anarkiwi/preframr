# pylint: disable=no-member
"""Targeted tests for ``preframr.train``."""

import argparse
import os
import tempfile
import unittest
from unittest import mock

import preframr.train.trainer as train_module
from preframr.train.trainer import train


def _train_args(**kw):
    """argparse.Namespace stub for ``train()`` -- defaults match a
    minimal valid arg set; tests override the fields they care about.
    """
    defaults = {
        "tb_logs": "/tmp/tb",
        "max_epochs": 10,
        "model_state": "/tmp/state.ckpt",
        "trainer_precision": "bf16-mixed",
        "accumulate_grad_batches": 1,
        "log_every_n_steps": 10,
        "ckpt_hours": 0,
        "stop_loss": None,
        "stop_delta": None,
        "early_stop_patience": 0,
        "early_stop_min_delta": 0.0,
        "val_check_every": 0,
    }
    defaults.update(kw)
    return argparse.Namespace(**defaults)


def _patch_trainer_stack(stack):
    """Install mocks for the Lightning stack train() touches. Returns
    the stack so callers can attach further mocks. The trainer mock's
    ``return_value`` is the per-instance ``Trainer`` so callers can
    inspect call_args.
    """
    stack.enter_context(mock.patch.object(train_module, "pl"))
    stack.enter_context(mock.patch.object(train_module, "ModelCheckpoint"))
    stack.enter_context(mock.patch.object(train_module, "EarlyStopping"))
    stack.enter_context(mock.patch.object(train_module, "SchedulerFreeModelCheckpoint"))
    return stack


class TestTrainCallbackWiring(unittest.TestCase):
    """Covers callback-list construction under different arg combos.
    The trainer.fit call is mocked; we don't actually train -- we
    inspect what Lightning would have received.
    """

    def _run(self, val_dataloader=None, val_subset_names=None, **arg_overrides):
        import contextlib

        args = _train_args(**arg_overrides)
        model = mock.MagicMock()
        dataloader = mock.MagicMock()
        logger = mock.MagicMock()
        with contextlib.ExitStack() as stack:
            _patch_trainer_stack(stack)
            train(
                model,
                dataloader,
                val_dataloader,
                args,
                ckpt_path=None,
                logger=logger,
                val_subset_names=val_subset_names,
            )
            trainer_mock = train_module.pl.Trainer
            sched_mock = train_module.SchedulerFreeModelCheckpoint
            es_mock = train_module.EarlyStopping
            mc_mock = train_module.ModelCheckpoint
        return model, trainer_mock, sched_mock, es_mock, mc_mock

    def test_default_callbacks_one_epoch_checkpoint_only(self):
        _, trainer, sched, es, mc = self._run()
        self.assertEqual(sched.call_count, 1)
        es.assert_not_called()
        mc.assert_not_called()
        callbacks = trainer.call_args.kwargs["callbacks"]
        self.assertEqual(len(callbacks), 1)

    def test_ckpt_hours_adds_hourly_snapshot(self):
        _, trainer, sched, _, _ = self._run(ckpt_hours=1)
        self.assertEqual(sched.call_count, 2)
        hourly_call = sched.call_args_list[1]
        self.assertIn("train_time_interval", hourly_call.kwargs)
        callbacks = trainer.call_args.kwargs["callbacks"]
        self.assertEqual(len(callbacks), 2)

    def test_stop_loss_adds_train_loss_early_stop(self):
        _, _, _, es, _ = self._run(stop_loss=2.5)
        es.assert_called_once()
        kw = es.call_args.kwargs
        self.assertEqual(kw["monitor"], "train_loss")
        self.assertEqual(kw["mode"], "min")
        self.assertEqual(kw["stopping_threshold"], 2.5)

    def test_stop_delta_adds_train_loss_early_stop(self):
        _, _, _, es, _ = self._run(stop_delta=0.01)
        es.assert_called_once()
        kw = es.call_args.kwargs
        self.assertEqual(kw["min_delta"], 0.01)

    def test_val_dataloader_adds_best_by_val_checkpoint(self):
        val_dl = mock.MagicMock()
        _, _, _, _, mc = self._run(val_dataloader=val_dl)
        mc.assert_called_once()
        kw = mc.call_args.kwargs
        self.assertEqual(kw["monitor"], "val_loss")
        self.assertEqual(kw["mode"], "min")
        self.assertIn("best-{epoch}", kw["filename"])

    def test_val_dataloader_with_patience_adds_val_loss_early_stop(self):
        val_dl = mock.MagicMock()
        _, _, _, es, _ = self._run(
            val_dataloader=val_dl, early_stop_patience=5, early_stop_min_delta=0.01
        )
        es.assert_called_once()
        kw = es.call_args.kwargs
        self.assertEqual(kw["monitor"], "val_loss")
        self.assertEqual(kw["patience"], 5)
        self.assertEqual(kw["min_delta"], 0.01)

    def test_val_dataloader_zero_patience_clamps_to_one(self):
        val_dl = mock.MagicMock()
        _, _, _, es, _ = self._run(
            val_dataloader=val_dl, early_stop_patience=0, early_stop_min_delta=0.01
        )
        es.assert_called_once()
        self.assertEqual(es.call_args.kwargs["patience"], 1)

    def test_val_check_every_threads_through_to_trainer(self):
        val_dl = mock.MagicMock()
        _, trainer, _, _, _ = self._run(val_dataloader=val_dl, val_check_every=3)
        self.assertEqual(trainer.call_args.kwargs["check_val_every_n_epoch"], 3)

    def test_val_check_every_unset_no_kwarg(self):
        val_dl = mock.MagicMock()
        _, trainer, _, _, _ = self._run(val_dataloader=val_dl, val_check_every=0)
        self.assertNotIn("check_val_every_n_epoch", trainer.call_args.kwargs)

    def test_val_subset_names_threaded_to_model(self):
        val_dl = mock.MagicMock()
        model, _, _, _, _ = self._run(
            val_dataloader=val_dl, val_subset_names=["eval_a", "eval_b"]
        )
        self.assertEqual(model.val_subset_names, ["eval_a", "eval_b"])

    def test_no_val_subset_names_leaves_model_untouched(self):
        import contextlib

        args = _train_args()
        model = mock.MagicMock(spec=[])
        with contextlib.ExitStack() as stack:
            _patch_trainer_stack(stack)
            train(
                model,
                mock.MagicMock(),
                None,
                args,
                ckpt_path=None,
                logger=mock.MagicMock(),
            )
        self.assertFalse(hasattr(model, "val_subset_names"))


class TestTrainFitInvocation(unittest.TestCase):
    def test_fit_called_with_ckpt_path(self):
        import contextlib

        args = _train_args()
        model = mock.MagicMock()
        dataloader = mock.MagicMock()
        val_dl = mock.MagicMock()
        logger = mock.MagicMock()
        with contextlib.ExitStack() as stack:
            _patch_trainer_stack(stack)
            train(
                model,
                dataloader,
                val_dl,
                args,
                ckpt_path="/tmp/resume.ckpt",
                logger=logger,
            )
            trainer_inst = train_module.pl.Trainer.return_value
        trainer_inst.fit.assert_called_once()
        kw = trainer_inst.fit.call_args.kwargs
        self.assertIs(kw["train_dataloaders"], dataloader)
        self.assertIs(kw["val_dataloaders"], val_dl)
        self.assertEqual(kw["ckpt_path"], "/tmp/resume.ckpt")
        logger.info.assert_any_call("resuming from %s", "/tmp/resume.ckpt")

    def test_returns_model(self):
        import contextlib

        args = _train_args()
        model = mock.MagicMock()
        with contextlib.ExitStack() as stack:
            _patch_trainer_stack(stack)
            out = train(
                model,
                mock.MagicMock(),
                None,
                args,
                ckpt_path=None,
                logger=mock.MagicMock(),
            )
        self.assertIs(out, model)


class TestMain(unittest.TestCase):
    """Covers main()'s argparse + collaborator dispatch. RegDataset /
    loaders / model / train are all mocked so the test stays in the
    unit-test budget.
    """

    def _run_main(self, argv):
        with mock.patch.object(train_module.os, "sys", create=True):
            pass
        import sys as real_sys

        with mock.patch.object(real_sys, "argv", ["train", *argv]):
            train_module.main()

    def _common_main_mocks(self, stack):
        stack.enter_context(mock.patch.object(train_module, "RegDataset"))
        stack.enter_context(
            mock.patch.object(train_module, "get_loader", return_value=mock.MagicMock())
        )
        stack.enter_context(
            mock.patch.object(
                train_module,
                "get_val_loader",
                return_value=(mock.MagicMock(), ["eval_a"]),
            )
        )
        stack.enter_context(mock.patch.object(train_module, "get_model"))
        return stack.enter_context(mock.patch.object(train_module, "train"))

    def test_happy_path_dispatches_through_to_train(self):
        import contextlib

        with contextlib.ExitStack() as stack:
            train_mock = self._common_main_mocks(stack)
            self._run_main([])
        train_mock.assert_called_once()
        self.assertEqual(train_mock.call_args.kwargs["val_subset_names"], ["eval_a"])

    def test_model_state_path_must_exist(self):
        import contextlib

        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "nope.ckpt")
            with contextlib.ExitStack() as stack:
                self._common_main_mocks(stack)
                with self.assertRaises(ValueError) as ctx:
                    self._run_main(["--model-state", missing])
        self.assertIn("No such checkpoint", str(ctx.exception))

    def test_model_state_existing_passed_as_ckpt_path(self):
        import contextlib

        with tempfile.TemporaryDirectory() as tmp:
            existing = os.path.join(tmp, "state.ckpt")
            with open(existing, "w") as f:
                f.write("stub")
            with contextlib.ExitStack() as stack:
                train_mock = self._common_main_mocks(stack)
                self._run_main(["--model-state", existing])
            self.assertEqual(train_mock.call_args.args[4], existing)


if __name__ == "__main__":
    unittest.main()
