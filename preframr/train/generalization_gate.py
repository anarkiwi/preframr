"""Auto-abort Lightning callback: pathological-distribution gates during training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info

from preframr_tokens import (
    detect_tail_cycle,
    distinct_n,
    tier_accuracy,
)


@dataclass
class GateThresholds:
    loop_collapse_max: float = 0.8
    loop_collapse_min_epoch: int = 10
    distinct_n_min: int = 30
    distinct_n_min_epoch: int = 15
    content_over_structural_min: float = 0.05
    content_over_structural_min_epoch: int = 20


class GeneralizationGate(pl.Callback):
    """Lightning callback that aborts training on pathological-distribution signatures."""

    def __init__(
        self,
        *,
        tier_map: dict,
        audit_prompts: Optional[list] = None,
        generate_fn: Optional[Callable] = None,
        generate_n: int = 512,
        generate_every_k_epochs: int = 5,
        thresholds: Optional[GateThresholds] = None,
    ):
        super().__init__()
        self.tier_map = tier_map
        self.audit_prompts = audit_prompts or []
        self.generate_fn = generate_fn
        self.generate_n = generate_n
        self.generate_every_k_epochs = generate_every_k_epochs
        self.thresholds = thresholds or GateThresholds()
        self._preds: list = []
        self._gt: list = []
        self.aborted_reason: Optional[str] = None

    def on_validation_epoch_start(self, _trainer, _pl_module):
        self._preds = []
        self._gt = []

    def on_validation_batch_end(
        self, _trainer, _pl_module, outputs, _batch, _batch_idx, _dataloader_idx=0
    ):
        if isinstance(outputs, dict) and "preds" in outputs and "gt" in outputs:
            self._preds.extend(int(t) for t in outputs["preds"].flatten().tolist())
            self._gt.extend(int(t) for t in outputs["gt"].flatten().tolist())

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        th = self.thresholds
        if self._preds:
            pca = tier_accuracy(self._preds, self._gt, self.tier_map)
            ratio = pca["content_over_structural"]
            pl_module.log("gate/content_over_structural", ratio, sync_dist=True)
            if (
                epoch >= th.content_over_structural_min_epoch
                and ratio < th.content_over_structural_min
            ):
                self._abort(
                    trainer,
                    pl_module,
                    f"content/structural {ratio:.3f} < {th.content_over_structural_min} "
                    f"at epoch {epoch}",
                )
                return
        if (
            self.audit_prompts
            and self.generate_fn is not None
            and epoch % self.generate_every_k_epochs == 0
        ):
            gens = [
                list(self.generate_fn(pl_module, p, self.generate_n))
                for p in self.audit_prompts
            ]
            collapsed = sum(detect_tail_cycle(g) is not None for g in gens)
            collapse_rate = collapsed / len(gens)
            distinct = sum(distinct_n(g) for g in gens) / len(gens)
            pl_module.log("gate/loop_collapse_rate", collapse_rate, sync_dist=True)
            pl_module.log("gate/distinct_n4_mean", distinct, sync_dist=True)
            if (
                epoch >= th.loop_collapse_min_epoch
                and collapse_rate > th.loop_collapse_max
            ):
                self._abort(
                    trainer,
                    pl_module,
                    f"loop_collapse_rate {collapse_rate:.2f} > {th.loop_collapse_max} "
                    f"at epoch {epoch}",
                )
                return
            if epoch >= th.distinct_n_min_epoch and distinct < th.distinct_n_min:
                self._abort(
                    trainer,
                    pl_module,
                    f"distinct_n4 {distinct:.1f} < {th.distinct_n_min} at epoch {epoch}",
                )
                return

    def _abort(self, trainer, pl_module, reason: str):
        rank_zero_info(f"[GeneralizationGate] ABORT: {reason}")
        pl_module.log("gate/aborted", 1.0)
        trainer.should_stop = True
        self.aborted_reason = reason
