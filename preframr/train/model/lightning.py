"""`Model(LightningModule)` -- pytorch-lightning wrapper that owns the torchtune body, the (label-smoothed) cross-entropy loss, and the training/validation steps. The BACC vocab is a fixed 33-symbol alphabet (no op/reg/tier structure), so the loss is plain per-token CE over non-PAD targets. This is the only file in `preframr.train.model` that imports `pytorch_lightning`."""

import torch
from pytorch_lightning import LightningModule

from preframr.train.model.bodies import MODEL_GETTERS, OPTIMIZER
from preframr.train.model.losses import chunked_cross_entropy


class Model(LightningModule):
    def __init__(self, args, n_vocab, tokens, tkmodel, metadata):
        super().__init__()
        self.args = args
        self.n_vocab = n_vocab
        self.tokens = tokens
        self.tkmodel = None
        self.metadata = metadata
        if tkmodel:
            self.tkmodel = tkmodel.to_str()
        self.save_hyperparameters("args", "n_vocab", "tokens", "tkmodel", "metadata")
        self.model = MODEL_GETTERS[args.model](n_vocab, args)
        num_output_chunks = int(getattr(args, "num_output_chunks", 0) or 0)
        if num_output_chunks == 0 and n_vocab >= 32768:
            num_output_chunks = 8
        if num_output_chunks > 0 and hasattr(self.model, "set_num_output_chunks"):
            self.model.set_num_output_chunks(num_output_chunks)
        self.num_output_chunks = num_output_chunks
        self.optimizer = OPTIMIZER(
            self.parameters(),
            lr=self.args.learning_rate,
            foreach=True,
            weight_decay=self.args.weight_decay,
        )
        self.val_subset_names = ["val"]
        self._val_epoch_buckets = {}

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        preds = self.model(x)
        per_tok = chunked_cross_entropy(
            preds,
            y,
            label_smoothing=self.args.label_smoothing,
        )
        pad_mask = (y != 0).float()
        return (per_tok * pad_mask).sum() / pad_mask.sum().clamp(min=1.0)

    def on_before_backward(self, loss):
        self.log("train_loss", loss, on_epoch=True, on_step=True)

    def _reset_val_epoch_buckets(self):
        """Reset per-subset accumulators for a new validation epoch."""
        self._val_epoch_buckets = {
            name: {"loss_sum": 0.0, "correct_sum": 0.0, "tok_sum": 0.0}
            for name in self.val_subset_names
        }

    def validation_step(
        self, batch, batch_idx, dataloader_idx=0
    ):  # pylint: disable=unused-argument
        """Clean CE + per-token accuracy on held-out blocks."""
        x, y = batch
        with torch.no_grad():
            preds = self.model(x)
        per_tok = chunked_cross_entropy(preds, y)
        pad_mask = (y != 0).float()
        denom = pad_mask.sum().clamp(min=1.0)
        val_loss = (per_tok * pad_mask).sum() / denom
        if isinstance(preds, list):
            pred_ids = torch.cat([c.argmax(dim=-1) for c in preds], dim=1)
        else:
            pred_ids = preds.argmax(dim=-1)
        correct = ((pred_ids == y).float() * pad_mask).sum()
        val_acc = correct / denom

        if dataloader_idx < len(self.val_subset_names):
            name = self.val_subset_names[dataloader_idx]
        else:
            name = f"dl{dataloader_idx}"
        bucket = self._val_epoch_buckets.setdefault(
            name, {"loss_sum": 0.0, "correct_sum": 0.0, "tok_sum": 0.0}
        )
        bucket["loss_sum"] += float(val_loss.detach()) * float(denom.detach())
        bucket["correct_sum"] += float(correct.detach())
        bucket["tok_sum"] += float(denom.detach())

        if len(self.val_subset_names) <= 1:
            self.log("val_loss", val_loss, on_epoch=True, on_step=False, prog_bar=True)
            self.log("val_acc", val_acc, on_epoch=True, on_step=False, prog_bar=True)
        return {"loss": val_loss, "preds": pred_ids.detach(), "gt": y.detach()}

    def on_validation_epoch_end(self):
        """Emit per-subset means + aggregate val_loss / val_acc scalars (token-weighted across subsets, so the EarlyStopping monitor matches the loss the model is actually learning down). The single-subset case already logged the un-suffixed scalars in validation_step; skip the aggregate."""
        super().on_validation_epoch_end()
        if len(self.val_subset_names) <= 1:
            return
        total_loss = 0.0
        total_correct = 0.0
        total_tok = 0.0
        subset_means = []
        subset_accs = []
        for name in self.val_subset_names:
            bucket = self._val_epoch_buckets.get(name)
            if not bucket or bucket["tok_sum"] <= 0:
                continue
            sub_loss = bucket["loss_sum"] / bucket["tok_sum"]
            sub_acc = bucket["correct_sum"] / bucket["tok_sum"]
            self.log(
                f"val_loss/{name}",
                sub_loss,
                on_epoch=True,
                on_step=False,
                add_dataloader_idx=False,
            )
            self.log(
                f"val_acc/{name}",
                sub_acc,
                on_epoch=True,
                on_step=False,
                add_dataloader_idx=False,
            )
            total_loss += bucket["loss_sum"]
            total_correct += bucket["correct_sum"]
            total_tok += bucket["tok_sum"]
            subset_means.append(sub_loss)
            subset_accs.append(sub_acc)
        if total_tok <= 0:
            return
        self.log(
            "val_loss",
            total_loss / total_tok,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            add_dataloader_idx=False,
        )
        self.log(
            "val_acc",
            total_correct / total_tok,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            add_dataloader_idx=False,
        )
        if subset_means:
            self.log(
                "val_loss_macro",
                sum(subset_means) / len(subset_means),
                on_epoch=True,
                on_step=False,
                add_dataloader_idx=False,
            )
        if subset_accs:
            self.log(
                "val_acc_macro",
                sum(subset_accs) / len(subset_accs),
                on_epoch=True,
                on_step=False,
                add_dataloader_idx=False,
            )

    def configure_optimizers(self):
        return self.optimizer

    def set_optimizer_state(self, state):
        opts = self.optimizers()
        if not isinstance(opts, list):
            opts = [opts]

        for opt in opts:
            if isinstance(opt, OPTIMIZER):
                if state == "train":
                    opt.train()
                elif state == "eval":
                    opt.eval()
                else:
                    raise ValueError(f"Unknown train state {state}")

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self.set_optimizer_state("train")

    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        if not getattr(self.args, "log_embeddings", False):
            return
        embeddings = self.model.tok_embeddings.weight.data
        self.logger.experiment.add_embedding(
            embeddings,
            metadata=self.metadata,
            global_step=self.current_epoch,
            tag="epoch_embeddings",
        )

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self.set_optimizer_state("eval")
        self._reset_val_epoch_buckets()
