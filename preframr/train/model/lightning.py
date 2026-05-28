"""`Model(LightningModule)` -- pytorch-lightning wrapper that owns the torchtune body, the per-tier heads (optional), the loss aggregation, training/validation steps. This is the only file in `preframr.train.model` that imports `pytorch_lightning`."""

import torch
from pytorch_lightning import LightningModule

from preframr.train.model.bodies import MODEL_GETTERS, OPTIMIZER
from preframr.train.model.heads import PerTierHeads, per_tier_unified_log_p
from preframr.train.model.losses import (
    _build_vocab_frame_weight,
    chunked_cross_entropy,
    content_contrastive_loss,
)
from preframr.train.model.tier_map import (
    _CONTENT_TIER_ID,
    _LOSS_TIER_ORDER,
    _N_LOSS_TIERS,
    _STRUCTURAL_TIER_ID,
    _build_tier_vocab_partition,
    _build_vocab_class_weight,
    _build_vocab_onset_weight,
    _build_vocab_tier_id,
)


class Model(LightningModule):
    def __init__(self, args, n_vocab, tokens, tkmodel, metadata, reg_widths=None):
        super().__init__()
        self.args = args
        self.n_vocab = n_vocab
        self.tokens = tokens
        self.tkmodel = None
        self.metadata = metadata
        self.reg_widths = dict(reg_widths) if reg_widths else {}
        if tkmodel:
            self.tkmodel = tkmodel.to_str()
        self.save_hyperparameters(
            "args", "n_vocab", "tokens", "tkmodel", "metadata", "reg_widths"
        )
        self.model = MODEL_GETTERS[args.model](n_vocab, args)
        num_output_chunks = int(getattr(args, "num_output_chunks", 0) or 0)
        if num_output_chunks == 0 and n_vocab >= 32768:
            num_output_chunks = 8
        if num_output_chunks > 0 and hasattr(self.model, "set_num_output_chunks"):
            self.model.set_num_output_chunks(num_output_chunks)
        self.num_output_chunks = num_output_chunks
        if getattr(args, "token_weighting", True):
            frame_weight = _build_vocab_frame_weight(args, n_vocab, tokens, tkmodel)
        else:
            frame_weight = torch.ones(n_vocab, dtype=torch.float32)
        self.register_buffer(
            "vocab_frame_weight",
            frame_weight,
            persistent=False,
        )
        class_weight = _build_vocab_class_weight(args, n_vocab, tokens, tkmodel)
        self.register_buffer(
            "vocab_class_weight",
            class_weight,
            persistent=False,
        )
        self.register_buffer(
            "vocab_onset_weight",
            _build_vocab_onset_weight(args, n_vocab, tokens, tkmodel),
            persistent=False,
        )
        tier_ids = _build_vocab_tier_id(args, n_vocab, tokens, tkmodel)
        self.register_buffer(
            "vocab_tier_id",
            tier_ids,
            persistent=False,
        )
        self.per_tier_heads_on = bool(getattr(args, "per_tier_heads", False))
        if self.per_tier_heads_on:
            partition = _build_tier_vocab_partition(args, n_vocab, tokens, tkmodel)
            self._tier_partition = {}
            for tier_name, (in_tier, full_to_local) in partition.items():
                self.register_buffer(f"tier_in_{tier_name}", in_tier, persistent=False)
                self.register_buffer(
                    f"tier_full_to_local_{tier_name}", full_to_local, persistent=False
                )
                self._tier_partition[tier_name] = (
                    f"tier_in_{tier_name}",
                    f"tier_full_to_local_{tier_name}",
                )
            mos_k = int(getattr(args, "per_tier_content_mos_k", 4))
            cluster_head_args = None
            if bool(getattr(args, "content_cluster_head", False)):
                index_path = str(getattr(args, "content_cluster_index", "") or "")
                if not index_path:
                    raise ValueError(
                        "--content-cluster-head requires --content-cluster-index"
                    )
                cluster_head_args = {
                    "c": int(getattr(args, "content_cluster_c", 256)),
                    "index_path": index_path,
                }
                mos_k = 0
            diffusion_head_args = None
            if bool(getattr(args, "content_diffusion", False)):
                diffusion_head_args = {
                    "t_max": int(getattr(args, "content_diffusion_t", 8)),
                    "d_time": int(getattr(args, "content_diffusion_d_time", 128)),
                }
                mos_k = 0
            self.per_tier_heads = PerTierHeads(
                args.embed,
                partition,
                mos_k=mos_k,
                cluster_head_args=cluster_head_args,
                diffusion_head_args=diffusion_head_args,
            )
            self.content_diffusion_on = diffusion_head_args is not None
            self.per_tier_mos_entropy_lambda = float(
                getattr(args, "per_tier_mos_entropy_lambda", 0.0)
            )
            if hasattr(self.model, "set_num_output_chunks"):
                self.model.set_num_output_chunks(0)
            self.num_output_chunks = 0
            self._patch_decoder_for_per_tier()
        self.learnable_class_loss = bool(getattr(args, "learnable_class_loss", False))
        if self.learnable_class_loss or self.per_tier_heads_on:
            self.log_sigma_per_tier = torch.nn.Parameter(torch.zeros(_N_LOSS_TIERS))
        if self.per_tier_heads_on:
            self.log_sigma_router = torch.nn.Parameter(torch.zeros(()))
        self.optimizer = OPTIMIZER(
            self.parameters(),
            lr=self.args.learning_rate,
            foreach=True,
            weight_decay=self.args.weight_decay,
        )
        self.structural_loss_lambda = float(
            getattr(args, "structural_loss_lambda", 0.0)
        )
        if self.structural_loss_lambda > 0.0:
            from preframr.train.structural_loss import (  # pylint: disable=import-outside-toplevel
                StructuralLoss,
            )

            self.structural_loss_fn = StructuralLoss(args, n_vocab, tokens, tkmodel)
        else:
            self.structural_loss_fn = None
        self.infonce_content_loss_weight = float(
            getattr(args, "infonce_content_loss_weight", 0.0)
        )
        self.infonce_distractors = int(getattr(args, "infonce_distractors", 32))
        self.mask_structural_tier_loss = bool(
            getattr(args, "mask_structural_tier_loss", False)
        )
        self.val_subset_names = ["val"]
        self._val_epoch_buckets = {}

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        if self.per_tier_heads_on:
            return self._per_tier_training_step(x, y)
        preds = self.model(x)
        per_tok = chunked_cross_entropy(
            preds,
            y,
            label_smoothing=self.args.label_smoothing,
        )
        pad_mask = (y != 0).float()
        if self.mask_structural_tier_loss:
            pad_mask = pad_mask * (self.vocab_tier_id[y] != _STRUCTURAL_TIER_ID).float()
        per_tok = per_tok * pad_mask
        onset_w = self.vocab_onset_weight[y]
        if self.learnable_class_loss:
            tier_ids = self.vocab_tier_id[y]
            log_sigma = self.log_sigma_per_tier[tier_ids]
            sigma_sq_inv_half = torch.exp(-2.0 * log_sigma) * 0.5
            base_w = self.vocab_frame_weight[y] * onset_w * pad_mask
            adjusted = per_tok * sigma_sq_inv_half + log_sigma * pad_mask
            loss = (adjusted * base_w).sum() / base_w.sum().clamp(min=1.0)
        else:
            weights = self.vocab_frame_weight[y] * self.vocab_class_weight[y] * onset_w
            weights = weights * pad_mask
            loss = (per_tok * weights).sum() / weights.sum().clamp(min=1.0)
        if self.structural_loss_fn is not None:
            aux = self.structural_loss_fn.compute(preds, y, pad_mask)
            self.log("train_struct_aux", aux, on_epoch=True, on_step=True)
            loss = loss + self.structural_loss_lambda * aux
        if self.infonce_content_loss_weight > 0.0:
            content_mask = (self.vocab_tier_id[y] == _CONTENT_TIER_ID) & (y != 0)
            infonce_aux = content_contrastive_loss(
                preds, y, content_mask, k=self.infonce_distractors
            )
            loss = loss + self.infonce_content_loss_weight * infonce_aux
        return loss

    def _per_tier_training_step(self, x, y):
        h_normed = self.model(x)
        head_outputs = self.per_tier_heads(h_normed)
        pad_mask = y != 0
        gt_tier_id = self.vocab_tier_id[y]
        router_logits = head_outputs["router"]
        active_mask = torch.tensor(
            [getattr(self, f"tier_in_{t}").numel() > 0 for t in _LOSS_TIER_ORDER],
            dtype=torch.bool,
            device=router_logits.device,
        )
        router_logits_masked = router_logits.masked_fill(~active_mask, float("-inf"))
        flat_pad = pad_mask.view(-1)
        if flat_pad.any():
            router_loss = torch.nn.functional.cross_entropy(
                router_logits_masked.view(-1, _N_LOSS_TIERS)[flat_pad],
                gt_tier_id.view(-1)[flat_pad],
            )
        else:
            router_loss = router_logits.new_zeros(())
        per_tier_losses = {}
        for tier_id, tier_name in enumerate(_LOSS_TIER_ORDER):
            tier_mask = (gt_tier_id == tier_id) & pad_mask
            if not tier_mask.any():
                per_tier_losses[tier_name] = h_normed.new_zeros(())
                continue
            tier_out = head_outputs[tier_name]
            ftl_buf = getattr(self, f"tier_full_to_local_{tier_name}")
            local_gt = ftl_buf[y[tier_mask]]
            tier_out_selected = tier_out[tier_mask]
            if tier_name == "content" and self.content_diffusion_on:
                from preframr.train.model.losses_diffusion import (  # pylint: disable=import-outside-toplevel
                    discrete_diffusion_content_loss,
                )

                head_module = self.per_tier_heads.heads["content"]
                per_tier_losses[tier_name] = discrete_diffusion_content_loss(
                    head_module,
                    h_normed,
                    y,
                    tier_mask,
                    ftl_buf,
                    t_max=head_module.t_max,
                )
            elif tier_name == "content" and self.per_tier_heads.mos_k > 0:
                per_tier_losses[tier_name] = torch.nn.functional.nll_loss(
                    tier_out_selected, local_gt
                )
            else:
                per_tier_losses[tier_name] = torch.nn.functional.cross_entropy(
                    tier_out_selected, local_gt
                )
        loss = router_logits.new_zeros(())
        for tier_id, tier_name in enumerate(_LOSS_TIER_ORDER):
            ls = self.log_sigma_per_tier[tier_id]
            loss = loss + torch.exp(-2.0 * ls) * 0.5 * per_tier_losses[tier_name] + ls
        lr = self.log_sigma_router
        loss = loss + torch.exp(-2.0 * lr) * 0.5 * router_loss + lr
        if (
            self.per_tier_mos_entropy_lambda > 0.0
            and head_outputs.get("mos_gate_log_p") is not None
        ):
            content_mask = (gt_tier_id == _CONTENT_TIER_ID) & pad_mask
            if content_mask.any():
                gate_log_p_c = head_outputs["mos_gate_log_p"][content_mask]
                gate_p_c = gate_log_p_c.exp()
                entropy = -(gate_p_c * gate_log_p_c).sum(dim=-1).mean()
                loss = loss - self.per_tier_mos_entropy_lambda * entropy
        self._log_per_tier_step_metrics(router_loss, per_tier_losses)
        return loss

    @torch._dynamo.disable
    def _log_per_tier_step_metrics(self, router_loss, per_tier_losses):
        self.log("train_router_loss", router_loss, on_epoch=True, on_step=False)
        for tier_name in _LOSS_TIER_ORDER:
            self.log(
                f"train_{tier_name}_loss",
                per_tier_losses[tier_name],
                on_epoch=True,
                on_step=False,
            )

    def _patch_decoder_for_per_tier(self):
        import types  # pylint: disable=import-outside-toplevel

        parent = self

        def unembed(self, h):  # pylint: disable=function-redefined
            h_normed = self.norm(h)
            if parent.training:
                return h_normed
            partition = {
                tier_name: (
                    getattr(parent, in_buf),
                    getattr(parent, ftl_buf),
                )
                for tier_name, (in_buf, ftl_buf) in parent._tier_partition.items()
            }
            with torch.amp.autocast(
                h.device.type, dtype=torch.bfloat16, enabled=h.is_cuda
            ):
                head_outputs = parent.per_tier_heads(h_normed)
                return per_tier_unified_log_p(head_outputs, partition, parent.n_vocab)

        self.model.unembed = types.MethodType(unembed, self.model)

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
        if self.per_tier_heads_on:
            per_tok = -preds.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        else:
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
        if self.learnable_class_loss:
            # pylint: disable=not-callable
            sigmas = torch.exp(self.log_sigma_per_tier.detach())
            # pylint: enable=not-callable
            for tier_id, name in enumerate(_LOSS_TIER_ORDER):
                self.log(f"sigma_{name}", sigmas[tier_id], on_epoch=True, on_step=False)
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
