#!/usr/bin/env python3

import argparse
from datetime import timedelta
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from preframr.args import add_args
from preframr.train.regdataset import RegDataset, get_loader, get_val_loader
from preframr.train.model import (
    build_op_map,
    build_tier_map,
    get_model,
    SchedulerFreeModelCheckpoint,
)
from preframr.utils import get_logger


def _build_audit_prompts(val_dataloader, n_prompts, prompt_len):
    """Collect up to n_prompts (truncated to prompt_len) from val_dataloader."""
    if val_dataloader is None or n_prompts <= 0:
        return []
    loaders = val_dataloader if isinstance(val_dataloader, list) else [val_dataloader]
    prompts = []
    for loader in loaders:
        for batch in loader:
            x, _ = batch
            for i in range(x.shape[0]):
                if len(prompts) >= n_prompts:
                    return prompts
                prompts.append(x[i, :prompt_len].tolist())
        if len(prompts) >= n_prompts:
            break
    return prompts


def _make_generate_fn(pad_id: int = 0):
    """Return a callable (pl_module, prompt_list, n) -> generated_tokens_list."""
    import torch
    from torchtune.generation import generate

    def _gen(pl_module, prompt, n):
        device = next(pl_module.parameters()).device
        prompt_t = torch.tensor(prompt, dtype=torch.long, device=device).unsqueeze(0)
        prior_chunks = getattr(pl_module, "num_output_chunks", 0) or 0
        if prior_chunks and hasattr(pl_module.model, "set_num_output_chunks"):
            pl_module.model.set_num_output_chunks(0)
        try:
            out, _logits = generate(
                pl_module.model,
                prompt_t,
                max_generated_tokens=n,
                pad_id=pad_id,
                temperature=1.0,
                top_k=None,
            )
        finally:
            if prior_chunks and hasattr(pl_module.model, "set_num_output_chunks"):
                pl_module.model.set_num_output_chunks(prior_chunks)
        return out.squeeze(0)[-n:].tolist()

    return _gen


def _maybe_generalization_gate(args, model, val_dataloader, logger):
    """Build the GeneralizationGate callback when --generalization-gate is on."""
    if not getattr(args, "generalization_gate", False) or val_dataloader is None:
        return None
    from preframr.train.generalization_gate import GateThresholds, GeneralizationGate

    tier_map = build_tier_map(args, model.n_vocab, model.tokens, model.tkmodel)
    op_map = build_op_map(args, model.n_vocab, model.tokens, model.tkmodel)
    audit_prompts = _build_audit_prompts(
        val_dataloader,
        n_prompts=args.gate_audit_prompts,
        prompt_len=args.gate_audit_prompt_len,
    )
    logger.info(
        "GeneralizationGate: %d tier entries, %d op classes, %d audit prompts (len %d)",
        len(tier_map),
        len(set(op_map.values())),
        len(audit_prompts),
        args.gate_audit_prompt_len,
    )
    return GeneralizationGate(
        tier_map=tier_map,
        op_map=op_map,
        audit_prompts=audit_prompts,
        generate_fn=_make_generate_fn(),
        generate_n=args.gate_generate_n,
        generate_every_k_epochs=args.gate_generate_every_k,
        thresholds=GateThresholds(),
    )


def train(
    model, dataloader, val_dataloader, args, ckpt_path, logger, val_subset_names=None
):
    tb_logger = pl.loggers.TensorBoardLogger(args.tb_logs, "preframr")
    epoch_checkpoint_callback = SchedulerFreeModelCheckpoint(
        save_top_k=1, save_last=False
    )
    callbacks = [
        epoch_checkpoint_callback,
    ]
    if args.ckpt_hours:
        callbacks.append(
            SchedulerFreeModelCheckpoint(
                save_top_k=1,
                save_last=False,
                train_time_interval=timedelta(hours=args.ckpt_hours),
            )
        )
    if args.stop_loss or args.stop_delta:
        kwargs = {}
        if args.stop_loss:
            kwargs["stopping_threshold"] = args.stop_loss
        if args.stop_delta:
            kwargs["min_delta"] = args.stop_delta
        callbacks.append(
            EarlyStopping(
                monitor="train_loss",
                mode="min",
                verbose=True,
                **kwargs,
            )
        )
    if val_dataloader is not None:
        if args.early_stop_patience or args.early_stop_min_delta:
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    mode="min",
                    patience=max(args.early_stop_patience, 1),
                    min_delta=args.early_stop_min_delta,
                    verbose=True,
                )
            )
        callbacks.append(
            ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                filename="best-{epoch}-{val_loss:.4f}",
            )
        )
        gate = _maybe_generalization_gate(args, model, val_dataloader, logger)
        if gate is not None:
            callbacks.append(gate)
    fit_kwargs = {}
    if val_dataloader is not None and args.val_check_every:
        fit_kwargs["check_val_every_n_epoch"] = args.val_check_every
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        default_root_dir=os.path.dirname(args.model_state),
        precision=args.trainer_precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=args.log_every_n_steps,
        enable_checkpointing=True,
        logger=tb_logger,
        callbacks=callbacks,
        **fit_kwargs,
    )
    if ckpt_path:
        logger.info("resuming from %s", ckpt_path)
    if val_subset_names is not None:
        model.val_subset_names = list(val_subset_names)
    trainer.fit(
        model,
        train_dataloaders=dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=ckpt_path,
    )
    return model


def _require_pretokenized(args):
    """Train.py expects ``stftokenize.py`` to have already produced the
    Unigram tkmodel + alphabet. Without this guard, a missing tokenizer
    silently falls through to ``RegDataset.make_tokens``, which trains
    the tokenizer inline -- minutes on toy corpora, hours on the full
    78K MUSICIANS dump cache, and almost never what the caller wants.
    """
    if not args.tkvocab:
        return
    missing = []
    for label, path in (
        ("--tkmodel", args.tkmodel),
        ("--token-csv", args.token_csv),
    ):
        if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
            missing.append(f"{label}={path!r}")
    if missing:
        raise SystemExit(
            f"--tkvocab={args.tkvocab} requires a pre-built tokenizer "
            f"but the following inputs are missing or empty: "
            f"{', '.join(missing)}. Run ``stftokenize.py`` first to "
            "build the alphabet + Unigram model, or pass --tkvocab=0 "
            "for the raw-alphabet path."
        )


def main():
    from preframr.args import apply_pipeline_spec_to_args

    parser = add_args(argparse.ArgumentParser())
    args = parser.parse_args()
    apply_pipeline_spec_to_args(args)
    logger = get_logger("INFO")
    ckpt_path = None
    if args.model_state:
        ckpt_path = args.model_state
        if not os.path.exists(ckpt_path):
            raise ValueError("No such checkpoint %s" % ckpt_path)
        logger.info("Will resume from %s", ckpt_path)
    _require_pretokenized(args)
    dataset = RegDataset(args, logger=logger)
    dataset.preload()
    assert dataset.tokenizer.token_metadata()
    dataloader = get_loader(args, dataset)
    val_dataloader, val_subset_names = get_val_loader(args, dataset)
    model = get_model(dataset, args, logger)
    train(
        model,
        dataloader,
        val_dataloader,
        args,
        ckpt_path,
        logger,
        val_subset_names=val_subset_names,
    )


if __name__ == "__main__":
    main()
