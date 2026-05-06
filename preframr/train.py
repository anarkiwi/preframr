#!/usr/bin/env python3

import argparse
from datetime import timedelta
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from preframr.args import add_args
from preframr.regdataset import RegDataset, get_loader, get_val_loader
from preframr.model import get_model, SchedulerFreeModelCheckpoint
from preframr.utils import get_logger


def train(model, dataloader, val_dataloader, args, ckpt_path, logger):
    tb_logger = pl.loggers.TensorBoardLogger(args.tb_logs, "preframr")
    epoch_checkpoint_callback = SchedulerFreeModelCheckpoint(save_top_k=-1)
    callbacks = [
        epoch_checkpoint_callback,
    ]
    if args.ckpt_hours:
        callbacks.append(
            SchedulerFreeModelCheckpoint(
                save_top_k=-1,
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
        # Val EarlyStopping fires when ``val_loss`` stops improving --
        # the generalisation signal, separate from the train_loss
        # stopping_threshold above (which is a memorisation gate).
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
        # Best-by-val checkpoint so the eval / predict stage can pick
        # the model that generalised best, not the latest by epoch.
        callbacks.append(
            ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                filename="best-{epoch}-{val_loss:.4f}",
            )
        )
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
    trainer.fit(
        model,
        train_dataloaders=dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=ckpt_path,
    )
    return model


def main():
    parser = add_args(argparse.ArgumentParser())
    args = parser.parse_args()
    logger = get_logger("INFO")
    ckpt_path = None
    if args.model_state:
        ckpt_path = args.model_state
        if not os.path.exists(ckpt_path):
            raise ValueError("No such checkpoint %s" % ckpt_path)
        logger.info("Will resume from %s", ckpt_path)
    dataset = RegDataset(args, logger=logger)
    dataset.preload()
    assert dataset.tokenizer.token_metadata()
    dataloader = get_loader(args, dataset)
    val_dataloader = get_val_loader(args, dataset)
    model = get_model(dataset, args, logger)
    train(model, dataloader, val_dataloader, args, ckpt_path, logger)


if __name__ == "__main__":
    main()
