#!/usr/bin/env python3

import argparse
from datetime import timedelta
import os
import pytorch_lightning as pl
from torchtune.utils import get_logger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from regdataset import RegDataset, get_loader
from args import add_args
from model import get_model, SchedulerFreeModelCheckpoint


def train(model, dataloader, args, ckpt_path, logger):
    tb_logger = pl.loggers.TensorBoardLogger(args.tb_logs, "preframr")
    epoch_checkpoint_callback = SchedulerFreeModelCheckpoint(save_top_k=-1)
    time_checkpoint_callback = SchedulerFreeModelCheckpoint(
        save_top_k=-1,
        train_time_interval=timedelta(hours=args.ckpt_hours),
    )
    callbacks = [
        epoch_checkpoint_callback,
        time_checkpoint_callback,
    ]
    if args.stop_loss:
        callbacks.append(
            EarlyStopping(
                monitor="train_loss",
                mode="min",
                verbose=True,
                stopping_threshold=args.stop_loss,
                check_on_train_epoch_end=True,
            )
        )
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        default_root_dir=os.path.dirname(args.model_state),
        precision=args.trainer_precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=args.log_every_n_steps,
        enable_checkpointing=True,
        logger=tb_logger,
        callbacks=callbacks,
    )
    if ckpt_path:
        logger.info("resuming from %s", ckpt_path)
    trainer.fit(model, dataloader, ckpt_path=ckpt_path)
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
    dataloader = get_loader(args, dataset)
    model = get_model(dataset, args, logger)
    train(model, dataloader, args, ckpt_path, logger)


if __name__ == "__main__":
    main()
