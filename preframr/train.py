#!/usr/bin/env python3

import argparse
from datetime import timedelta
import os
import pytorch_lightning as pl
from torchtune.utils import get_logger
from regdataset import RegDataset, get_loader
from args import add_args
from model import get_model, SchedulerFreeModelCheckpoint


def train(model, dataloader, args):
    tb_logger = pl.loggers.TensorBoardLogger(args.tb_logs, "preframr")
    epoch_checkpoint_callback = SchedulerFreeModelCheckpoint(save_top_k=-1)
    time_checkpoint_callback = SchedulerFreeModelCheckpoint(
        save_top_k=-1,
        train_time_interval=timedelta(hours=args.ckpt_hours),
    )
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        default_root_dir=os.path.dirname(args.model_state),
        precision=args.trainer_precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=args.log_every_n_steps,
        enable_checkpointing=True,
        logger=tb_logger,
        callbacks=[epoch_checkpoint_callback, time_checkpoint_callback],
    )
    ckpt_path = None
    if os.path.exists(args.model_state):
        ckpt_path = args.model_state
    trainer.fit(model, dataloader, ckpt_path=ckpt_path)
    return model


def main():
    parser = add_args(argparse.ArgumentParser())
    args = parser.parse_args()
    logger = get_logger("INFO")
    dataset = RegDataset(args, logger=logger)
    dataloader = get_loader(args, dataset)
    model = get_model(dataset, args, logger)
    train(model, dataloader, args)


if __name__ == "__main__":
    main()
