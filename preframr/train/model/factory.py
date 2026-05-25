"""Model construction + device dispatch + the Lightning ModelCheckpoint subclass that calls schedule-free optimizer `.eval()` / `.train()` around `_save_checkpoint`. Pulls together the body factory + Lightning wrapper for the trainer to call."""

import copy

import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from preframr.train.model.bodies import OPTIMIZER
from preframr.train.model.lightning import Model


class SchedulerFreeModelCheckpoint(ModelCheckpoint):
    def _save_checkpoint(self, trainer, filepath):
        opts = trainer.optimizers
        if not isinstance(opts, list):
            opts = [opts]

        opt_modes = [
            (opt, opt.param_groups[0]["train_mode"])
            for opt in opts
            if isinstance(opt, OPTIMIZER)
        ]

        for opt, train_mode in opt_modes:
            if train_mode:
                opt.eval()

        super()._save_checkpoint(trainer, filepath)

        for opt, train_mode in opt_modes:
            if train_mode:
                opt.train()


def get_model(dataset, args, logger, args_override=None):
    if args_override:
        for k, v in args_override.items():
            setattr(args, k, v)
    model = Model(
        args,
        dataset.n_vocab,
        dataset.tokenizer.tokens.copy(),
        copy.deepcopy(dataset.tokenizer.tkmodel),
        dataset.tokenizer.token_metadata(),
        reg_widths=getattr(dataset, "reg_widths", {}),
    )
    _device, model_compiler = get_device(args, logger)
    return model_compiler(args, model)


def cpu_compile(args, model, option_keys=()):
    if not getattr(args, "compile", True):
        return model
    return torch.compile(
        model,
        options={k: True for k in option_keys},
    )


def cuda_compile(args, model):
    if not getattr(args, "compile", True):
        return model
    option_keys = ["epilogue_fusion"]
    if args.max_autotune:
        option_keys.append("max_autotune")
    if args.accumulate_grad_batches == 1:
        option_keys.append("triton.cudagraphs")
    return cpu_compile(args, model, option_keys)


def get_device(args, logger):
    if torch.cuda.is_available():
        logger.info("using cuda")
        torch.set_float32_matmul_precision(args.precision)
        return (
            torch.device("cuda:0"),
            cuda_compile,
        )
    if torch.xpu.is_available():
        logger.info("using xpu")
        return (
            torch.device("xpu"),
            cpu_compile,
        )
    logger.info("using cpu")
    return (torch.device("cpu"), cpu_compile)
