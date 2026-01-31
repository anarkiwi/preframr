import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from schedulefree import AdamWScheduleFree
from torchtune.models.gemma._component_builders import gemma
from torchtune.models.llama2._component_builders import llama2
from torchtune.models.llama3_2._component_builders import llama3_2
from torchtune.models.mistral._component_builders import mistral
from torchtune.models.phi3._component_builders import phi3
from torchtune.models.qwen2._component_builders import qwen2

MODEL_PRECISION = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}
OPTIMIZER = AdamWScheduleFree


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


def get_gemma(n_vocab, args):
    intermediate = args.intermediate if args.intermediate else args.embed
    head_dim = args.embed // args.heads
    kv_heads = args.kv_heads if args.kv_heads else args.heads

    return gemma(
        vocab_size=n_vocab,
        num_layers=args.layers,
        num_heads=args.heads,
        num_kv_heads=kv_heads,
        embed_dim=args.embed,
        head_dim=head_dim,
        intermediate_dim=intermediate,
        max_seq_len=args.max_seq_len,
        attn_dropout=args.attn_dropout,
        norm_eps=args.norm_eps,
        rope_base=args.rope_base,
    )


def get_gemma2(n_vocab, args):
    intermediate = args.intermediate if args.intermediate else args.embed
    head_dim = args.embed // args.heads
    kv_heads = args.kv_heads if args.kv_heads else args.heads

    return gemma(
        vocab_size=n_vocab,
        num_layers=args.layers,
        num_heads=args.heads,
        num_kv_heads=kv_heads,
        embed_dim=args.embed,
        head_dim=head_dim,
        intermediate_dim=intermediate,
        max_seq_len=args.max_seq_len,
        attn_dropout=args.attn_dropout,
        norm_eps=args.norm_eps,
        rope_base=args.rope_base,
    )


def get_llama2(n_vocab, args):
    return llama2(
        vocab_size=n_vocab,
        num_layers=args.layers,
        num_heads=args.heads,
        num_kv_heads=args.kv_heads,
        embed_dim=args.embed,
        max_seq_len=args.max_seq_len,
        attn_dropout=args.attn_dropout,
        norm_eps=args.norm_eps,
        rope_base=args.rope_base,
    )


def get_llama3_2(n_vocab, args):
    return llama3_2(
        vocab_size=n_vocab,
        num_layers=args.layers,
        num_heads=args.heads,
        num_kv_heads=args.kv_heads,
        embed_dim=args.embed,
        max_seq_len=args.max_seq_len,
        attn_dropout=args.attn_dropout,
        norm_eps=args.norm_eps,
        rope_base=args.rope_base,
        scale_factor=args.rope_scale,
        tie_word_embeddings=args.tie_word_embeddings,
    )


def get_mistral(n_vocab, args):
    intermediate = args.intermediate if args.intermediate else args.embed

    return mistral(
        vocab_size=n_vocab,
        num_layers=args.layers,
        num_heads=args.heads,
        num_kv_heads=args.kv_heads,
        embed_dim=args.embed,
        intermediate_dim=intermediate,
        max_seq_len=args.max_seq_len,
        attn_dropout=args.attn_dropout,
        norm_eps=args.norm_eps,
        rope_base=args.rope_base,
    )


def get_phi3(n_vocab, args):
    intermediate = args.intermediate if args.intermediate else args.embed

    return phi3(
        vocab_size=n_vocab,
        num_layers=args.layers,
        num_heads=args.heads,
        num_kv_heads=args.kv_heads,
        embed_dim=args.embed,
        intermediate_dim=intermediate,
        max_seq_len=args.max_seq_len,
        attn_dropout=args.attn_dropout,
        norm_eps=args.norm_eps,
        rope_base=args.rope_base,
    )


def get_qwen2(n_vocab, args):
    intermediate = args.intermediate if args.intermediate else args.embed

    return qwen2(
        vocab_size=n_vocab,
        num_layers=args.layers,
        num_heads=args.heads,
        num_kv_heads=args.kv_heads,
        embed_dim=args.embed,
        intermediate_dim=intermediate,
        max_seq_len=args.max_seq_len,
        attn_dropout=args.attn_dropout,
        norm_eps=args.norm_eps,
        rope_base=args.rope_base,
    )


MODEL_GETTERS = {
    "gemma": get_gemma,
    "gemma2": get_gemma2,
    "llama2": get_llama2,
    "mistral": get_mistral,
    "phi3": get_phi3,
    "qwen2": get_qwen2,
    "llama3_2": get_llama3_2,
}


class Model(LightningModule):
    def __init__(self, args, n_vocab, tokens, tkmodel):
        super().__init__()
        self.args = args
        self.n_vocab = n_vocab
        self.tokens = tokens
        self.tkmodel = None
        if tkmodel:
            self.tkmodel = tkmodel.to_str()
        self.save_hyperparameters("args", "n_vocab", "tokens", "tkmodel")
        self.model = MODEL_GETTERS[args.model](n_vocab, args)
        self.optimizer = OPTIMIZER(
            self.parameters(),
            lr=self.args.learning_rate,
            foreach=True,
            weight_decay=self.args.weight_decay,
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x)
        swapped_preds = preds.swapaxes(1, 2)
        # acc = torchmetrics.functional.classification.accuracy(
        #    swapped_preds, y, task="multiclass", num_classes=swapped_preds.shape[1]
        # )
        ce_loss = torch.nn.functional.cross_entropy(
            input=swapped_preds,
            target=y,
            reduction="none",
            label_smoothing=self.args.label_smoothing,
        )
        focal_loss = (
            self.args.focal_alpha
            * (1 - torch.exp(-ce_loss)) ** self.args.focal_gamma
            * ce_loss
        )
        loss = focal_loss.mean()

        # self.log("train_loss", loss, on_epoch=True, on_step=True)
        # self.log("train_acc", acc, on_epoch=True, on_step=True)
        return loss

    def on_before_backward(self, loss):
        self.log("train_loss", loss, on_epoch=True, on_step=True)

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

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self.set_optimizer_state("eval")


def get_model(dataset, args, logger, args_override=None):
    if args_override:
        for k, v in args_override.items():
            setattr(args, k, v)
    model = Model(
        args, dataset.n_vocab, dataset.tokenizer.tokens, dataset.tokenizer.tkmodel
    )
    _device, model_compiler = get_device(args, logger)
    return model_compiler(args, model)


def cpu_compile(args, model, option_keys=[]):
    return torch.compile(
        model,
        options={k: True for k in option_keys},
    )


def cuda_compile(args, model):
    option_keys = ["epilogue_fusion", "max_autotune"]
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
