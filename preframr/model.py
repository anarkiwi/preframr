import copy
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from schedulefree import AdamWScheduleFree
import pyarrow

pyarrow.PyExtensionType = pyarrow.ExtensionType
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


def _build_vocab_frame_weight(args, n_vocab, tokens, tkmodel):
    """Per-vocab-id audio-frame weight used to scale per-token CE loss.

    Macro tokens that expand to multiple frames get proportionally higher
    weight, so a misprediction on a BACK_REF/DO_LOOP/DELAY costs more than
    a misprediction on a single within-frame write. Each base token in a
    Unigram subword contributes:

      * BACK_REF      -> length frames (from packed val payload)
      * DO_LOOP BEGIN -> iteration count N (rough proxy for N x body)
      * DELAY_REG     -> N empty frames (from val)
      * FRAME_REG     -> 1 frame
      * everything else (within-frame writes, DO_LOOP END) -> 0

    Subword weights are summed across constituents and floored to 1.0 so
    every token contributes at least the default weighting.
    """
    weights = torch.ones(n_vocab, dtype=torch.float32)
    if tokens is None or len(tokens) == 0:
        return weights
    # Lazy import to avoid a circular reglogparser <- macros <- ... cycle.
    from preframr.macros import _unpack_back_ref
    from preframr.regtokenizer import RegTokenizer
    from preframr.stfconstants import (
        BACK_REF_OP,
        DELAY_REG,
        DO_LOOP_OP,
        FRAME_REG,
    )

    # RegTokenizer.load expects a JSON string (it calls Tokenizer.from_str).
    # Callers may pass either the live Tokenizer object (fresh training) or
    # the already-serialized string (after checkpoint reload), so normalize.
    if tkmodel is not None and not isinstance(tkmodel, str):
        tkmodel = tkmodel.to_str()
    rt = RegTokenizer(args, tokens=tokens)
    rt.load(tkmodel, tokens)
    n_base = len(tokens)
    for vid in range(n_vocab):
        if rt.tkmodel:
            base_ids = rt.decode([vid])
        else:
            base_ids = [vid]
        w = 0.0
        for bid in base_ids:
            bid = int(bid)
            if bid >= n_base:
                continue
            row = tokens.iloc[bid]
            reg = int(row.reg)
            op = int(row.op)
            val = int(row.val)
            subreg = int(row.subreg)
            if op == BACK_REF_OP:
                _, length = _unpack_back_ref(val)
                w += length
            elif op == DO_LOOP_OP and subreg == 0:
                w += val
            elif reg == DELAY_REG:
                w += val
            elif reg == FRAME_REG:
                w += 1.0
            # within-frame writes (SET/DIFF/REPEAT/FLIP/PWM/...) and
            # DO_LOOP END contribute 0 -- the floor below covers them.
        if w > 0.0:
            weights[vid] = w
    return weights


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
        self.l1_lambda = self.args.l1_lambda
        self.optimizer = OPTIMIZER(
            self.parameters(),
            lr=self.args.learning_rate,
            foreach=True,
            weight_decay=self.args.weight_decay,
        )
        # Audio-frame weight per vocab id; used to scale per-token CE so
        # macro tokens expanding to many frames (BACK_REF, DO_LOOP, DELAY)
        # cost more when mispredicted than single-frame writes.
        self.register_buffer(
            "vocab_frame_weight",
            _build_vocab_frame_weight(args, n_vocab, tokens, tkmodel),
            persistent=False,
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x)
        swapped_preds = preds.swapaxes(1, 2)
        # Per-token CE so we can post-weight by audio-frame impact and
        # optionally apply focal scaling.
        per_tok = torch.nn.functional.cross_entropy(
            input=swapped_preds,
            target=y,
            reduction="none",
            label_smoothing=self.args.label_smoothing,
        )
        # Focal loss: alpha * (1 - p)^gamma * CE, where p is the predicted
        # probability of the true class (= exp(-CE) when label_smoothing=0).
        # focal_gamma=0 (default) is a no-op.
        if self.args.focal_gamma:
            with torch.no_grad():
                p = (-per_tok).exp().clamp(max=1.0)
                focal = self.args.focal_alpha * (1.0 - p).pow(self.args.focal_gamma)
            per_tok = per_tok * focal
        # Audio-frame weighting: penalize wrong predictions on multi-frame
        # macro tokens (BACK_REF, DO_LOOP_BEGIN, DELAY_REG) proportionally
        # to the audio they would have produced.
        weights = self.vocab_frame_weight[y]
        # Pad mask: BlockMapper pads short songs with token id 0 (the
        # Unigram tokenizer's pad/unk slot) so songs ≤ seq_len fit a
        # fixed-size block. Without masking, cross-entropy at pad
        # positions would teach the model to emit pad tokens, which is
        # nonsense at inference. Zero out both the per-token CE and the
        # frame-weight at pads so they contribute nothing to the loss.
        pad_mask = (y != 0).float()
        per_tok = per_tok * pad_mask
        weights = weights * pad_mask
        loss = (per_tok * weights).sum() / weights.sum().clamp(min=1.0)
        if self.l1_lambda:
            l1_norm = self.model.tok_embeddings.weight.abs().sum()
            loss = loss + self.l1_lambda * l1_norm
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

    def on_train_epoch_end(self):
        super().on_train_epoch_end()
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
    )
    _device, model_compiler = get_device(args, logger)
    return model_compiler(args, model)


def cpu_compile(args, model, option_keys=[]):
    return torch.compile(
        model,
        options={k: True for k in option_keys},
    )


def cuda_compile(args, model):
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
