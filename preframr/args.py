import argparse

from preframr.train.model import MODEL_GETTERS, MODEL_PRECISION


def add_args(parser):
    parser.add_argument(
        "reglog",
        type=str,
        default="",
        nargs="?",
    )
    parser.add_argument(
        "--reglogs",
        type=str,
        default="/scratch/preframr/training-dumps/**/*dump.parquet",
    )
    parser.add_argument("--eval-reglogs", type=str, default="")
    parser.add_argument(
        "--model-state",
        type=str,
        default="",
    )
    parser.add_argument("--tb-logs", type=str, default="/scratch/preframr/tb_logs")
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--prompt-seq-len", type=int, default=2048)
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tkvocab", type=int, default=4096)
    parser.add_argument("--wav", type=str, default="/scratch/preframr/preframr.wav")
    parser.add_argument(
        "--play",
        action="store_true",
        help=(
            "after the wav write, also stream the prediction through the "
            "real-time resid driver via ``preframr_audio.audio_driver.play_samples``. "
            "Requires the ``sounddevice`` PortAudio binding; logs and skips "
            "playback if it isn't installed (wav write still completes)."
        ),
    )
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument(
        "--predict-dump",
        type=str,
        default=None,
        help="Write the prediction window of the audio-ready df as parquet "
        "(register-level rows, description=1 only). For automated melody-quality "
        "scoring downstream of predict.",
    )
    parser.add_argument("--dataset-csv", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default="unigram")
    parser.add_argument("--token-csv", type=str, default="/scratch/preframr/token.csv")
    parser.add_argument("--tkmodel", type=str, default="/scratch/preframr/tkmodel.json")
    parser.add_argument(
        "--df-map-csv",
        type=str,
        default="/scratch/preframr/dataset-map.csv",
    )
    parser.add_argument("--shuffle", type=float, default=0.05)
    parser.add_argument("--max-files", type=int, default=2048)
    parser.add_argument("--min-song-tokens", type=int, default=256)
    parser.add_argument("--min-irq", type=int, default=int(1.5e4))
    parser.add_argument("--max-irq", type=int, default=int(2.5e4))
    parser.add_argument("--exclude-list", type=str, default=None)
    parser.add_argument("--diffq", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--embed", type=int, default=512)
    parser.add_argument("--intermediate", type=int, default=1408)
    parser.add_argument("--norm-eps", type=float, default=1e-5)
    parser.add_argument("--rope-base", type=float, default=1e4)
    parser.add_argument("--rope-scale", type=float, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--start-seq", type=int, default=0)
    parser.add_argument("--start-block", type=int, default=0)
    parser.add_argument("--predict-set", type=str, default="train")
    parser.add_argument("--attn-dropout", type=float, default=0.1)
    parser.add_argument(
        "--model", choices=list(MODEL_GETTERS.keys()), default="llama3_2"
    )
    parser.add_argument(
        "--precision",
        choices=["highest", "high", "medium"],
        default="high",
    )
    parser.add_argument("--trainer-precision", type=str, default="bf16-mixed")
    parser.add_argument(
        "--model-precision",
        type=str,
        default="bfloat16",
        choices=list(MODEL_PRECISION.keys()),
    )
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--accumulate-grad-batches", type=int, default=4)
    parser.add_argument("--log-every-n-steps", type=int, default=2)
    parser.add_argument("--max-perm", type=int, default=3)
    parser.add_argument("--ckpt-hours", type=int, default=12)
    parser.add_argument("--min-acc", type=float, default=0)
    parser.add_argument("--stop-loss", type=float, default=0)
    parser.add_argument("--stop-delta", type=float, default=0)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--val-check-every", type=int, default=1)
    parser.add_argument(
        "--log-embeddings", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--label-smoothing", type=float, default=0)
    parser.add_argument(
        "--token-weighting", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--onset-loss-weight", type=float, default=1.0)
    parser.add_argument("--structural-loss-lambda", type=float, default=0.0)
    parser.add_argument(
        "--token-class-loss", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--token-class-weight-structural", type=float, default=0.5)
    parser.add_argument("--token-class-weight-mid", type=float, default=1.0)
    parser.add_argument("--token-class-weight-content", type=float, default=2.0)
    parser.add_argument("--token-class-weight-zero", type=float, default=4.0)
    parser.add_argument(
        "--learnable-class-loss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Replace fixed --token-class-weight-* with learnable per-tier "
            "uncertainty weighting (Kendall-Gal multi-task loss form): each "
            "tier carries a learnable log-sigma; gradient descent finds the "
            "tier weighting that minimises overall loss."
        ),
    )
    parser.add_argument(
        "--generalization-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Auto-abort training on pathological-distribution signatures: "
            "content/structural acc ratio, loop-collapse rate, distinct-n4."
        ),
    )
    parser.add_argument(
        "--infonce-content-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Lambda for InfoNCE-style contrastive auxiliary loss on content "
            "tokens (Direction-1 multi-modal objective). 0 = disabled."
        ),
    )
    parser.add_argument("--infonce-distractors", type=int, default=32)
    parser.add_argument(
        "--per-tier-heads",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Replace the single output projection with 4 tier-specific heads "
            "(structural / mid / content / zero) + a 4-way tier router, "
            "marginal-factorisation unified at inference."
        ),
    )
    parser.add_argument(
        "--per-tier-content-mos-k",
        type=int,
        default=4,
        help="Mixture-of-Softmaxes K on the content head when --per-tier-heads is on.",
    )
    parser.add_argument(
        "--per-tier-mos-entropy-lambda",
        type=float,
        default=0.0,
        help="Entropy regularisation weight on MoS mixture gates (anti-mode-collapse).",
    )
    parser.add_argument(
        "--mask-structural-tier-loss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Zero structural-tier positions in the plain-CE loss so the "
            "gradient comes only from mid + content + zero tier targets. "
            "Probes whether router-saturation is downstream of structural "
            "being the easiest-to-predict tier."
        ),
    )
    parser.add_argument(
        "--content-cluster-head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Replace MoSHead with ClusterContentHead in PerTierHeads. "
            "Hierarchical: cluster-first then within-cluster token. "
            "Requires --per-tier-heads and --content-cluster-index; "
            "incompatible with --per-tier-content-mos-k > 0."
        ),
    )
    parser.add_argument(
        "--content-cluster-c",
        type=int,
        default=256,
        help="Number of acoustic clusters for ClusterContentHead.",
    )
    parser.add_argument(
        "--content-cluster-index",
        type=str,
        default="",
        help="Path to cluster_assignments.json (built offline by a "
        "content-clustering pass).",
    )
    parser.add_argument(
        "--content-diffusion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Replace MoSHead with DiffusionContentHead in PerTierHeads. "
            "D3PM absorbing-state discrete diffusion on the content tier; "
            "T-step denoising. Requires --per-tier-heads; mutually exclusive "
            "with --per-tier-content-mos-k > 0 and --content-cluster-head."
        ),
    )
    parser.add_argument(
        "--content-diffusion-t",
        type=int,
        default=8,
        help="D3PM denoising steps for DiffusionContentHead.",
    )
    parser.add_argument(
        "--content-diffusion-d-time",
        type=int,
        default=128,
        help="Sinusoidal time-embedding dimensionality for DiffusionContentHead.",
    )
    parser.add_argument("--gate-audit-prompts", type=int, default=8)
    parser.add_argument("--gate-audit-prompt-len", type=int, default=128)
    parser.add_argument("--gate-generate-n", type=int, default=256)
    parser.add_argument("--gate-generate-every-k", type=int, default=5)
    parser.add_argument(
        "--require-pq", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--tie-word-embeddings", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--predictions", type=int, default=1)
    parser.add_argument("--cents", type=int, default=50)
    parser.add_argument(
        "--constrained-decode", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--max-autotune", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--macro-flags",
        type=str,
        default="",
        help=(
            "Comma/space-separated macro-pass names to enable; each must be in "
            "preframr_tokens.tokenizer_config.MACRO_FLAGS. Dependencies are added "
            "automatically and conflicting passes are rejected. Default: all macro "
            "passes OFF."
        ),
    )
    parser.add_argument(
        "--macro-config",
        type=str,
        default="",
        help=(
            "Named macro preset from tokenizer_config.NAMED_CONFIGS "
            "(baseline|full_macros), merged under --macro-flags."
        ),
    )
    parser.add_argument("--loop-lookahead", type=int, default=3)
    parser.add_argument(
        "--mode-vol-flip-pass", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--voice-id-on-marker",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--voice-order-on-marker",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--freq-onset-interval",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--melody-merge-split", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--coarsen-min-len", type=int, default=16)
    parser.add_argument(
        "--write-blocks", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--project-eval-to-train",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--block-stride", type=int, default=None)
    parser.add_argument(
        "--meta-exclude-digi",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=("Filter dumps via DumpMeta sidecar: skip is_digi=True dumps."),
    )
    parser.add_argument(
        "--meta-irq-lo",
        type=int,
        default=0,
        help=(
            "Filter dumps via DumpMeta sidecar: minimum irq value (inclusive). "
            "0 = no lower bound."
        ),
    )
    parser.add_argument(
        "--meta-irq-hi",
        type=int,
        default=0,
        help=(
            "Filter dumps via DumpMeta sidecar: maximum irq value (inclusive). "
            "0 = no upper bound."
        ),
    )
    parser.add_argument(
        "--meta-require",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Filter dumps via DumpMeta sidecar: drop dumps with missing/stale "
            "metas (default: keep them, let parser regenerate)."
        ),
    )
    return parser


def apply_macro_flags_to_args(args):
    """Resolve ``--macro-flags`` (+ optional ``--macro-config`` preset) into a boolean attr on
    ``args`` for every flag in ``macro_flag_names()``. Validates each requested name, adds
    transitive dependencies, and raises on a conflicting combination (``resolve_flags``). The
    resolved set is written back to ``args.macro_flags`` as a canonical sorted CSV so the
    checkpoint carries the full pipeline and predict can reconstruct it."""
    import re

    from preframr_tokens.macros.flag_registry import macro_flag_names, resolve_flags
    from preframr_tokens.tokenizer_config import NAMED_CONFIGS, named_config

    all_flags = set(macro_flag_names())
    names = [
        tok
        for tok in re.split(r"[,\s]+", (getattr(args, "macro_flags", "") or "").strip())
        if tok
    ]
    config = getattr(args, "macro_config", "") or ""
    requested = set()
    if config:
        if config not in NAMED_CONFIGS:
            raise KeyError(
                f"unknown macro_config {config!r}; known: {sorted(NAMED_CONFIGS)}"
            )
        cfg = named_config(config)
        requested |= {flag for flag in all_flags if getattr(cfg, flag, False)}
    bad = [name for name in names if name not in all_flags]
    if bad:
        raise ValueError(f"unknown macro flag(s) {bad}; valid: {sorted(all_flags)}")
    requested |= set(names)
    resolved = resolve_flags(requested)
    for flag in all_flags:
        setattr(args, flag, flag in resolved)
    args.macro_flags = ",".join(sorted(resolved))
    return resolved
