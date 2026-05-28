import argparse

from preframr.train.model import MODEL_GETTERS, MODEL_PRECISION
from preframr_tokens.macros.flag_registry import macro_flag_names


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
        "--loop-pass", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--loop-transposed", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--fuzzy-loop-pass", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--fuzzy-fp-adsr", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--loop-lookahead", type=int, default=3)
    parser.add_argument(
        "--mode-vol-flip-pass", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--hard-restart-pass", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--ctrl-bigram-pass",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--legato-pass-c2", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--legato-pass-c4", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--legato-pass-c7", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--coarsen-pass", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--voice-canonical-block-order",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--voice-trajectory-pass",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--voice-trajectory-window",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--voice-trajectory-distributed-pass",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--set-to-diff-pass",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--freq-trajectory-pass", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--trajectory-anchor-pass", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--freq-v0-interval", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--preset-pass", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--voice-track-pass", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--freq-nudge-pass", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--release-update-pass", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--ctrl-triple-pass", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--strict-lonely", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--lonely-catch-all", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--motif-pass", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--motif-dict",
        type=str,
        default="",
        help="Path to a motif_dict.json (mined by preframr/mine_motifs.py); "
        "loaded lazily when --motif-pass is set.",
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
        "--pipeline-spec",
        type=str,
        default="",
        help=(
            "Parse pipeline as JSON-serialised list of transforms. Accepts an "
            "inline JSON string or '@/path/to/spec.json'. Empty = use legacy "
            "boolean flags."
        ),
    )
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


def load_pipeline_spec(pipeline_spec_arg):
    if not pipeline_spec_arg:
        return None
    import json

    raw = pipeline_spec_arg
    if raw.startswith("@"):
        with open(raw[1:]) as f:
            raw = f.read()
    return json.loads(raw)


_PIPELINE_NAME_TO_FLAG = {
    "hard_restart": ("hard_restart_pass", True),
    "ctrl_bigram": ("ctrl_bigram_pass", True),
    "voice_block_order": ("voice_canonical_block_order", True),
    "voice_trajectory": ("voice_trajectory_pass", True),
    "voice_trajectory_distributed": ("voice_trajectory_distributed_pass", True),
    "set_to_diff": ("set_to_diff_pass", True),
    "freq_trajectory": ("freq_trajectory_pass", True),
    "preset": ("preset_pass", True),
    "loop": ("loop_pass", True),
    "coarsen": ("coarsen_pass", True),
    "fuzzy_loop": ("fuzzy_loop_pass", True),
}

_LEGATO_CLUSTERS = tuple(
    sorted(
        int(flag[len("legato_pass_c") :])
        for flag in macro_flag_names()
        if flag.startswith("legato_pass_c")
    )
)


def apply_pipeline_spec_to_args(args):
    """Translate a pipeline_spec arg into the legacy boolean flag attrs. Resolves @path references and stashes the JSON content back into args.pipeline_spec so the checkpoint always carries the resolved spec."""
    import json

    spec = load_pipeline_spec(getattr(args, "pipeline_spec", ""))
    if spec is None:
        return None
    args.pipeline_spec = json.dumps(spec, separators=(",", ":"))
    for cluster in _LEGATO_CLUSTERS:
        setattr(args, f"legato_pass_c{cluster}", False)
    for flag_pair in _PIPELINE_NAME_TO_FLAG.values():
        attr, _ = flag_pair
        setattr(args, attr, False)
    entries = spec.get("transforms", []) if isinstance(spec, dict) else spec
    for entry in entries:
        name = entry["name"] if isinstance(entry, dict) else entry
        params = entry.get("params", {}) if isinstance(entry, dict) else {}
        if name == "legato_per_cluster":
            for cluster in params.get("clusters", []):
                setattr(args, f"legato_pass_c{int(cluster)}", True)
            continue
        if name in _PIPELINE_NAME_TO_FLAG:
            attr, value = _PIPELINE_NAME_TO_FLAG[name]
            setattr(args, attr, value)
        if name == "voice_trajectory":
            window = params.get("window")
            if window is not None:
                setattr(args, "voice_trajectory_window", int(window))
    return spec
