import argparse

from preframr.train.model import MODEL_GETTERS, MODEL_PRECISION


def add_args(parser):
    parser.add_argument(
        "manifest_arg",
        type=str,
        default="",
        nargs="?",
        help="predict: a single manifest path (overrides --manifest).",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="",
        help="train manifest: lines of 'relpath<TAB>subtune' (.sid HVSC-relative + "
        "1-based subtune), resolved under --sid-root and recovered sid-only via the "
        "BACC codec (undecodable tunes skipped).",
    )
    parser.add_argument(
        "--eval-manifest",
        type=str,
        default="",
        help="named eval subsets 'name=path;name=path' (each becomes a val subset).",
    )
    parser.add_argument(
        "--sid-root",
        type=str,
        default="/scratch/preframr/hvsc/C64Music",
        help="root the manifest relpaths resolve against.",
    )
    parser.add_argument(
        "--songlengths",
        type=str,
        default="/scratch/preframr/hvsc/C64Music/DOCUMENTS/Songlengths.md5",
        help="HVSC Songlengths.md5 for per-subtune frame budgets.",
    )
    parser.add_argument(
        "--model-state",
        type=str,
        default="",
    )
    parser.add_argument("--tb-logs", type=str, default="/scratch/preframr/tb_logs")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--prompt-seq-len", type=int, default=2048)
    parser.add_argument("--block-stride", type=int, default=None)
    parser.add_argument("--max-files", type=int, default=2048)
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
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
        help="Write the prediction window of the rendered df as parquet "
        "(register-level rows). For automated melody-quality scoring "
        "downstream of predict.",
    )
    parser.add_argument("--shuffle", type=float, default=0.05)
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
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--decode-penalty-window", type=int, default=128)
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
        "--tie-word-embeddings", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--predictions", type=int, default=1)
    parser.add_argument("--cents", type=int, default=50)
    parser.add_argument(
        "--max-autotune", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True
    )
    return parser
