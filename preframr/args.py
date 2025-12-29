import argparse
from preframr.model import MODEL_GETTERS, MODEL_PRECISION


def add_args(parser):
    parser.add_argument(
        "--reglogs",
        type=str,
        default="/scratch/preframr/training-dumps/**/*dump.zst",
    )
    parser.add_argument(
        "--reglog",
        type=str,
        default="",
    )
    parser.add_argument(
        "--model-state",
        type=str,
        default="",
    )
    parser.add_argument("--tb-logs", type=str, default="/scratch/preframr/tb_logs")
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--max-seq-len", type=int, default=32768)
    parser.add_argument("--prompt-seq-len", type=int, default=2048)
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--tkvocab", type=int, default=0)
    parser.add_argument("--wav", type=str, default="/scratch/preframr/preframr.wav")
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument(
        "--dataset-csv", type=str, default="/scratch/preframr/dataset.csv.zst"
    )
    parser.add_argument("--token-csv", type=str, default="/scratch/preframr/token.csv")
    parser.add_argument("--tkmodel", type=str, default="/scratch/preframr/tkmodel.json")
    parser.add_argument(
        "--df-map-csv",
        type=str,
        default="/scratch/preframr/dataset-map.csv.zst",
    )
    parser.add_argument("--shuffle", type=float, default=0.1)
    parser.add_argument("--max-files", type=int, default=1024)
    parser.add_argument("--min-dump-size", type=int, default=int(1e5))
    parser.add_argument("--min-irq", type=int, default=int(1.5e4))
    parser.add_argument("--max-irq", type=int, default=int(2.5e4))
    parser.add_argument("--diffq", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--kv-heads", type=int, default=12)
    parser.add_argument("--embed", type=int, default=192)
    parser.add_argument("--intermediate", type=int, default=None)
    parser.add_argument("--norm-eps", type=float, default=1e-5)
    parser.add_argument("--rope-base", type=float, default=1e4)
    parser.add_argument("--rope-scale", type=float, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--start-seq", type=int, default=0)
    parser.add_argument("--start-n", type=int, default=None)
    parser.add_argument("--attn-dropout", type=float, default=0)
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
        default="float32",
        choices=list(MODEL_PRECISION.keys()),
    )
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--log-every-n-steps", type=int, default=2)
    parser.add_argument("--max-perm", type=int, default=99)
    parser.add_argument("--ckpt-hours", type=int, default=12)
    parser.add_argument("--asid", type=str, default=None)
    parser.add_argument("--sysex-delay", type=float, default=0.002)
    parser.add_argument("--min-acc", type=float, default=0)
    parser.add_argument("--stop-loss", type=float, default=0)
    parser.add_argument("--focal-alpha", type=float, default=1)
    parser.add_argument("--focal-gamma", type=float, default=0)
    parser.add_argument("--label-smoothing", type=float, default=0)
    parser.add_argument(
        "--require-pq", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--tie-word-embeddings", action=argparse.BooleanOptionalAction, default=False
    )
    return parser
