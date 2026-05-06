import argparse
from preframr.model import MODEL_GETTERS, MODEL_PRECISION


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
    parser.add_argument("--min-dump-size", type=int, default=int(1e5))
    # Minimum post-encode token count for a song to enter training. Was
    # implicitly seq_len*2 (16384) via the parser's _filter check, which
    # rejected ~75% of HVSC. BlockMapper pads short blocks, so a small
    # floor (~256 tokens) is enough to skip degenerate / empty dumps
    # while admitting normal-length songs.
    parser.add_argument("--min-song-tokens", type=int, default=256)
    parser.add_argument("--min-irq", type=int, default=int(1.5e4))
    parser.add_argument("--max-irq", type=int, default=int(2.5e4))
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
    parser.add_argument("--start-n", type=int, default=None)
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
        default="float32",
        choices=list(MODEL_PRECISION.keys()),
    )
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--l1-lambda", type=float, default=0)
    parser.add_argument("--accumulate-grad-batches", type=int, default=4)
    parser.add_argument("--log-every-n-steps", type=int, default=2)
    parser.add_argument("--max-perm", type=int, default=3)
    parser.add_argument("--ckpt-hours", type=int, default=12)
    parser.add_argument("--asid", type=str, default=None)
    parser.add_argument("--sysex-delay", type=float, default=0.005)
    parser.add_argument("--min-acc", type=float, default=0)
    parser.add_argument("--stop-loss", type=float, default=0)
    parser.add_argument("--stop-delta", type=float, default=0)
    parser.add_argument("--focal-alpha", type=float, default=1)
    parser.add_argument("--focal-gamma", type=float, default=0)
    parser.add_argument("--label-smoothing", type=float, default=0)
    parser.add_argument(
        "--require-pq", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--tie-word-embeddings", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--predictions", type=int, default=1)
    parser.add_argument("--cents", type=int, default=50)
    parser.add_argument(
        "--max-autotune", action=argparse.BooleanOptionalAction, default=True
    )
    # Macro pass toggles. The other macro passes (PWM, TRANSPOSE, FLIP2,
    # INTERVAL, SubregPass, FilterSweep, EndTerminator) are unconditional;
    # only LoopPass has a kill switch since it's the most aggressive
    # encoder and the one most worth A/B-testing. Default on -- adds
    # ~60% compression vs raw for ~8% parse cost on songs with
    # repeated structure.
    parser.add_argument(
        "--loop-pass", action=argparse.BooleanOptionalAction, default=True
    )
    # Transposed-loop matching inside LoopPass: matches frame patterns
    # whose freq SET vals differ from a prior occurrence by a uniform
    # delta, emitting BACK_REF_TRANSPOSED_OP. Targets defmon-style
    # orderlist transposition (same pattern played at different keys).
    parser.add_argument(
        "--loop-transposed", action=argparse.BooleanOptionalAction, default=True
    )
    # Fuzzy-loop matching inside LoopPass: catches frame patterns
    # whose musical fingerprint matches a prior occurrence even when
    # byte-level writes differ. Encoded as PATTERN_REPLAY_OP + N
    # PATTERN_OVERLAY_OP rows; decoder replays the source body and
    # applies overlays as additional SET writes. Stacks with the
    # exact / transposed / DO_LOOP matchers (single unified pass).
    # Default ON: adds ~8% compression on the /tmp/inv corpus on top
    # of LoopPass-only (varies 4-15% per song; biggest wins on songs
    # with state-drift between repeats like instrument envelope
    # counters or vibrato phase).
    parser.add_argument(
        "--fuzzy-loop-pass", action=argparse.BooleanOptionalAction, default=True
    )
    # Per-(voice, direction) cap on the GateMacroPass palette. Default
    # ``None`` keeps v1 behaviour (unbounded palette: every distinct
    # ``(ctrl, AD, SR)`` end-of-frame state earns a slot). Setting a small
    # integer caps vocab pressure on GATE_REPLAY_OP tokens at the cost of
    # compression on long-tail bundles -- transitions whose bundle would
    # land in slot >= cap stay literal.
    parser.add_argument("--gate-palette-cap", type=int, default=None)
    # InstrumentProgramPass v2 knobs. ``instrument-window`` is the maximum
    # number of frames captured into one program; capture closes earlier on
    # the next gate event for that voice. ``instrument-palette-cap`` bounds
    # the per-stream palette size; over-cap programs stay literal.
    parser.add_argument("--instrument-window", type=int, default=8)
    parser.add_argument("--instrument-palette-cap", type=int, default=None)
    # Per-pass kill switches. Default-on (the production pipeline runs
    # all of these). The integration test disables them to skip the
    # per-block run_passes overhead during make_tokens, since for a
    # memorise-back smoke test the model just needs to predict the
    # literal token sequence -- it doesn't need to learn the macro
    # decoders.
    parser.add_argument(
        "--instrument-pass", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--gate-macro-pass", action=argparse.BooleanOptionalAction, default=True
    )
    # Parse-time generation of self-contained ``.blocks.npy`` files for
    # the BlockMapper training data path. Each block is the output of
    # ``self_contain_slice`` (literal expansion + per-slice re-encode
    # via ``run_passes``), so palette indices are slice-local and
    # decoding never references state defined outside the block. Pass
    # ``--no-write-blocks`` to skip the per-rotation block file and
    # fall back to the SeqMapper sliding-window training path.
    parser.add_argument(
        "--write-blocks", action=argparse.BooleanOptionalAction, default=True
    )
    # Stride between adjacent self-contained blocks, in logical frame
    # slots. Default ``None`` => non-overlapping tiling (stride =
    # frames_per_block ~= seq_len // 2). Smaller values produce
    # overlapping blocks with different musical phases, expanding the
    # training-sample count per song. Each block is independently
    # self-contained, so overlap is correctness-neutral. Useful when
    # the corpus is small relative to model capacity.
    parser.add_argument("--block-stride", type=int, default=None)
    # Predict from BlockMapper instead of SeqMapper. The two streams are
    # encoded differently: SeqMapper holds the full-song parse output
    # (with whole-song-scope macro detection), while BlockMapper holds
    # per-block re-encodes (slice -> ``run_passes`` per block). Even with
    # macro passes disabled, cross-frame patterns the full-song parse
    # captures (TRANSPOSE, PWM, etc.) can vanish when re-encoding a
    # slice. The model trains on BlockMapper, so for memorise-back tests
    # we want predict to read from BlockMapper too -- otherwise the
    # prompt is in a token shape the model never trained on and the
    # accuracy collapses to ~chance.
    parser.add_argument(
        "--predict-from-blocks",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser
