#!/bin/bash
#
# Memorize-back smoke test: train on 4 Goto80 songs until train_loss
# converges, then verify predict reproduces block 0 of the first
# rotation with accuracy >= MIN_ACC. Disables most macro passes for a
# fast preload (literal-SET regime).
#
# See run_generalize_int_test.sh for the harder generalisation gate
# on held-out songs.

set -e

# memorize-specific overrides MUST be set before sourcing common, because
# common's LIMITS_TRAIN string captures TRAIN_MAX_MEM at source time.
TRAIN_MAX_MEM=${TRAIN_MAX_MEM:-12g}
ROOT=${ROOT:-/tmp/preframr}
LOG_DIR="${ROOT}/logs"

# shellcheck disable=SC1091
source "$(dirname "$0")/int_test_common.sh"

# ----- memorize-specific config -----
TEST_SIDS="
  MUSICIANS/G/Goto80/Truth.sid
  MUSICIANS/G/Goto80/Acid_10000.sid
  MUSICIANS/G/Goto80/CBM_85.sid
  MUSICIANS/G/Goto80/Skybox.sid
"
# train_loss EarlyStopping. With macro passes enabled the encoded
# vocab is ~10K (vs ~3K under literal SETs); same train_loss => lower
# per-token confidence, so memorise-back greedy reconstruction needs
# a tighter target. At 0.005 (~99.5% per-token), 512-token greedy
# reconstruction is ~8% likely to land all-correct -- one divergent
# token then trips the safety-net's GATE_REPLAY/BACK_REF checks.
# 0.001 ⇒ per-token p ≈ 99.9%, 512-token greedy p ≈ 60%.
STOP_LOSS=0.001
STOP_DELTA=0.0001
MAX_EPOCHS=500
MIN_ACC=0.2
SLEN=1024
PLEN=$((SLEN / 2))
TKVOCAB=0
MIN_SONG_TOKENS=128
BLOCK_STRIDE=$((SLEN / 4))     # 4x sample density on this small corpus

# ----- Stage 1: prep + dump -----
prepare_root
for sid in ${TEST_SIDS}; do
    dump_one "${sid}" &
done
wait

ls -l "${ROOT}"/*.dump.parquet

# ----- Stage 2: tensorboard + build -----
start_tensorboard
detect_gpu
./build.sh

# ----- Stage 3: train -----
CARGS="--no-require-pq --seq-len ${SLEN} --tkvocab ${TKVOCAB} \
       --df-map-csv /scratch/preframr/df-map.csv --no-max-autotune \
       --min-song-tokens ${MIN_SONG_TOKENS} --block-stride ${BLOCK_STRIDE} \
       --max-perm 1"

# Memorise dial: capacity (10 layers / embed=384 / ~10M params),
# enough updates (--shuffle 32 -> ~60 steps/epoch on ~120 blocks),
# low regularisation (--attn-dropout 0.0, weight_decay 0.01), higher
# LR (5e-4 -- the small dataset can absorb it).
docker run ${FLAGS} ${LIMITS_TRAIN} --rm --name preframr-train-test \
    -v "${ROOT}":/scratch/preframr ${IMG} \
    /preframr/train.py ${CARGS} \
    --model=llama3_2 \
    --shuffle 32 \
    --min-dump-size 1 \
    --accumulate-grad-batches 2 --batch-size 16 \
    --learning-rate 5e-4 --l1-lambda 0 --weight-decay 0.01 \
    --layers 10 --heads 8 --kv-heads 4 --embed 384 --intermediate 1024 \
    --attn-dropout 0.0 \
    --max-epochs ${MAX_EPOCHS} \
    --stop-loss ${STOP_LOSS} --stop-delta ${STOP_DELTA} \
    --reglogs '/scratch/preframr/*.dump.parquet' \
    --dataset-csv /scratch/preframr/dataset.csv.zst \
    --token-csv /scratch/preframr/tokens.csv \
    2>&1 | tee "${LOG_DIR}/train.log"

# ----- Stage 4: predict -----
# Greedy decoding (top-k=1, near-zero temperature) so a fully-memorised
# model deterministically reproduces the trained tokens.
# --start-block 0 anchors the prompt to the model's first trained
# block of rotation 0; without it the random offset in get_prompt
# almost never hits a block boundary the model trained on.
docker run ${FLAGS} ${LIMITS_TRAIN} --rm --name preframr-predict-test \
    -v "${ROOT}":/scratch/preframr ${IMG} \
    /preframr/predict.py ${CARGS} \
    --prompt-seq-len ${PLEN} --max-seq-len ${SLEN} \
    --min-acc ${MIN_ACC} --predictions 10 \
    --temperature 0.1 --top-k 1 --start-block 0 \
    2>&1 | tee "${LOG_DIR}/predict.log"
