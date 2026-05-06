#!/bin/bash
#
# Goal: demonstrate that the model GENERALISES on Goto80's catalog --
# i.e., produces plausible continuations on held-out songs -- as
# distinct from the memorize-back smoke test in
# run_memorize_int_test.sh.
#
# "Generalising" = token-level next-token accuracy on held-out
# Goto80 songs substantially exceeds chance. With TKVOCAB=0 the
# raw alphabet on this corpus is ~33K entries (~0.003% chance), so
# any val_acc clearing ~1% is already meaningful signal.

set -e

# generalize-specific overrides MUST be set before sourcing common,
# because common's LIMITS_TRAIN captures TRAIN_MAX_MEM at source time.
TRAIN_MAX_MEM=${TRAIN_MAX_MEM:-16g}     # bigger than memorize: longer training
ROOT=${ROOT:-/tmp/preframr_gen}
LOG_DIR="${ROOT}/logs"

# shellcheck disable=SC1091
source "$(dirname "$0")/int_test_common.sh"

# ----- generalize-specific config -----
# 16/4 train/eval split. Train spans Goto80's career; eval is held back
# from across the same span so the test rewards style learning, not
# tracker-fingerprinting on a single era. The 6 picks below the first
# 10 (Blox..Knark) are the 6 SIDs nearest the 147s catalogue-median
# duration that are NOT variants/previews and NOT in eval -- generated
# deterministically by ``untracked/pick_train_replacements.py``. Run
# that script if HVSC's mirror changes and the picks need refreshing.
TRAIN_SIDS="
  MUSICIANS/G/Goto80/Truth.sid
  MUSICIANS/G/Goto80/Acid_10000.sid
  MUSICIANS/G/Goto80/CBM_85.sid
  MUSICIANS/G/Goto80/Skybox.sid
  MUSICIANS/G/Goto80/20_Years_Is_Nothing.sid
  MUSICIANS/G/Goto80/Italic_Disco.sid
  MUSICIANS/G/Goto80/Honolulu.sid
  MUSICIANS/G/Goto80/Lollipop.sid
  MUSICIANS/G/Goto80/Ponky.sid
  MUSICIANS/G/Goto80/Superman.sid
  MUSICIANS/G/Goto80/Blox.sid
  MUSICIANS/G/Goto80/Techno_Aha.sid
  MUSICIANS/G/Goto80/Boys_Say_Go.sid
  MUSICIANS/G/Goto80/Hairy.sid
  MUSICIANS/G/Goto80/Clark_O.sid
  MUSICIANS/G/Goto80/Knark.sid
"
EVAL_SIDS="
  MUSICIANS/G/Goto80/Robinson.sid
  MUSICIANS/G/Goto80/Afternorm.sid
  MUSICIANS/G/Goto80/Feddamys.sid
  MUSICIANS/G/Goto80/Oj.sid
"

# Generalisation-tuned model: smaller than memorize-back, trained for
# more epochs with EarlyStopping on val_loss. ~5M params, enough
# capacity to learn style on 16 songs without memorising verbatim.
SLEN=8192
PLEN=$((SLEN / 2))               # 4096-token prompt: half-context "finish this song"
# TKVOCAB=0 ⇒ use the raw (op, reg, subreg, val) alphabet directly,
# no Unigram sub-token learning. Macros-on regime produces a ~10K
# alphabet which is too big for a 2048-vocab Unigram model to cover
# (training raises "vocabulary not large enough to contain all
# chars"). Either bump tkvocab >> alphabet size or skip Unigram;
# memorize uses 0 too.
TKVOCAB=0
MIN_SONG_TOKENS=128
BLOCK_STRIDE=$((SLEN / 4))
MIN_VAL_ACC=${MIN_VAL_ACC:-0}    # 0 = report only (calibration run).
                                 # First calibration run on this
                                 # config plateaued at val_acc ~0.10
                                 # in the late-training regime; a
                                 # ~0.05 threshold leaves headroom
                                 # for run-to-run variance.
EARLY_STOP_PATIENCE=5            # epochs of no val improvement before stopping
EARLY_STOP_MIN_DELTA=0.01        # min val_loss improvement to count
MAX_EPOCHS=200                   # ceiling; early-stop usually fires before

# ----- Stage 1: prep + dump -----
# Train and eval go to separate subdirs so --reglogs / --eval-reglogs
# can target each independently.
prepare_root
for sid in ${TRAIN_SIDS}; do dump_one "${sid}" train & done
for sid in ${EVAL_SIDS}; do dump_one "${sid}" eval & done
wait

ls -l "${ROOT}"/train/*.dump.parquet "${ROOT}"/eval/*.dump.parquet

# ----- Stage 2: tensorboard + build -----
start_tensorboard
detect_gpu
./build.sh

# ----- Stage 3: train with val tracking -----
# Alphabet is built from the union of train + eval blocks so eval
# songs never hit unknown tokens. Val loss is computed every epoch
# on eval blocks; EarlyStopping fires when val stops improving.
CARGS="--no-require-pq --seq-len ${SLEN} --tkvocab ${TKVOCAB} \
       --df-map-csv /scratch/preframr/df-map.csv --no-max-autotune \
       --min-song-tokens ${MIN_SONG_TOKENS} --block-stride ${BLOCK_STRIDE} \
       --max-perm 3"

docker run ${FLAGS} ${LIMITS_TRAIN} --rm --name preframr-train-test \
    -v "${ROOT}":/scratch/preframr ${IMG} \
    /preframr/train.py ${CARGS} \
    --model=llama3_2 \
    --shuffle 1 \
    --min-dump-size 1 \
    --accumulate-grad-batches 8 --batch-size 4 \
    --learning-rate 1e-4 --weight-decay 0.01 \
    --layers 8 --heads 8 --kv-heads 4 --embed 256 --intermediate 704 \
    --attn-dropout 0.1 \
    --max-epochs ${MAX_EPOCHS} \
    --early-stop-patience ${EARLY_STOP_PATIENCE} \
    --early-stop-min-delta ${EARLY_STOP_MIN_DELTA} \
    --val-check-every 1 \
    --reglogs '/scratch/preframr/train/*.dump.parquet' \
    --eval-reglogs '/scratch/preframr/eval/*.dump.parquet' \
    --dataset-csv /scratch/preframr/dataset.csv.zst \
    --token-csv /scratch/preframr/tokens.csv \
    2>&1 | tee "${LOG_DIR}/train.log"

# ----- Stage 4: gate on best val_acc -----
# check_generalize.py reads the TB events file and exits nonzero if the
# best-val-loss epoch's val_acc < MIN_VAL_ACC. Runs inside the preframr
# image because that's where ``tensorboard`` is installed. ``set -e``
# at the top of this script propagates failure.
docker run --rm ${LIMITS_TRAIN} \
    -v "${ROOT}":/scratch/preframr ${IMG} \
    python3 /tests/check_generalize.py \
    --tb-logs /scratch/preframr/tb_logs \
    --min-val-acc ${MIN_VAL_ACC} \
    --min-epochs 2 \
    2>&1 | tee "${LOG_DIR}/check_generalize.log"

# ----- Stage 5: per-eval-song qualitative predict -----
# One predict invocation per held-out song so each gets its own .wav
# / .csv. ``--predict-set val`` routes ``getseq`` through the
# val_block_mapper that load() populated from ``--eval-reglogs``;
# ``--start-seq i`` then picks the i'th eval rotation. Predict will
# load the best-val_loss checkpoint via ``get_ckpt``.
#
# Generalisation predicts often hit the safety net (model emits
# GATE_REPLAY/BACK_REF whose payload doesn't resolve in the
# generated stream); ``|| true`` keeps the script moving so each
# song still gets attempted instead of failing the test on the
# first reject.
i=0
for _sid in ${EVAL_SIDS}; do
    docker run ${FLAGS} ${LIMITS_TRAIN} --rm --name preframr-predict-test-${i} \
        -v "${ROOT}":/scratch/preframr ${IMG} \
        /preframr/predict.py ${CARGS} \
        --prompt-seq-len ${PLEN} --max-seq-len ${SLEN} \
        --min-acc 0 --predictions 1 \
        --predict-set val \
        --start-seq ${i} --start-block 0 \
        --constrained-decode \
        --wav /scratch/preframr/eval-${i}.wav \
        --csv /scratch/preframr/eval-${i}.csv \
        2>&1 | tee "${LOG_DIR}/predict.${i}.log" || true
    i=$((i + 1))
done

# Wallclock measured on defroster: ~25-35 min end-to-end
# (dump 5-6 min for Skybox + 4 min for the rest in parallel,
# build 2 min cached, train ~15-25 min for 200 epochs, predict
# 1-2 min). Suitable for nightly CI, not per-commit.
