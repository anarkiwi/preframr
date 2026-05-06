#!/bin/bash
#
# DESIGN SKETCH ONLY -- not yet runnable end-to-end. Marks every spot
# where pipeline support is needed before this test can be exercised.
#
# Goal: demonstrate that the model GENERALISES on Goto80's catalog --
# i.e., it produces plausible continuations on songs it has never seen
# during training -- as distinct from the memorize-back smoke test in
# run_memorize_int_test.sh.
#
# Definition of "generalising" used here: token-level next-token
# accuracy on held-out Goto80 songs substantially exceeds chance.
# Concrete bands at vocab~500:
#   chance:   ~0.2%   (1/vocab)
#   memorise: 80-95%  (current memorize-back test)
#   GENERALISE: 20-40%  <- the band this test gates against
# We pick MIN_VAL_ACC = 0.25 as the pass threshold (well above noise,
# below the gap to memorisation).
#
# Pipeline gaps marked PIPELINE-TODO; see "Required pipeline changes"
# section at the bottom for details.

set -e

# ----- Static config ---------------------------------------------------
LOCAL_HVSC=/scratch/hvsc
ROOT=/tmp/preframr_gen
IMG=anarkiwi/preframr

# 16/4 train/eval split, picked deterministically from goto80_breakdown
# output (see notes below). Train spans his career; eval is held back
# from across the same span so the test rewards style learning, not
# tracker-fingerprinting on a single era.
#
# PIPELINE-TODO: replace these placeholders with the chosen split. The
# split is calibrated against goto80_breakdown.py's token-count
# distribution -- pick songs near the median (~11K tokens) for both
# splits so neither is dominated by long-form outliers.
TRAIN_SIDS="
  MUSICIANS/G/Goto80/Truth.sid
  MUSICIANS/G/Goto80/Acid_10000.sid
  MUSICIANS/G/Goto80/CBM_85.sid
  MUSICIANS/G/Goto80/Skybox.sid
  MUSICIANS/G/Goto80/Adventures_of_Pippin_Tom.sid
  MUSICIANS/G/Goto80/Apollo_Launch_2x.sid
  MUSICIANS/G/Goto80/424.sid
  MUSICIANS/G/Goto80/80squares.sid
  MUSICIANS/G/Goto80/20_Years_Is_Nothing.sid
  MUSICIANS/G/Goto80/Aeppelepsi_Gubbjaevel.sid
  MUSICIANS/G/Goto80/Italic_Disco.sid
  MUSICIANS/G/Goto80/Skybox.sid
  MUSICIANS/G/Goto80/Honolulu.sid
  MUSICIANS/G/Goto80/Lollipop.sid
  MUSICIANS/G/Goto80/Ponky.sid
  MUSICIANS/G/Goto80/Superman.sid
"
EVAL_SIDS="
  MUSICIANS/G/Goto80/Robinson.sid
  MUSICIANS/G/Goto80/Afternorm.sid
  MUSICIANS/G/Goto80/Feddamys.sid
  MUSICIANS/G/Goto80/Oj.sid
"

# Generalisation-tuned model: smaller than memorize-back, trained for
# more epochs with early stopping. Embed grows from 128 -> 256 to give
# ~5M params -- enough capacity to learn style on 16 songs without
# memorising verbatim.
SLEN=8192
PLEN=$(expr $SLEN / 2)         # 4096-token prompt: half-context "finish this song"
TKVOCAB=2048                   # Unigram, learned over train+eval (alphabet must cover eval)
LIMITCYCLES=600000000          # fallback; per-song lookup overrides
MIN_SONG_TOKENS=128
BLOCK_STRIDE=$(expr $SLEN / 4)
MIN_VAL_ACC=0.25               # generalisation threshold: 100x random baseline (~0.2%)
EARLY_STOP_PATIENCE=5          # epochs of no val improvement before stopping
EARLY_STOP_MIN_DELTA=0.01      # min val_loss improvement to count
MAX_EPOCHS=200                 # ceiling; early-stop usually fires before
SONGLENGTHS_DB=${SONGLENGTHS_DB:-${LOCAL_HVSC}/DOCUMENTS/Songlengths.md5}
PAL_HZ=985248
LIMITCYCLES_MARGIN_PCT=10

# Container resource caps (same shape as memorize test).
DUMP_MAX_MEM=${DUMP_MAX_MEM:-4g}
TRAIN_MAX_MEM=${TRAIN_MAX_MEM:-16g}     # bigger than memorize: longer training
SHM_SIZE=${SHM_SIZE:-2g}
LIMITS_DUMP="--memory=${DUMP_MAX_MEM} --memory-swap=${DUMP_MAX_MEM}"
LIMITS_TRAIN="--memory=${TRAIN_MAX_MEM} --memory-swap=${TRAIN_MAX_MEM} --shm-size=${SHM_SIZE} --oom-kill-disable=false"

# ----- Stage 1: dump -----
# Both train AND eval SIDs need to be dumped:
#   - train SIDs feed the LM
#   - eval SIDs are needed at predict time AND (PIPELINE-TODO #1) at
#     train time for the validation_step's val_dataloader.

if [[ -d "${ROOT}" ]]; then sudo chown -R $(id -u) "${ROOT}"; rm -rf "${ROOT}"; fi
mkdir -p "${ROOT}"
LOG_DIR="${ROOT}/logs"; mkdir -p "${LOG_DIR}"

# Same song_length_cycles helper as memorize test -- right-sizes the
# dump to actual song length.
song_length_cycles() {
    local hvsc_path="$1"
    if [ ! -f "${SONGLENGTHS_DB}" ]; then echo "${LIMITCYCLES}"; return; fi
    local target="; /${hvsc_path}"
    local dur
    dur=$(awk -v t="${target}" '
        { sub(/\r$/, "") }
        $0 == t { getline; sub(/\r$/, ""); sub(/^[a-f0-9]+=/, ""); print $1; exit }
    ' "${SONGLENGTHS_DB}")
    if [ -z "${dur}" ]; then echo "${LIMITCYCLES}"; return; fi
    local minutes seconds
    minutes=${dur%%:*}
    local rest=${dur#*:}
    seconds=${rest%%.*}
    if [ "${rest}" != "${seconds}" ]; then seconds=$((seconds + 1)); fi
    local total_sec=$((minutes * 60 + seconds))
    local with_margin=$((total_sec * (100 + LIMITCYCLES_MARGIN_PCT) / 100 + 2))
    echo $((with_margin * PAL_HZ))
}

dump_one() {
    local sid="$1" subdir="$2"
    local bsid=$(basename "${sid}")
    local localsid=${LOCAL_HVSC}/${sid}
    local outsid="${ROOT}/${subdir}/${bsid}"
    mkdir -p "${ROOT}/${subdir}"
    if [[ -f "${localsid}" ]]; then cp "${localsid}" "${outsid}"
    else wget -O"${outsid}" http://www.hvsc.c64.org/download/C64Music/"${sid}"
    fi
    local cycles=$(song_length_cycles "${sid}")
    docker run --rm ${LIMITS_DUMP} -v ${ROOT}/${subdir}:/scratch/preframr -t \
        anarkiwi/headlessvice /usr/local/bin/vsiddump.py \
        --dumpdir=/scratch/preframr --sid /scratch/preframr/"${bsid}" \
        -tune 1 -limitcycles ${cycles} \
        > "${LOG_DIR}/dump.${subdir}.${bsid}.log" 2>&1
}

# Dump train + eval into separate subdirs so --reglogs and
# --eval-reglogs (PIPELINE-TODO #1) can target each independently.
for sid in ${TRAIN_SIDS}; do dump_one "${sid}" train & done
for sid in ${EVAL_SIDS}; do dump_one "${sid}" eval & done
wait

ls -l ${ROOT}/train/*.dump.parquet ${ROOT}/eval/*.dump.parquet

# ----- Stage 2: tensorboard + build -----
docker rm -f tensorboard-test || true
docker run -v "${ROOT}/tb_logs":/tb_logs --rm --name tensorboard-test \
    -d -p 6006:6006 -ti anarkiwi/tensorboard

FLAGS=""
NVGPUS=$(nvidia-smi -L 2>/dev/null || true)
if [[ -n "${NVGPUS}" ]]; then FLAGS=--gpus=all; fi
./build.sh

# ----- Stage 3: train with val tracking -----
# Args carry both train (--reglogs) and eval (--eval-reglogs) sets.
# Alphabet is built from the union (PIPELINE-TODO #2). Val loss is
# computed every epoch on eval blocks; EarlyStopping fires when val
# stops improving (PIPELINE-TODO #3).

CARGS="--no-require-pq --seq-len ${SLEN} --tkvocab ${TKVOCAB} \
       --df-map-csv /scratch/preframr/df-map.csv --no-max-autotune \
       --min-song-tokens ${MIN_SONG_TOKENS} --block-stride ${BLOCK_STRIDE} \
       --max-perm 3 --no-fuzzy-loop-pass --no-loop-transposed"

# PIPELINE-TODO #1: --eval-reglogs new arg
# PIPELINE-TODO #2: alphabet must cover eval -- either Unigram on
#                   train+eval (--tkvocab > 0 with alphabet-reglogs
#                   spanning both), or new --alphabet-reglogs arg.
# PIPELINE-TODO #3: --early-stop-patience, --early-stop-min-delta.
# PIPELINE-TODO #4: --val-check-every (epochs).
docker run ${FLAGS} ${LIMITS_TRAIN} --rm --name preframr-train-test \
    -v ${ROOT}:/scratch/preframr ${IMG} \
    /preframr/train.py ${CARGS} \
    --model=llama3_2 \
    --shuffle 1 \
    --min-dump-size 1 \
    --accumulate-grad-batches 2 --batch-size 16 \
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

# ----- Stage 4: assertion ----- (PIPELINE-TODO #5)
# After training, parse the TB scalar log (or a CSV the trainer dumps)
# and verify:
#   - best val_loss < val_loss_random_baseline * 0.7    # learned something
#   - best val_acc >= MIN_VAL_ACC (=0.25)               # generalised
#   - best_val_loss_epoch > 0                           # didn't collapse
#   - best_val_loss_epoch <= 0.8 * MAX_EPOCHS           # early-stop sensibly
#
# Calibration: val_loss_random_baseline must be measured ONCE by
# running this test with random-init weights, then hard-coded.
#
# A separate predict.py invocation per eval song to produce audio
# (predict.py reads the best checkpoint via --model-state) gives a
# qualitative listen for humans:
docker run ${FLAGS} ${LIMITS_TRAIN} --rm --name preframr-predict-test \
    -v ${ROOT}:/scratch/preframr ${IMG} \
    /preframr/predict.py ${CARGS} \
    --prompt-seq-len ${PLEN} --max-seq-len ${SLEN} \
    --min-acc ${MIN_VAL_ACC} --predictions $(echo ${EVAL_SIDS} | wc -w) \
    --reglogs '/scratch/preframr/eval/*.dump.parquet' \
    2>&1 | tee "${LOG_DIR}/predict.log"

# ----- Required pipeline changes (summary) -----
#
# 1. ``--eval-reglogs`` arg in args.py: glob for held-out songs.
#    Default empty (no val).
#
# 2. RegDataset: dual BlockMapper.
#    - ``self.block_mapper`` for train (existing)
#    - ``self.val_block_mapper`` for eval (new)
#    Alphabet (make_tokens) covers union of train + eval -- either by
#    walking eval songs through accumulate_tokens too (simplest), or
#    by training Unigram (--tkvocab > 0) over train+eval.
#    Block .npy files written under .blocks.npy (train) or
#    .val_blocks.npy (eval); load() registers each with the matching
#    mapper.
#
# 3. Model.validation_step: clean cross-entropy on val blocks (no
#    audio-frame weighting, no focal loss). Logs ``val_loss`` and
#    ``val_acc`` (per-token accuracy) every epoch via
#    self.log("val_loss", ...).
#
# 4. Trainer wiring (train.py):
#    - get_loader(args, dataset, mapper="val") returns the val
#      DataLoader; trainer.fit(model, train_dl, val_dl).
#    - Lightning callbacks:
#        EarlyStopping(monitor="val_loss",
#                      patience=args.early_stop_patience,
#                      min_delta=args.early_stop_min_delta,
#                      mode="min")
#        ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")
#    - val_check_interval=args.val_check_every
#
# 5. Test assertion driver: a small Python helper that reads the
#    TensorBoard event file (or trainer-emitted CSV) and asserts the
#    bands above. Returns nonzero exit if generalisation thresholds
#    aren't met.
#
# Estimated wall-time on RTX 4090: ~50-80 min (dump 12 min + build 2
# min + train 30-60 min + predict 5 min). Suitable for nightly CI,
# not per-commit.
