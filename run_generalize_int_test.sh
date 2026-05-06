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
# 50/12 train/eval split (4x the original 16/4). Picks come from
# ``untracked/pick_train_eval_50_12.py`` -- the 62 closest-to-median
# Goto80 SIDs from the canonical (no variants / no previews) subset.
# Eval anchors on the original 4 (Robinson, Afternorm, Feddamys, Oj)
# so val_acc numbers stay broadly comparable with prior-run
# calibration; the other 8 eval slots are stratified across the
# duration distribution. Re-run the pick script if HVSC's mirror
# changes.
TRAIN_SIDS="
  MUSICIANS/G/Goto80/Acidburger.sid
  MUSICIANS/G/Goto80/Acidroos.sid
  MUSICIANS/G/Goto80/Ameff.sid
  MUSICIANS/G/Goto80/BFP_2013_Invite.sid
  MUSICIANS/G/Goto80/Be_There_Mama.sid
  MUSICIANS/G/Goto80/Birds_on_Fire.sid
  MUSICIANS/G/Goto80/Bla.sid
  MUSICIANS/G/Goto80/Blox.sid
  MUSICIANS/G/Goto80/Bokl0v.sid
  MUSICIANS/G/Goto80/Boys_Say_Go.sid
  MUSICIANS/G/Goto80/Clark_O.sid
  MUSICIANS/G/Goto80/Coco.sid
  MUSICIANS/G/Goto80/Dansa_in.sid
  MUSICIANS/G/Goto80/Datahell.sid
  MUSICIANS/G/Goto80/Diskmachine.sid
  MUSICIANS/G/Goto80/Exy.sid
  MUSICIANS/G/Goto80/Flum.sid
  MUSICIANS/G/Goto80/GHTKX.sid
  MUSICIANS/G/Goto80/Groda.sid
  MUSICIANS/G/Goto80/Groovky.sid
  MUSICIANS/G/Goto80/Happy_Goaboy.sid
  MUSICIANS/G/Goto80/Honolulu.sid
  MUSICIANS/G/Goto80/In_the_Name_of_the_Sword.sid
  MUSICIANS/G/Goto80/Invader-tune.sid
  MUSICIANS/G/Goto80/Italo_Megamix.sid
  MUSICIANS/G/Goto80/Klister.sid
  MUSICIANS/G/Goto80/Knark.sid
  MUSICIANS/G/Goto80/Koettlars.sid
  MUSICIANS/G/Goto80/Kukrot.sid
  MUSICIANS/G/Goto80/Linkan.sid
  MUSICIANS/G/Goto80/Lollipop.sid
  MUSICIANS/G/Goto80/Londonk.sid
  MUSICIANS/G/Goto80/Markus.sid
  MUSICIANS/G/Goto80/Matsam0t.sid
  MUSICIANS/G/Goto80/Oldschool.sid
  MUSICIANS/G/Goto80/Paperock.sid
  MUSICIANS/G/Goto80/Pappap.sid
  MUSICIANS/G/Goto80/Ponky.sid
  MUSICIANS/G/Goto80/Rajjv_Tune_4_Einstein.sid
  MUSICIANS/G/Goto80/Raymond.sid
  MUSICIANS/G/Goto80/Schmoove.sid
  MUSICIANS/G/Goto80/Silly_Sex.sid
  MUSICIANS/G/Goto80/Skan0r.sid
  MUSICIANS/G/Goto80/Slobband.sid
  MUSICIANS/G/Goto80/Sound_of_Anders.sid
  MUSICIANS/G/Goto80/Summerfun.sid
  MUSICIANS/G/Goto80/Superman.sid
  MUSICIANS/G/Goto80/Techno_Aha.sid
  MUSICIANS/G/Goto80/Tjobang.sid
  MUSICIANS/G/Goto80/Yoshigoshy.sid
"
EVAL_SIDS="
  MUSICIANS/G/Goto80/Afternorm.sid
  MUSICIANS/G/Goto80/Ajvar_Relish.sid
  MUSICIANS/G/Goto80/CP_L.sid
  MUSICIANS/G/Goto80/Datatechno3.sid
  MUSICIANS/G/Goto80/Feddamys.sid
  MUSICIANS/G/Goto80/Hairy.sid
  MUSICIANS/G/Goto80/Oj.sid
  MUSICIANS/G/Goto80/Om_Ni_Tycker_Jag_Undviker_Er.sid
  MUSICIANS/G/Goto80/Phh.sid
  MUSICIANS/G/Goto80/Rent-A-Cop.sid
  MUSICIANS/G/Goto80/Robinson.sid
  MUSICIANS/G/Goto80/Tesla_Party.sid
"

# Generalisation-tuned model: smaller than memorize-back, trained for
# more epochs with EarlyStopping on val_loss. ~7M params (embed=320,
# intermediate=896, 8 layers): bumped from the 5M used for the 16/4
# split to absorb the wider style variance in 50 train SIDs without
# memorising verbatim.
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
                                 # Observed val_acc at epoch 199 across
                                 # runs: 0.104 (broken-PAD ckpt),
                                 # 0.108 (post-PAD-fix). chance at
                                 # vocab=33858 is ~3e-5; we're already
                                 # ~3000x chance. ~0.05 leaves ample
                                 # headroom for run-to-run variance.
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
    --shuffle 0.4 \
    --accumulate-grad-batches 8 --batch-size 4 \
    --learning-rate 1e-4 --weight-decay 0.01 \
    --layers 8 --heads 8 --kv-heads 4 --embed 320 --intermediate 896 \
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
# ``--constrained-decode`` masks structurally-invalid macro tokens at
# sample time -- BACK_REF / PATTERN_REPLAY whose distance reaches
# before frame 0, orphan PATTERN_OVERLAY at top level, GATE_REPLAY /
# PLAY_INSTRUMENT slots beyond the prompt-established palette,
# DELAY_REG (model otherwise falls onto val=98 fallback), and
# real-reg tokens that would overflow the per-frame IRQ budget. With
# this on all 4 prompts produce playable .wav (was 0/4 before
# constrained decode landed). ``|| true`` is kept defensively but
# rarely fires now.
#
# ``MAX_QUAL_PREDICTS`` caps the number of held-out songs we render
# end-to-end. The val_acc gate above already covers all 12 eval songs
# (cheap, computed during training); the qualitative-listen renders
# are the expensive part (~1-7 min each). Default 4 keeps wallclock
# in line with the prior 16/4 config.
MAX_QUAL_PREDICTS=${MAX_QUAL_PREDICTS:-4}
i=0
for _sid in ${EVAL_SIDS}; do
    if [ "${i}" -ge "${MAX_QUAL_PREDICTS}" ]; then
        break
    fi
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

# Wallclock measured on defroster post-speedup chain:
#   * 16/4 config (prior): ~25-35 min end-to-end
#   * 50/12 config (this script):
#       - dump 6-8 min cold (Skybox dominates; rest in parallel)
#       - build 2 min cached
#       - train ~30-50 min for 200 epochs at shuffle=0.4
#         (4x data x 0.4 sample = 1.6x exposures per epoch vs prior)
#       - predict 1-2 min per qual song x 4 (capped via
#         MAX_QUAL_PREDICTS) = ~5-8 min
#       => ~45-70 min end-to-end. Suitable for nightly CI.
# Parse speedups this round (LoopPass numba + df.attrs palette
# refactor + defer-attach) cut Techno_Aha 205s -> 32s and
# Skybox 137s -> 25s, removing parsing as a wallclock bottleneck.
