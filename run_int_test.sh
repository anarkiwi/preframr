#!/bin/bash

set -e

# Verify we can train to a reasonable loss, and predict to a reasonable accuracy.
# Trains over entire songs (LIMITCYCLES = ~10 min @ ~1 MHz so the dumper
# captures the full song instead of a 3-min head). With the
# short-song-aware training pipeline (--min-song-tokens, BlockMapper
# pad masking), even short subtunes contribute samples.
LOCAL_HVSC=/scratch/hvsc
TEST_SIDS="MUSICIANS/G/Goto80/Truth.sid MUSICIANS/G/Goto80/Acid_10000.sid MUSICIANS/G/Goto80/CBM_85.sid MUSICIANS/G/Goto80/Skybox.sid"
STOP_LOSS=0.02
STOP_DELTA=0.1
MIN_ACC=0.2
SLEN=1024
PLEN=$(expr $SLEN / 2)
TKVOCAB=0
LIMITCYCLES=600000000        # ~10 min capture per song (was 180M = 3 min)
MIN_SONG_TOKENS=128          # accept short songs (BlockMapper pads them)
BLOCK_STRIDE=$(expr $SLEN / 4)  # 4x sample density on the tiny test corpus

# Container resource limits. The previous run hung the host; capping
# host RAM + disabling swap inside containers ensures the OOM-killer
# fires inside the container instead of taking the whole machine
# down. Tune via env: `MAX_MEM=8g ./run_int_test.sh` etc.
# - DUMP_MAX_MEM caps each parallel vsiddump container.
# - TRAIN_MAX_MEM caps the train + predict containers.
# - SHM_SIZE bumps /dev/shm so PyTorch DataLoader workers don't OOM
#   on the default 64MB shared-memory cap.
# Setting --memory-swap equal to --memory disables the container's
# access to host swap; the container OOM-kills cleanly instead of
# pushing the host into thrashing.
DUMP_MAX_MEM=${DUMP_MAX_MEM:-4g}
TRAIN_MAX_MEM=${TRAIN_MAX_MEM:-12g}
SHM_SIZE=${SHM_SIZE:-2g}
LIMITS_DUMP="--memory=${DUMP_MAX_MEM} --memory-swap=${DUMP_MAX_MEM}"
LIMITS_TRAIN="--memory=${TRAIN_MAX_MEM} --memory-swap=${TRAIN_MAX_MEM} --shm-size=${SHM_SIZE} --oom-kill-disable=false"

# setup test environment
ROOT=/tmp/preframr
IMG=anarkiwi/preframr
ID=$(id -u)

if [[ -d "${ROOT}" ]] ; then
    sudo chown -R "${ID}" "${ROOT}"
    rm -rf "${ROOT}"
fi

mkdir -p "${ROOT}"

# Capture container output so a crash leaves diagnostics on disk -- the
# host hard-rebooted last run, which lost everything in tty scrollback.
# Each docker run streams stdout+stderr to a per-stage log; ``tee``
# keeps output flowing to the user's terminal too.
LOG_DIR="${ROOT}/logs"
mkdir -p "${LOG_DIR}"
echo "logs in ${LOG_DIR}"

# obtain test SID, extract up to LIMITCYCLES (full song)
for sid in ${TEST_SIDS} ; do
    bsid=$(basename "${sid}")
    localsid=${LOCAL_HVSC}/${sid}
    outsid="${ROOT}"/"${bsid}"
    if [[ -f "${localsid}" ]] ; then
        cp "${localsid}" "${outsid}"
    else
        wget -O"${outsid}" http://www.hvsc.c64.org/download/C64Music/"${sid}"
    fi
    docker run --rm ${LIMITS_DUMP} -v ${ROOT}:/scratch/preframr -t anarkiwi/headlessvice /usr/local/bin/vsiddump.py --dumpdir=/scratch/preframr --sid /scratch/preframr/"${bsid}" -tune 1 -limitcycles ${LIMITCYCLES} > "${LOG_DIR}/dump.${bsid}.log" 2>&1 &
done
wait

ls -l ${ROOT}/*.dump.parquet

docker rm -f tensorboard-test || true
docker run -v "${ROOT}"/tb_logs:/tb_logs --rm --name tensorboard-test -d -p 6006:6006 -ti anarkiwi/tensorboard

# GPU if any
FLAGS=""
NVGPUS=$(nvidia-smi -L 2>/dev/null || true)
if [[ -n "${NVGPUS}" ]] ; then
    FLAGS=--gpus=all
fi

./build.sh
CARGS="--no-require-pq --seq-len ${SLEN} --tkvocab ${TKVOCAB} --df-map-csv /scratch/preframr/df-map.csv --no-max-autotune --min-song-tokens ${MIN_SONG_TOKENS} --block-stride ${BLOCK_STRIDE}"
# train to the stop loss. ``-ti`` is dropped because we redirect output
# through ``tee``; the log file is the sole record if the container is
# OOM-killed by the kernel.
docker run ${FLAGS} ${LIMITS_TRAIN} --rm --name preframr-train-test -v ${ROOT}:/scratch/preframr ${IMG} /preframr/train.py ${CARGS} --model=llama3_2 --shuffle 1 --min-dump-size 1 --accumulate-grad-batches 1 --stop-loss ${STOP_LOSS} --stop-delta ${STOP_DELTA} --learning-rate 1e-4 --l1-lambda 0 --weight-decay 0.01 --layers 4 --heads 4 --kv-heads 4 --embed 128 --attn-dropout 0.2 --batch-size 32 --reglogs /scratch/preframr/*.dump.parquet --dataset-csv /scratch/preframr/dataset.csv.zst --token-csv /scratch/preframr/tokens.csv 2>&1 | tee "${LOG_DIR}/train.log"
# predict with min accuracy.
echo docker run ${FLAGS} ${LIMITS_TRAIN} --rm --name preframr-predict-test -v ${ROOT}:/scratch/preframr ${IMG} /preframr/predict.py ${CARGS} --prompt-seq-len ${PLEN} --max-seq-len ${SLEN} --min-acc ${MIN_ACC} --predictions 10
docker run ${FLAGS} ${LIMITS_TRAIN} --rm --name preframr-predict-test -v ${ROOT}:/scratch/preframr ${IMG} /preframr/predict.py ${CARGS} --prompt-seq-len ${PLEN} --max-seq-len ${SLEN} --min-acc ${MIN_ACC} --predictions 10 2>&1 | tee "${LOG_DIR}/predict.log"
