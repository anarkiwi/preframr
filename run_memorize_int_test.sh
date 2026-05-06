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
LIMITCYCLES=600000000        # ~10 min fallback if Songlengths.md5 lookup misses
MIN_SONG_TOKENS=128          # accept short songs (BlockMapper pads them)
BLOCK_STRIDE=$(expr $SLEN / 4)  # 4x sample density on the tiny test corpus
SONGLENGTHS_DB=${SONGLENGTHS_DB:-${LOCAL_HVSC}/DOCUMENTS/Songlengths.md5}
PAL_HZ=985248
LIMITCYCLES_MARGIN_PCT=10    # capture +10% past the listed duration

# Look up song length in HVSC's Songlengths.md5 and return cycles to
# pass to vsiddump's -limitcycles. Format of the db is:
#   ; /PATH/TO/SONG.sid
#   md5=M:SS[.mmm] [M:SS[.mmm] ...]
# (one duration per subtune; we use the first since vsiddump runs
# -tune 1.) Falls back to ${LIMITCYCLES} if the song isn't listed.
song_length_cycles() {
    local hvsc_path="$1"  # e.g., MUSICIANS/G/Goto80/Truth.sid
    if [ ! -f "${SONGLENGTHS_DB}" ]; then
        echo "${LIMITCYCLES}"
        return
    fi
    local target="; /${hvsc_path}"
    local dur
    # Strip CRLF line endings (Songlengths.md5 ships with \r\n) so the
    # path comparison and the duration parse both work.
    dur=$(awk -v t="${target}" '
        { sub(/\r$/, "") }
        $0 == t { getline; sub(/\r$/, ""); sub(/^[a-f0-9]+=/, ""); print $1; exit }
    ' "${SONGLENGTHS_DB}")
    if [ -z "${dur}" ]; then
        echo "${LIMITCYCLES}"
        return
    fi
    # Parse M:SS or M:SS.mmm into integer seconds (round up, drop ms).
    local minutes seconds
    minutes=${dur%%:*}
    local rest=${dur#*:}
    seconds=${rest%%.*}
    # If there were milliseconds, round seconds up by one to err over.
    if [ "${rest}" != "${seconds}" ]; then
        seconds=$((seconds + 1))
    fi
    local total_sec=$((minutes * 60 + seconds))
    local with_margin=$((total_sec * (100 + LIMITCYCLES_MARGIN_PCT) / 100 + 2))
    echo $((with_margin * PAL_HZ))
}

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

# obtain test SID, extract for the song's actual length (looked up
# from HVSC's Songlengths.md5; falls back to LIMITCYCLES if missing).
# Right-sizing matters because vsiddump faithfully replays the SID
# tune at C64-real-time, and short songs that loop will produce
# duplicated training data far beyond their actual content.
for sid in ${TEST_SIDS} ; do
    bsid=$(basename "${sid}")
    localsid=${LOCAL_HVSC}/${sid}
    outsid="${ROOT}"/"${bsid}"
    if [[ -f "${localsid}" ]] ; then
        cp "${localsid}" "${outsid}"
    else
        wget -O"${outsid}" http://www.hvsc.c64.org/download/C64Music/"${sid}"
    fi
    cycles=$(song_length_cycles "${sid}")
    cycles_sec=$((cycles / PAL_HZ))
    echo "  ${bsid}: -limitcycles ${cycles} (~${cycles_sec}s)"
    docker run --rm ${LIMITS_DUMP} -v ${ROOT}:/scratch/preframr -t anarkiwi/headlessvice /usr/local/bin/vsiddump.py --dumpdir=/scratch/preframr --sid /scratch/preframr/"${bsid}" -tune 1 -limitcycles ${cycles} > "${LOG_DIR}/dump.${bsid}.log" 2>&1 &
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
CARGS="--no-require-pq --seq-len ${SLEN} --tkvocab ${TKVOCAB} --df-map-csv /scratch/preframr/df-map.csv --no-max-autotune --min-song-tokens ${MIN_SONG_TOKENS} --block-stride ${BLOCK_STRIDE} --max-perm 1 --no-fuzzy-loop-pass --no-loop-transposed"
# train to the stop loss. ``-ti`` is dropped because we redirect output
# through ``tee``; the log file is the sole record if the container is
# OOM-killed by the kernel.
docker run ${FLAGS} ${LIMITS_TRAIN} --rm --name preframr-train-test -v ${ROOT}:/scratch/preframr ${IMG} /preframr/train.py ${CARGS} --model=llama3_2 --shuffle 1 --min-dump-size 1 --accumulate-grad-batches 1 --stop-loss ${STOP_LOSS} --stop-delta ${STOP_DELTA} --learning-rate 1e-4 --l1-lambda 0 --weight-decay 0.01 --layers 4 --heads 4 --kv-heads 4 --embed 128 --attn-dropout 0.2 --batch-size 32 --reglogs /scratch/preframr/*.dump.parquet --dataset-csv /scratch/preframr/dataset.csv.zst --token-csv /scratch/preframr/tokens.csv 2>&1 | tee "${LOG_DIR}/train.log"
# predict with min accuracy.
echo docker run ${FLAGS} ${LIMITS_TRAIN} --rm --name preframr-predict-test -v ${ROOT}:/scratch/preframr ${IMG} /preframr/predict.py ${CARGS} --prompt-seq-len ${PLEN} --max-seq-len ${SLEN} --min-acc ${MIN_ACC} --predictions 10
docker run ${FLAGS} ${LIMITS_TRAIN} --rm --name preframr-predict-test -v ${ROOT}:/scratch/preframr ${IMG} /preframr/predict.py ${CARGS} --prompt-seq-len ${PLEN} --max-seq-len ${SLEN} --min-acc ${MIN_ACC} --predictions 10 2>&1 | tee "${LOG_DIR}/predict.log"
