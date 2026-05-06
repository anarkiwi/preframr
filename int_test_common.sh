#!/bin/bash
# Shared helpers for run_*_int_test.sh.
#
# Source this file from a sibling test script:
#     source "$(dirname "$0")/int_test_common.sh"
#
# All tunables are exposed as env-overridable variables so individual
# tests can override before sourcing (memorize uses TRAIN_MAX_MEM=12g
# while generalize wants 16g, etc.). The functions below assume the
# caller has set ROOT and LOG_DIR before invoking them.

# ----- Static defaults -----
LOCAL_HVSC=${LOCAL_HVSC:-/scratch/hvsc}
IMG=${IMG:-anarkiwi/preframr}
LIMITCYCLES=${LIMITCYCLES:-600000000}      # ~10 min @ ~1MHz fallback
PAL_HZ=${PAL_HZ:-985248}
LIMITCYCLES_MARGIN_PCT=${LIMITCYCLES_MARGIN_PCT:-10}
SONGLENGTHS_DB=${SONGLENGTHS_DB:-${LOCAL_HVSC}/DOCUMENTS/Songlengths.md5}

# Container resource limits. Capping host RAM + disabling container
# swap forces the OOM-killer to fire inside the container instead of
# taking the whole machine down (an earlier integration run hung the
# host this way).
DUMP_MAX_MEM=${DUMP_MAX_MEM:-4g}
TRAIN_MAX_MEM=${TRAIN_MAX_MEM:-12g}
SHM_SIZE=${SHM_SIZE:-2g}                   # PyTorch DataLoader workers
LIMITS_DUMP="--memory=${DUMP_MAX_MEM} --memory-swap=${DUMP_MAX_MEM}"
LIMITS_TRAIN="--memory=${TRAIN_MAX_MEM} --memory-swap=${TRAIN_MAX_MEM} --shm-size=${SHM_SIZE} --oom-kill-disable=false"

# ----- Helpers -----

# Look up a SID's listed runtime in HVSC's Songlengths.md5 and emit
# the cycle count to feed vsiddump's -limitcycles. The DB format:
#     ; /PATH/TO/SONG.sid
#     md5=M:SS[.mmm] [M:SS[.mmm] ...]
# The first duration is for tune 1 (which vsiddump always uses here).
# Falls back to ${LIMITCYCLES} if the song isn't listed.
song_length_cycles() {
    local hvsc_path="$1"
    if [ ! -f "${SONGLENGTHS_DB}" ]; then
        echo "${LIMITCYCLES}"
        return
    fi
    local target="; /${hvsc_path}"
    local dur
    # Strip CRLF (Songlengths.md5 ships with \r\n) before comparing.
    dur=$(awk -v t="${target}" '
        { sub(/\r$/, "") }
        $0 == t { getline; sub(/\r$/, ""); sub(/^[a-f0-9]+=/, ""); print $1; exit }
    ' "${SONGLENGTHS_DB}")
    if [ -z "${dur}" ]; then
        echo "${LIMITCYCLES}"
        return
    fi
    local minutes seconds
    minutes=${dur%%:*}
    local rest=${dur#*:}
    seconds=${rest%%.*}
    # If a sub-second component was present, round up by 1s (err over).
    if [ "${rest}" != "${seconds}" ]; then
        seconds=$((seconds + 1))
    fi
    local total_sec=$((minutes * 60 + seconds))
    local with_margin=$((total_sec * (100 + LIMITCYCLES_MARGIN_PCT) / 100 + 2))
    echo $((with_margin * PAL_HZ))
}

# Reset and create ROOT + LOG_DIR. Caller sets ROOT and LOG_DIR.
prepare_root() {
    if [[ -d "${ROOT}" ]]; then
        sudo chown -R "$(id -u)" "${ROOT}"
        rm -rf "${ROOT}"
    fi
    mkdir -p "${ROOT}" "${LOG_DIR}"
    echo "logs in ${LOG_DIR}"
}

# Dump one SID via vsiddump. Args:
#   $1 = HVSC-relative path (e.g. MUSICIANS/G/Goto80/Truth.sid)
#   $2 = subdir under ROOT to dump to (empty = ROOT itself)
# The dump container streams into ${ROOT}/${subdir}; the vsiddump
# container always sees /scratch/preframr as that subdir, so the
# generated *.dump.parquet lands next to the source SID.
dump_one() {
    local sid="$1" subdir="${2:-}"
    local bsid
    bsid=$(basename "${sid}")
    local localsid="${LOCAL_HVSC}/${sid}"
    local dest_dir="${ROOT}"
    local log_label="${bsid}"
    if [ -n "${subdir}" ]; then
        dest_dir="${ROOT}/${subdir}"
        log_label="${subdir}.${bsid}"
        mkdir -p "${dest_dir}"
    fi
    local outsid="${dest_dir}/${bsid}"
    if [[ -f "${localsid}" ]]; then
        cp "${localsid}" "${outsid}"
    else
        wget -O"${outsid}" "http://www.hvsc.c64.org/download/C64Music/${sid}"
    fi
    local cycles
    cycles=$(song_length_cycles "${sid}")
    local cycles_sec=$((cycles / PAL_HZ))
    echo "  ${bsid}: -limitcycles ${cycles} (~${cycles_sec}s)"
    docker run --rm ${LIMITS_DUMP} -v "${dest_dir}":/scratch/preframr -t \
        anarkiwi/headlessvice /usr/local/bin/vsiddump.py \
        --dumpdir=/scratch/preframr --sid /scratch/preframr/"${bsid}" \
        -tune 1 -limitcycles "${cycles}" \
        > "${LOG_DIR}/dump.${log_label}.log" 2>&1
}

# Start the tensorboard sidecar container with mounted tb_logs.
start_tensorboard() {
    docker rm -f tensorboard-test || true
    docker run -v "${ROOT}/tb_logs":/tb_logs --rm --name tensorboard-test \
        -d -p 6006:6006 -ti anarkiwi/tensorboard
}

# Set ``FLAGS`` to "--gpus=all" if any nvidia GPU is visible, else "".
detect_gpu() {
    FLAGS=""
    local nvgpus
    nvgpus=$(nvidia-smi -L 2>/dev/null || true)
    if [[ -n "${nvgpus}" ]]; then
        FLAGS="--gpus=all"
    fi
}
