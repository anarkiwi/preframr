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

# setup test environment
ROOT=/tmp/preframr
IMG=anarkiwi/preframr
ID=$(id -u)

if [[ -d "${ROOT}" ]] ; then
    sudo chown -R "${ID}" "${ROOT}"
    rm -rf "${ROOT}"
fi

mkdir -p "${ROOT}"

# obtain test SID, extract no more than 60s
for sid in ${TEST_SIDS} ; do
    bsid=$(basename "${sid}")
    localsid=${LOCAL_HVSC}/${sid}
    outsid="${ROOT}"/"${bsid}"
    if [[ -f "${localsid}" ]] ; then
        cp "${localsid}" "${outsid}"
    else
        wget -O"${outsid}" http://www.hvsc.c64.org/download/C64Music/"${sid}"
    fi
    docker run --rm -v ${ROOT}:/scratch/preframr -t anarkiwi/headlessvice /usr/local/bin/vsiddump.py --dumpdir=/scratch/preframr --sid /scratch/preframr/"${bsid}" -tune 1 -limitcycles ${LIMITCYCLES} &
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
# train to the stop loss.
docker run ${FLAGS} --rm --name preframr-train-test -v ${ROOT}:/scratch/preframr -ti ${IMG} /preframr/train.py ${CARGS} --model=llama3_2 --shuffle 1 --min-dump-size 1 --accumulate-grad-batches 1 --stop-loss ${STOP_LOSS} --stop-delta ${STOP_DELTA} --learning-rate 1e-4 --l1-lambda 0 --weight-decay 0.01 --layers 4 --heads 4 --kv-heads 4 --embed 128 --attn-dropout 0.2 --batch-size 32 --reglogs /scratch/preframr/*.dump.parquet --dataset-csv /scratch/preframr/dataset.csv.zst --token-csv /scratch/preframr/tokens.csv
# predict with min accuracy.
echo docker run ${FLAGS} --rm --name preframr-predict-test -v ${ROOT}:/scratch/preframr -ti ${IMG} /preframr/predict.py ${CARGS} --prompt-seq-len ${PLEN} --max-seq-len ${SLEN} --min-acc ${MIN_ACC} --predictions 10
docker run ${FLAGS} --rm --name preframr-predict-test -v ${ROOT}:/scratch/preframr -ti ${IMG} /preframr/predict.py ${CARGS} --prompt-seq-len ${PLEN} --max-seq-len ${SLEN} --min-acc ${MIN_ACC} --predictions 10
