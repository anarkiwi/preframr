#!/bin/bash

set -e

# Verify we can train to a reasonable loss, and predict to a reasonable accuracy.
LOCAL_HVSC=/scratch/hvsc
TEST_SIDS="MUSICIANS/G/Goto80/Truth.sid MUSICIANS/G/Goto80/Acid_10000.sid MUSICIANS/G/Goto80/CBM_85.sid MUSICIANS/G/Goto80/Skybox.sid"
STOP_LOSS=0.01
STOP_DELTA=0.01
MIN_ACC=0.2
SLEN=4096
PLEN=$(expr $SLEN / 2)
TKVOCAB=0

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
    docker run --rm -v ${ROOT}:/scratch/preframr -t anarkiwi/headlessvice /usr/local/bin/vsiddump.py --dumpdir=/scratch/preframr --sid /scratch/preframr/"${bsid}" -tune 1 -limitcycles 180000000 &
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
CARGS="--no-require-pq --seq-len ${SLEN} --tkvocab ${TKVOCAB} --df-map-csv /scratch/preframr/df-map.csv --no-max-autotune"
# train to the stop loss.
docker run ${FLAGS} --rm --name preframr-train-test -v ${ROOT}:/scratch/preframr -ti ${IMG} /preframr/train.py ${CARGS} --model=llama3_2 --shuffle 1 --min-dump-size 1 --accumulate-grad-batches 1 --stop-loss ${STOP_LOSS} --stop-delta ${STOP_DELTA} --learning-rate 1e-4 --weight-decay 0.01 --layers 4 --heads 4 --kv-heads 4 --embed 256 --batch-size 32 --reglogs /scratch/preframr/*.dump.parquet --dataset-csv /scratch/preframr/dataset.csv.zst --token-csv /scratch/preframr/tokens.csv
# predict with min accuracy.
echo docker run ${FLAGS} --rm --name preframr-predict-test -v ${ROOT}:/scratch/preframr -ti ${IMG} /preframr/predict.py ${CARGS} --prompt-seq-len ${PLEN} --max-seq-len ${SLEN} --min-acc ${MIN_ACC} --predictions 10
docker run ${FLAGS} --rm --name preframr-predict-test -v ${ROOT}:/scratch/preframr -ti ${IMG} /preframr/predict.py ${CARGS} --prompt-seq-len ${PLEN} --max-seq-len ${SLEN} --min-acc ${MIN_ACC} --predictions 10
