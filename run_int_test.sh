#!/bin/bash

set -e

# Verify we can train to a reasonable loss, and predict to a reasonable accuracy.
TEST_SID="http://www.hvsc.c64.org/download/C64Music/MUSICIANS/G/Goto80/Truth.sid"
STOP_LOSS=0.025
MIN_ACC=0.2
SLEN=512

# setup test environment
ROOT=/tmp/preframr
IMG=anarkiwi/preframr
ID=$(id -u)

if [[ -d "${ROOT}" ]] ; then
    sudo chown -R ${ID} ${ROOT}
    rm -rf ${ROOT}
fi

mkdir -p ${ROOT}

# obtain test SID, extract no more than 60s
wget -O${ROOT}/test.sid ${TEST_SID}
docker run --rm -v ${ROOT}:/scratch/preframr -t anarkiwi/headlessvice /usr/local/bin/vsiddump.py --dumpdir=/scratch/preframr --sid /scratch/preframr/test.sid -tune 1 -limitcycles 60000000

ls -l ${ROOT}/test.None.dump.parquet

# GPU if any
FLAGS=""
NVGPUS=$(nvidia-smi -L 2>/dev/null || true)
if [[ ! -z "${NVGPUS}" ]] ; then
    FLAGS=--gpus=all
fi

./build.sh
CARGS="--no-require-pq --layers 4 --heads 4 --kv-heads 4 --embed 128 --batch-size 64 --seq-len ${SLEN} --tkvocab 0"
# train to the stop loss.
docker run ${FLAGS} --rm --name preframr-train-test -v ${ROOT}:/scratch/preframr -ti ${IMG} /preframr/train.py ${CARGS} --model=llama3_2 --shuffle 1 --min-dump-size 1 --accumulate-grad-batches 1 --stop-loss ${STOP_LOSS} --reglogs /scratch/preframr/test.None.dump.parquet --dataset-csv /scratch/preframr/dataset.csv.zst --token-csv /scratch/preframr/tokens.csv --df-map-csv /scratch/preframr/df-map.csv
# predict with min accuracy.
docker run ${FLAGS} --rm --name preframr-predict-test -v ${ROOT}:/scratch/preframr -ti ${IMG} /preframr/predict.py ${CARGS} --prompt-seq-len 256 --max-seq-len ${SLEN} --start-n 0 --min-acc ${MIN_ACC} /scratch/preframr/test.None.dump.parquet
