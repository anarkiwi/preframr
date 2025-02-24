#!/bin/bash
set -o noglob

IMG=anarkiwi/preframr-xpu
FLAGS=""
NVGPUS=$(nvidia-smi -L 2>/dev/null)
if [[ $? -eq 0 && ! -z "$NVGPUS" ]] ; then
    IMG=anarkiwi/preframr
    FLAGS=--gpus=all
else
    if [[ -e /dev/dri/renderD128 ]] ; then
        FLAGS="--device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path --ipc=host"
    fi
fi
exec docker run $FLAGS -eTORCHINDUCTOR_FX_GRAPH_CACHE=1 -eTORCHINDUCTOR_CACHE_DIR=/scratch/preframr/inductorcache --rm --name preframr-predict -v /scratch:/scratch -ti $IMG /preframr/predict.py $*
