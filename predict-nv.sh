#!/bin/bash
set -o noglob

IMG=anarkiwi/preframr
FLAGS=""
NVGPUS=$(nvidia-smi --query-gpu name --format=csv,noheader 2>/dev/null|head -1|cut -f 1 -d" ")
if [[ ! -z "${NVGPUS}" ]] ; then
    FLAGS=--gpus=all
    if [[ "${NVGPUS}" == "Orin" ]] ; then
        IMG=anarkiwi/preframr-jetson
        FLAGS=--runtime=nvidia
    fi
fi
exec docker run ${FLAGS} -eTORCHINDUCTOR_FX_GRAPH_CACHE=1 -eTORCHINDUCTOR_CACHE_DIR=/scratch/preframr/inductorcache --rm --name preframr-predict -v /scratch:/scratch --device /dev/snd -ti ${IMG} /preframr/predict.py $*
