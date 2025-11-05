#!/bin/bash
set -o noglob

IMG=anarkiwi/preframr-jetson
FLAGS=""
NVGPUS=$(nvidia-smi -L 2>/dev/null)
if [[ $? -eq 0 && ! -z "$NVGPUS" ]] ; then
    FLAGS=--runtime=nvidia
fi
exec docker run $FLAGS -eTORCHINDUCTOR_FX_GRAPH_CACHE=1 -eTORCHINDUCTOR_CACHE_DIR=/scratch/preframr/inductorcache --rm --name preframr-predict -v /scratch:/scratch --device /dev/snd -ti $IMG /preframr/predict.py $*
