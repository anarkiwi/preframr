#!/bin/bash
set -o noglob

IMG=anarkiwi/preframr
FLAGS=""
NVGPUS=$(nvidia-smi -L 2>/dev/null)
if [[ $? -eq 0 && ! -z "$NVGPUS" ]] ; then
    FLAGS=--gpus=all
fi
exec docker run $FLAGS -eTORCHINDUCTOR_FX_GRAPH_CACHE=1 -eTORCHINDUCTOR_CACHE_DIR=/scratch/preframr/inductorcache --rm --name preframr-predict -v /scratch:/scratch --device /dev/snd/seq -ti $IMG /preframr/predict.py $*
