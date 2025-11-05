#!/bin/bash
set -o noglob

IMG=anarkiwi/preframr-xpu
FLAGS=""
if [[ -e /dev/dri/renderD128 ]] ; then
    FLAGS="--device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path --ipc=host"
fi
exec docker run $FLAGS -eTORCHINDUCTOR_FX_GRAPH_CACHE=1 -eTORCHINDUCTOR_CACHE_DIR=/scratch/preframr/inductorcache --rm --name preframr-predict -v /scratch:/scratch --device /dev/snd -ti $IMG /preframr/predict.py $*
