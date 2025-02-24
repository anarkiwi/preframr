#!/bin/sh
set -o noglob
exec docker run --gpus=all -eTORCHINDUCTOR_FX_GRAPH_CACHE=1 -eTORCHINDUCTOR_CACHE_DIR=/scratch/preframr/inductorcache --rm --name preframr-train -v /scratch:/scratch -d -ti anarkiwi/preframr /preframr/train.py $*
