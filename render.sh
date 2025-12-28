#!/bin/bash
set -o noglob

IMG=anarkiwi/preframr
exec docker run --rm --name preframr-render -v /scratch:/scratch --device /dev/snd -ti $IMG /preframr/render.py --no-require-pq --prompt-seq-len 99999 --max-seq-len 999999 --start-n 0 --csv /scratch/preframr/preframr.csv $*
