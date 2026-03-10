#!/bin/bash
set -o noglob

IMG=anarkiwi/preframr-jetson
exec docker run --rm --name preframr-render -v /scratch:/scratch --device /dev/snd --device /dev/snd/seq -ti $IMG /preframr/render.py --max-seq-len 9999999 $*
