#!/bin/bash
set -o noglob

IMG=anarkiwi/preframr-jetson
exec docker run --privileged --rm --name preframr-render -v /scratch:/scratch --device /dev/snd -ti $IMG /preframr/render.py --max-seq-len 9999999 $*
