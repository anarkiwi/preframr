#!/bin/bash
set -o noglob

IMG=anarkiwi/preframr-xpu
FLAGS=""
exec docker run $FLAGS --rm --name preframr-render -v /scratch:/scratch --device /dev/snd -ti $IMG /preframr/render.py $*
