#!/bin/bash
set -o noglob

IMG=anarkiwi/preframr
exec docker run --rm --name preframr-tokenize -v /scratch:/scratch -ti $IMG /preframr/stftokenize.py $*
