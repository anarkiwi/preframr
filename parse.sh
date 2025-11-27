#!/bin/bash
set -o noglob

IMG=anarkiwi/preframr
exec docker run --rm --name preframr-parse -v /scratch:/scratch -ti $IMG /preframr/parse.py --min-dump-size 0 $*
