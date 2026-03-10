#!/bin/bash
set -o noglob

IMG=anarkiwi/preframr
exec docker run --rm --name preframr-tokenize -v /scratch:/scratch -ti $IMG /preframr/stftokenize.py --df-map-csv /scratch/preframr/testing/dataset-map.csv --token-csv /scratch/preframr/testing/token.csv --tkmodel /scratch/preframr/testing/tkmodel.json $*
