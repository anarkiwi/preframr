#!/bin/bash
set -e
# PIP_OPTS="--index-url http://192.168.5.1:5001/index/ --trusted-host 192.168.5.1"

docker build --build-arg PIP_OPTS="$PIP_OPTS" -f Dockerfile.tensorboard . -t anarkiwi/tensorboard &
B0=$!

NVGPUS=$(nvidia-smi --query-gpu name --format=csv,noheader 2>/dev/null|head -1|cut -f 1 -d" ")
if [[ ! -z "${NVGPUS}" ]] ; then
    if [[ "${NVGPUS}" == "Orin" ]] ; then
        docker build --build-arg PIP_OPTS="$PIP_OPTS" -f Dockerfile.jetson . -t anarkiwi/preframr-jetson
        exit 0
    fi
fi

docker build --build-arg PIP_OPTS="$PIP_OPTS" -f Dockerfile . -t anarkiwi/preframr &
B1=$!
docker build --build-arg PIP_OPTS="$PIP_OPTS" --build-arg="BASE=anarkiwi/pytorch-xpu:v2.10.0" -f Dockerfile . -t anarkiwi/preframr-xpu &
B2=$!

wait $B0
wait $B1
wait $B2
