#!/bin/bash
# Image matrix:
#   cuda:      anarkiwi/preframr (train + predict; full Dockerfile)
#              anarkiwi/preframr-predict (predict-only slim; Dockerfile.predict)
#   xpu:       anarkiwi/preframr-xpu (predict-only slim; Dockerfile.predict + xpu base)
#   jetson:    anarkiwi/preframr-jetson (predict-only slim; Dockerfile.predict +
#              jetson base + jetson/predict-requirements.txt)
# Training runs on cuda only. xpu + jetson are predict-only (eval experiments,
# generation, serving).
set -e
# Optional local build config (gitignored): sets PIP_OPTS to a PyPI cache
# (proxpi) for fast rebakes. The NFS repo dir is shared across build hosts that
# reach the cache on different subnets, so prefer a per-host .env.<hostname>,
# falling back to .env. Neither present -> PIP_OPTS empty -> upstream PyPI
# (slower, still works). See .env.example.
ENV_FILE=".env.$(hostname -s)"
[ -f "$ENV_FILE" ] || ENV_FILE=".env"
[ -f "$ENV_FILE" ] && . "./$ENV_FILE"
PIP_OPTS="${PIP_OPTS:-}"

docker build --build-arg PIP_OPTS="$PIP_OPTS" -f Dockerfile.tensorboard . -t anarkiwi/tensorboard &
B0=$!

NVGPUS=$(nvidia-smi --query-gpu name --format=csv,noheader 2>/dev/null|head -1|cut -f 1 -d" ")
if [[ ! -z "${NVGPUS}" ]] ; then
    if [[ "${NVGPUS}" == "Orin" ]] ; then
        # BASE flipped from anarkiwi/jetson-pytorch to anarkiwi/jetson-triton
        # so torch.compile's Inductor backend can emit Triton kernels on Jetson.
        docker build --build-arg PIP_OPTS="$PIP_OPTS" \
            --build-arg="BASE=anarkiwi/jetson-triton:v2.12.0" \
            --build-arg="REQ=jetson/predict-requirements.txt" \
            -f Dockerfile.predict . -t anarkiwi/preframr-jetson
        wait $B0
        exit 0
    fi
fi

docker build --build-arg PIP_OPTS="$PIP_OPTS" -f Dockerfile . -t anarkiwi/preframr &
B1=$!
docker build --build-arg PIP_OPTS="$PIP_OPTS" -f Dockerfile.predict . -t anarkiwi/preframr-predict &
B2=$!
docker build --build-arg PIP_OPTS="$PIP_OPTS" \
    --build-arg="BASE=anarkiwi/pytorch-xpu:v2.12.0" \
    -f Dockerfile.predict . -t anarkiwi/preframr-xpu &
B3=$!

wait $B0
wait $B1
wait $B2
wait $B3
