#!/bin/sh
docker rm -f preframr_tb 2>/dev/null
mkdir -p /scratch/preframr/tb_logs
exec docker run --rm -d --name preframr_tb -p 6006:6006 \
    -v /scratch/preframr/tb_logs:/tb_logs \
    anarkiwi/preframr \
    tensorboard --bind_all --logdir /tb_logs --reload_multifile=true
