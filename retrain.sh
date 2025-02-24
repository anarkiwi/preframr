#!/bin/sh
docker rm -f preframr-train
sudo rm -rf /scratch/preframr/tb_logs/*
./build.sh && ./train.sh && docker logs -f preframr-train
