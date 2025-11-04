#!/bin/sh
docker rm -f preframr-train
sudo rm -rf /scratch/preframr/tb_logs/*
#./build.sh &&
./train.sh $* --seq-len 8192 --max-seq-len 16384 --batch-size 16 --reglogs '/scratch/preframr/training-dumps/MUSICIANS/G/Goto80/*zst' --shuffle 0.1 --layers 8 --heads 8 --kv-heads 8 --embed 256 && docker logs -f preframr-train
