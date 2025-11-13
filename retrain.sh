#!/bin/sh
docker rm -f preframr-train
while true ; do
  sudo rm -rf /scratch/preframr/tb_logs/*
  if [ $? -eq 0 ]; then break ; fi
  sleep 1
done
./build.sh && ./train.sh $* && docker logs -f preframr-train
