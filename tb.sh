#!/bin/sh
docker run -v /scratch/preframr/tb_logs:/tb_logs --rm --name tensorboard -d -p 6006:6006 -ti anarkiwi/tensorboard
