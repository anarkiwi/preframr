#!/bin/bash
set -e
find /scratch/preframr/training-dumps/ -name "*.npy" -delete
find /scratch/preframr/training-dumps/ -name "*.uni.zst" -delete
find /scratch/preframr/training-dumps/ -name "*.[0-9]*.[0-9]*.parquet" -delete

