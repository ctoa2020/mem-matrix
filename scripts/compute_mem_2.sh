#!/bin/bash -e
mkdir -p log/score_1.0


CUDA_VISIBLE_DEVICES=2 python -u compute_mem.py --CHECKPOINT=42 --BETA=nd0.1 --START=0 --LENGTH=2500 --NUM_LABELS=10 --OUTPUT='saved_stl10' --DATA_PATH='data_stl10' --TRAIN_DEV_DATA='stl10_mix.csv' &
PIDS[2]=$!
CUDA_VISIBLE_DEVICES=3 python -u compute_mem.py --CHECKPOINT=42 --BETA=nd0.1 --START=2500 --LENGTH=2500 --NUM_LABELS=10 --OUTPUT='saved_stl10' --DATA_PATH='data_stl10' --TRAIN_DEV_DATA='stl10_mix.csv'&
PIDS[3]=$!



trap "kill ${PIDS[*]}" SIGINT

wait
