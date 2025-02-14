#!/bin/bash -e
mkdir -p log/score_0.7


CUDA_VISIBLE_DEVICES=0 python -u compute_mem.py --CHECKPOINT=42 --BETA=mix_nd_0 --START=0 --LENGTH=2500 --NUM_LABELS=10 --OUTPUT='saved' --DATA_PATH='data' --TRAIN_DEV_DATA='train_10000.csv' &
PIDS[0]=$!
CUDA_VISIBLE_DEVICES=1 python -u compute_mem.py --CHECKPOINT=42 --BETA=mix_nd_0 --START=2500 --LENGTH=2500 --NUM_LABELS=10 --OUTPUT='saved' --DATA_PATH='data' --TRAIN_DEV_DATA='train_10000.csv'&
PIDS[1]=$!
CUDA_VISIBLE_DEVICES=2 python -u compute_mem.py --CHECKPOINT=42 --BETA=mix_nd_0 --START=5000 --LENGTH=2500 --NUM_LABELS=10 --OUTPUT='saved' --DATA_PATH='data' --TRAIN_DEV_DATA='train_10000.csv'&
PIDS[2]=$!
CUDA_VISIBLE_DEVICES=3 python -u compute_mem.py --CHECKPOINT=42 --BETA=mix_nd_0 --START=7500 --LENGTH=2500 --NUM_LABELS=10 --OUTPUT='saved' --DATA_PATH='data' --TRAIN_DEV_DATA='train_10000.csv'&
PIDS[3]=$!


trap "kill ${PIDS[*]}" SIGINT

wait
