#!/bin/bash -e

CUDA_VISIBLE_DEVICES=2 python -u compute_mem.py --CHECKPOINT=42 --BETA=mix_nd0.5 --START=0 --LENGTH=2600 --OUTPUT='saved_cifar100' --DATA_PATH='data_cifar100' --TRAIN_DEV_DATA='cifar100_mix_nd0.csv' --NUM_LABELS=100 &
PIDS[2]=$!
CUDA_VISIBLE_DEVICES=3 python -u compute_mem.py --CHECKPOINT=42 --BETA=mix_nd0.5 --START=2600 --LENGTH=2600 --OUTPUT='saved_cifar100' --DATA_PATH='data_cifar100' --TRAIN_DEV_DATA='cifar100_mix_nd0.csv' --NUM_LABELS=100 &
PIDS[3]=$!



trap "kill ${PIDS[*]}" SIGINT

wait
