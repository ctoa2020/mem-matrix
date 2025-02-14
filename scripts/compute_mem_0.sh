#!/bin/bash -e
#mkdir -p log/score_42_1layer_noisy

CUDA_VISIBLE_DEVICES=0 python -u compute_mem.py --CHECKPOINT=42 --BETA=mix_nd0.2 --START=0 --LENGTH=2600  --OUTPUT='saved_cifar100' --DATA_PATH='data_cifar100' --TRAIN_DEV_DATA='cifar100_mix_nd0.csv' --NUM_LABELS=100 &
PIDS[0]=$!
CUDA_VISIBLE_DEVICES=1 python -u compute_mem.py --CHECKPOINT=42 --BETA=mix_nd0.2 --START=2600 --LENGTH=2600 --OUTPUT='saved_cifar100' --DATA_PATH='data_cifar100' --TRAIN_DEV_DATA='cifar100_mix_nd0.csv' --NUM_LABELS=100 &
PIDS[1]=$!





trap "kill ${PIDS[*]}" SIGINT

wait
