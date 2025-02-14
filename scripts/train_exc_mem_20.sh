#!/bin/bash -e
mkdir -p log/mem_nd_nd/20

CUDA_VISIBLE_DEVICES=2 python -u train.py --DATA_PATH='data_cifar100'  --NUM_LABELS=100  --OUTPUT='saved_cifar100'  --ORDER='mem_nd' --PERCENTAGE=20 --SEED=0 &
PIDS[0]=$!

CUDA_VISIBLE_DEVICES=2 python -u train.py --DATA_PATH='data_cifar100'  --NUM_LABELS=100  --OUTPUT='saved_cifar100'  --ORDER='mem_nd' --PERCENTAGE=20 --SEED=1 &
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=2 python -u train.py --DATA_PATH='data_cifar100'  --NUM_LABELS=100  --OUTPUT='saved_cifar100'  --ORDER='mem_nd' --PERCENTAGE=20 --SEED=2 &
PIDS[2]=$!

CUDA_VISIBLE_DEVICES=2 python -u train.py --DATA_PATH='data_cifar100'  --NUM_LABELS=100  --OUTPUT='saved_cifar100'  --ORDER='mem_nd' --PERCENTAGE=20 --SEED=3 &
PIDS[3]=$!

CUDA_VISIBLE_DEVICES=2 python -u train.py --DATA_PATH='data_cifar100'  --NUM_LABELS=100  --OUTPUT='saved_cifar100'  --ORDER='mem_nd' --PERCENTAGE=20 --SEED=42 &
PIDS[4]=$!

trap "kill ${PIDS[*]}" SIGINT

wait
