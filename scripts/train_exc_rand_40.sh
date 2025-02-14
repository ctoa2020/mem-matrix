#!/bin/bash -e


CUDA_VISIBLE_DEVICES=0 python -u train.py --DATA_PATH='data_cifar100'  --NUM_LABELS=100  --OUTPUT='saved_cifar100' --ORDER='random_nd' --PERCENTAGE=40 --SEED=0 &
PIDS[0]=$!

CUDA_VISIBLE_DEVICES=0 python -u train.py --DATA_PATH='data_cifar100'  --NUM_LABELS=100  --OUTPUT='saved_cifar100' --ORDER='random_nd' --PERCENTAGE=40 --SEED=1 &
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=0 python -u train.py --DATA_PATH='data_cifar100'  --NUM_LABELS=100  --OUTPUT='saved_cifar100' --ORDER='random_nd' --PERCENTAGE=40 --SEED=2 &
PIDS[2]=$!

CUDA_VISIBLE_DEVICES=1 python -u train.py --DATA_PATH='data_cifar100'  --NUM_LABELS=100  --OUTPUT='saved_cifar100' --ORDER='random_nd' --PERCENTAGE=40 --SEED=3 &
PIDS[3]=$!

CUDA_VISIBLE_DEVICES=1 python -u train.py --DATA_PATH='data_cifar100'  --NUM_LABELS=100  --OUTPUT='saved_cifar100' --ORDER='random_nd' --PERCENTAGE=40 --SEED=42 &
PIDS[4]=$!

trap "kill ${PIDS[*]}" SIGINT

wait
