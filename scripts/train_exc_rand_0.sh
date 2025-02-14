 #!/bin/bash -e


CUDA_VISIBLE_DEVICES=1 python -u train.py --DATA_PATH='data_cifar100' --NUM_LABELS=100  --OUTPUT='saved_cifar100' --ORDER='random_nd' --SEED=0 --SAVE_CHECKPOINT=True &
PIDS[0]=$!

CUDA_VISIBLE_DEVICES=1 python -u train.py --DATA_PATH='data_cifar100' --NUM_LABELS=100  --OUTPUT='saved_cifar100' --ORDER='random_nd' --SEED=1 --SAVE_CHECKPOINT=True &
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=1 python -u train.py --DATA_PATH='data_cifar100' --NUM_LABELS=100  --OUTPUT='saved_cifar100' --ORDER='random_nd' --SEED=2 --SAVE_CHECKPOINT=True &
PIDS[2]=$!

CUDA_VISIBLE_DEVICES=1 python -u train.py --DATA_PATH='data_cifar100' --NUM_LABELS=100  --OUTPUT='saved_cifar100' --ORDER='random_nd' --SEED=3 --SAVE_CHECKPOINT=True &
PIDS[3]=$!

CUDA_VISIBLE_DEVICES=1 python -u train.py --DATA_PATH='data_cifar100' --NUM_LABELS=100  --OUTPUT='saved_cifar100' --ORDER='random_nd' --SEED=42 --SAVE_CHECKPOINT=True &
PIDS[4]=$!

trap "kill ${PIDS[*]}" SIGINT

wait
