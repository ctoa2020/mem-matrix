#!/bin/bash -e


CUDA_VISIBLE_DEVICES=0 python -u test.py --DATA_PATH='data_cifar100' --OUTPUT='saved_cifar100'   --NUM_LABELS=100 --ORDER='mem' --FOLDER='test_uni_0.1' --PERCENTAGE='0' --SEED='0' --SAVE_CHECKPOINT=True &
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=0 python -u test.py --DATA_PATH='data_cifar100' --OUTPUT='saved_cifar100'   --NUM_LABELS=100 --ORDER='mem' --FOLDER='test_uni_0.1' --PERCENTAGE='0' --SEED='1' --SAVE_CHECKPOINT=True &
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=0 python -u test.py --DATA_PATH='data_cifar100' --OUTPUT='saved_cifar100'   --NUM_LABELS=100 --ORDER='mem' --FOLDER='test_uni_0.1' --PERCENTAGE='0' --SEED='2' --SAVE_CHECKPOINT=True &
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=0 python -u test.py --DATA_PATH='data_cifar100' --OUTPUT='saved_cifar100'   --NUM_LABELS=100 --ORDER='mem' --FOLDER='test_uni_0.1' --PERCENTAGE='0' --SEED='3' --SAVE_CHECKPOINT=True &
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=0 python -u test.py --DATA_PATH='data_cifar100' --OUTPUT='saved_cifar100'   --NUM_LABELS=100 --ORDER='mem' --FOLDER='test_uni_0.1' --PERCENTAGE='0' --SEED='42' --SAVE_CHECKPOINT=True &
PIDS[1]=$!

#0.2
CUDA_VISIBLE_DEVICES=0 python -u test.py --DATA_PATH='data_cifar100' --OUTPUT='saved_cifar100'   --NUM_LABELS=100 --ORDER='mem' --FOLDER='test_uni_0.2' --PERCENTAGE='0' --SEED='0' --SAVE_CHECKPOINT=True &
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=0 python -u test.py --DATA_PATH='data_cifar100' --OUTPUT='saved_cifar100'   --NUM_LABELS=100 --ORDER='mem' --FOLDER='test_uni_0.2' --PERCENTAGE='0' --SEED='1' --SAVE_CHECKPOINT=True &
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=0 python -u test.py --DATA_PATH='data_cifar100' --OUTPUT='saved_cifar100'   --NUM_LABELS=100 --ORDER='mem' --FOLDER='test_uni_0.2' --PERCENTAGE='0' --SEED='2' --SAVE_CHECKPOINT=True &
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=0 python -u test.py --DATA_PATH='data_cifar100' --OUTPUT='saved_cifar100'   --NUM_LABELS=100 --ORDER='mem' --FOLDER='test_uni_0.2' --PERCENTAGE='0' --SEED='3' --SAVE_CHECKPOINT=True &
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=0 python -u test.py --DATA_PATH='data_cifar100' --OUTPUT='saved_cifar100'   --NUM_LABELS=100 --ORDER='mem' --FOLDER='test_uni_0.2' --PERCENTAGE='0' --SEED='42' --SAVE_CHECKPOINT=True &
PIDS[1]=$!

#0.5
CUDA_VISIBLE_DEVICES=0 python -u test.py --DATA_PATH='data_cifar100' --OUTPUT='saved_cifar100'   --NUM_LABELS=100 --ORDER='mem' --FOLDER='test_uni_0.5' --PERCENTAGE='0' --SEED='0' --SAVE_CHECKPOINT=True &
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=0 python -u test.py --DATA_PATH='data_cifar100' --OUTPUT='saved_cifar100'   --NUM_LABELS=100 --ORDER='mem' --FOLDER='test_uni_0.5' --PERCENTAGE='0' --SEED='1' --SAVE_CHECKPOINT=True &
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=0 python -u test.py --DATA_PATH='data_cifar100' --OUTPUT='saved_cifar100'   --NUM_LABELS=100 --ORDER='mem' --FOLDER='test_uni_0.5' --PERCENTAGE='0' --SEED='2' --SAVE_CHECKPOINT=True &
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=0 python -u test.py --DATA_PATH='data_cifar100' --OUTPUT='saved_cifar100'   --NUM_LABELS=100 --ORDER='mem' --FOLDER='test_uni_0.5' --PERCENTAGE='0' --SEED='3' --SAVE_CHECKPOINT=True &
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=0 python -u test.py --DATA_PATH='data_cifar100' --OUTPUT='saved_cifar100'   --NUM_LABELS=100 --ORDER='mem' --FOLDER='test_uni_0.5' --PERCENTAGE='0' --SEED='42' --SAVE_CHECKPOINT=True &
PIDS[1]=$!

#0.7
CUDA_VISIBLE_DEVICES=0 python -u test.py --DATA_PATH='data_cifar100' --OUTPUT='saved_cifar100'   --NUM_LABELS=100 --ORDER='mem' --FOLDER='test_uni_0.7' --PERCENTAGE='0' --SEED='0' --SAVE_CHECKPOINT=True &
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=0 python -u test.py --DATA_PATH='data_cifar100' --OUTPUT='saved_cifar100'   --NUM_LABELS=100 --ORDER='mem' --FOLDER='test_uni_0.7' --PERCENTAGE='0' --SEED='1' --SAVE_CHECKPOINT=True &
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=0 python -u test.py --DATA_PATH='data_cifar100' --OUTPUT='saved_cifar100'   --NUM_LABELS=100 --ORDER='mem' --FOLDER='test_uni_0.7' --PERCENTAGE='0' --SEED='2' --SAVE_CHECKPOINT=True &
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=0 python -u test.py --DATA_PATH='data_cifar100' --OUTPUT='saved_cifar100'   --NUM_LABELS=100 --ORDER='mem' --FOLDER='test_uni_0.7' --PERCENTAGE='0' --SEED='3' --SAVE_CHECKPOINT=True &
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=0 python -u test.py --DATA_PATH='data_cifar100' --OUTPUT='saved_cifar100'   --NUM_LABELS=100 --ORDER='mem' --FOLDER='test_uni_0.7' --PERCENTAGE='0' --SEED='42' --SAVE_CHECKPOINT=True &
PIDS[1]=$!

#1.0
CUDA_VISIBLE_DEVICES=0 python -u test.py --DATA_PATH='data_cifar100' --OUTPUT='saved_cifar100'   --NUM_LABELS=100 --ORDER='mem' --FOLDER='test_uni_1.0' --PERCENTAGE='0' --SEED='0' --SAVE_CHECKPOINT=True &
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=0 python -u test.py --DATA_PATH='data_cifar100' --OUTPUT='saved_cifar100'   --NUM_LABELS=100 --ORDER='mem' --FOLDER='test_uni_1.0' --PERCENTAGE='0' --SEED='1' --SAVE_CHECKPOINT=True &
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=0 python -u test.py --DATA_PATH='data_cifar100' --OUTPUT='saved_cifar100'   --NUM_LABELS=100 --ORDER='mem' --FOLDER='test_uni_1.0' --PERCENTAGE='0' --SEED='2' --SAVE_CHECKPOINT=True &
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=0 python -u test.py --DATA_PATH='data_cifar100' --OUTPUT='saved_cifar100'   --NUM_LABELS=100 --ORDER='mem' --FOLDER='test_uni_1.0' --PERCENTAGE='0' --SEED='3' --SAVE_CHECKPOINT=True &
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=0 python -u test.py --DATA_PATH='data_cifar100' --OUTPUT='saved_cifar100'   --NUM_LABELS=100 --ORDER='mem' --FOLDER='test_uni_1.0' --PERCENTAGE='0' --SEED='42' --SAVE_CHECKPOINT=True &
PIDS[1]=$!



trap "kill ${PIDS[*]}" SIGINT

wait
