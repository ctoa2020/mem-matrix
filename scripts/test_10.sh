#!/bin/bash -e

CUDA_VISIBLE_DEVICES=3 python -u test.py --DATA_PATH='data' --OUTPUT='saved' --ORDER='mem' --FOLDER='test_uni_100' --PERCENTAGE='0' --SEED='0' --SAVE_CHECKPOINT=True &
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=3 python -u test.py --DATA_PATH='data' --OUTPUT='saved' --ORDER='mem' --FOLDER='test_uni_100' --PERCENTAGE='0' --SEED='1' --SAVE_CHECKPOINT=True &
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=3 python -u test.py --DATA_PATH='data' --OUTPUT='saved' --ORDER='mem' --FOLDER='test_uni_100' --PERCENTAGE='0' --SEED='2' --SAVE_CHECKPOINT=True &
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=3 python -u test.py --DATA_PATH='data' --OUTPUT='saved' --ORDER='mem' --FOLDER='test_uni_100' --PERCENTAGE='0' --SEED='3' --SAVE_CHECKPOINT=True &
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=3 python -u test.py --DATA_PATH='data' --OUTPUT='saved' --ORDER='mem' --FOLDER='test_uni_100' --PERCENTAGE='0' --SEED='42' --SAVE_CHECKPOINT=True &
PIDS[1]=$!



CUDA_VISIBLE_DEVICES=3 python -u test.py --DATA_PATH='data' --OUTPUT='saved' --ORDER='mem' --FOLDER='test_uni_100' --PERCENTAGE='10' --SEED='0' --SAVE_CHECKPOINT=True &
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=3 python -u test.py --DATA_PATH='data' --OUTPUT='saved' --ORDER='mem' --FOLDER='test_uni_100' --PERCENTAGE='10' --SEED='1' --SAVE_CHECKPOINT=True &
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=3 python -u test.py --DATA_PATH='data' --OUTPUT='saved' --ORDER='mem' --FOLDER='test_uni_100' --PERCENTAGE='10' --SEED='2' --SAVE_CHECKPOINT=True &
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=3 python -u test.py --DATA_PATH='data' --OUTPUT='saved' --ORDER='mem' --FOLDER='test_uni_100' --PERCENTAGE='10' --SEED='3' --SAVE_CHECKPOINT=True &
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=3 python -u test.py --DATA_PATH='data' --OUTPUT='saved' --ORDER='mem' --FOLDER='test_uni_100' --PERCENTAGE='10' --SEED='42' --SAVE_CHECKPOINT=True &
PIDS[1]=$!


CUDA_VISIBLE_DEVICES=3 python -u test.py --DATA_PATH='data' --OUTPUT='saved' --ORDER='mem' --FOLDER='test_uni_100' --PERCENTAGE='20' --SEED='0' --SAVE_CHECKPOINT=True &
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=3 python -u test.py --DATA_PATH='data' --OUTPUT='saved' --ORDER='mem' --FOLDER='test_uni_100' --PERCENTAGE='20' --SEED='1' --SAVE_CHECKPOINT=True &
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=3 python -u test.py --DATA_PATH='data' --OUTPUT='saved' --ORDER='mem' --FOLDER='test_uni_100' --PERCENTAGE='20' --SEED='2' --SAVE_CHECKPOINT=True &
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=3 python -u test.py --DATA_PATH='data' --OUTPUT='saved' --ORDER='mem' --FOLDER='test_uni_100' --PERCENTAGE='20' --SEED='3' --SAVE_CHECKPOINT=True &
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=3 python -u test.py --DATA_PATH='data' --OUTPUT='saved' --ORDER='mem' --FOLDER='test_uni_100' --PERCENTAGE='20' --SEED='42' --SAVE_CHECKPOINT=True &
PIDS[1]=$!


CUDA_VISIBLE_DEVICES=0 python -u test.py --DATA_PATH='data' --OUTPUT='saved' --ORDER='mem' --FOLDER='test_uni_100' --PERCENTAGE='30' --SEED='0' --SAVE_CHECKPOINT=True &
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=0 python -u test.py --DATA_PATH='data' --OUTPUT='saved' --ORDER='mem' --FOLDER='test_uni_100' --PERCENTAGE='30' --SEED='1' --SAVE_CHECKPOINT=True &
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=0 python -u test.py --DATA_PATH='data' --OUTPUT='saved' --ORDER='mem' --FOLDER='test_uni_100' --PERCENTAGE='30' --SEED='2' --SAVE_CHECKPOINT=True &
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=0 python -u test.py --DATA_PATH='data' --OUTPUT='saved' --ORDER='mem' --FOLDER='test_uni_100' --PERCENTAGE='30' --SEED='3' --SAVE_CHECKPOINT=True &
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=0 python -u test.py --DATA_PATH='data' --OUTPUT='saved' --ORDER='mem' --FOLDER='test_uni_100' --PERCENTAGE='30' --SEED='42' --SAVE_CHECKPOINT=True &
PIDS[1]=$!


CUDA_VISIBLE_DEVICES=0 python -u test.py --DATA_PATH='data' --OUTPUT='saved' --ORDER='mem' --FOLDER='test_uni_100' --PERCENTAGE='40' --SEED='0' --SAVE_CHECKPOINT=True &
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=0 python -u test.py --DATA_PATH='data' --OUTPUT='saved' --ORDER='mem' --FOLDER='test_uni_100' --PERCENTAGE='40' --SEED='1' --SAVE_CHECKPOINT=True &
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=0 python -u test.py --DATA_PATH='data' --OUTPUT='saved' --ORDER='mem' --FOLDER='test_uni_100' --PERCENTAGE='40' --SEED='2' --SAVE_CHECKPOINT=True &
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=0 python -u test.py --DATA_PATH='data' --OUTPUT='saved' --ORDER='mem' --FOLDER='test_uni_100' --PERCENTAGE='40' --SEED='3' --SAVE_CHECKPOINT=True &
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=0 python -u test.py --DATA_PATH='data' --OUTPUT='saved' --ORDER='mem' --FOLDER='test_uni_100' --PERCENTAGE='40' --SEED='42' --SAVE_CHECKPOINT=True &
PIDS[1]=$!


CUDA_VISIBLE_DEVICES=0 python -u test.py --DATA_PATH='data' --OUTPUT='saved' --ORDER='mem' --FOLDER='test_uni_100' --PERCENTAGE='50' --SEED='0' --SAVE_CHECKPOINT=True &
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=0 python -u test.py --DATA_PATH='data' --OUTPUT='saved' --ORDER='mem' --FOLDER='test_uni_100' --PERCENTAGE='50' --SEED='1' --SAVE_CHECKPOINT=True &
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=0 python -u test.py --DATA_PATH='data' --OUTPUT='saved' --ORDER='mem' --FOLDER='test_uni_100' --PERCENTAGE='50' --SEED='2' --SAVE_CHECKPOINT=True &
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=0 python -u test.py --DATA_PATH='data' --OUTPUT='saved' --ORDER='mem' --FOLDER='test_uni_100' --PERCENTAGE='50' --SEED='3' --SAVE_CHECKPOINT=True &
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=0 python -u test.py --DATA_PATH='data' --OUTPUT='saved' --ORDER='mem' --FOLDER='test_uni_100' --PERCENTAGE='50' --SEED='42' --SAVE_CHECKPOINT=True &
PIDS[1]=$!


trap "kill ${PIDS[*]}" SIGINT

wait
