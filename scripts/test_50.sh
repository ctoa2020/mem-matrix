#!/bin/bash -e


CUDA_VISIBLE_DEVICES=0 python -u test.py --ORDER='random' --FOLDER='test_uni_1.0' --PERCENTAGE='10' --SEED='0' --SAVE_CHECKPOINT=True
PIDS[0]=$!

CUDA_VISIBLE_DEVICES=0 python -u test.py --ORDER='random' --FOLDER='test_uni_1.0' --PERCENTAGE='10' --SEED='1' --SAVE_CHECKPOINT=True
PIDS[0]=$!

CUDA_VISIBLE_DEVICES=0 python -u test.py --ORDER='random' --FOLDER='test_uni_1.0' --PERCENTAGE='10' --SEED='2' --SAVE_CHECKPOINT=True
PIDS[0]=$!

CUDA_VISIBLE_DEVICES=0 python -u test.py --ORDER='random' --FOLDER='test_uni_1.0' --PERCENTAGE='10' --SEED='3' --SAVE_CHECKPOINT=True
PIDS[0]=$!

CUDA_VISIBLE_DEVICES=0 python -u test.py --ORDER='random' --FOLDER='test_uni_1.0' --PERCENTAGE='10' --SEED='42' --SAVE_CHECKPOINT=True
PIDS[0]=$!

CUDA_VISIBLE_DEVICES=1 python -u test.py --ORDER='random' --FOLDER='test_uni_1.0' --PERCENTAGE='0' --SEED='0' --SAVE_CHECKPOINT=True
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=1 python -u test.py --ORDER='random' --FOLDER='test_uni_1.0' --PERCENTAGE='0' --SEED='1' --SAVE_CHECKPOINT=True
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=1 python -u test.py --ORDER='random' --FOLDER='test_uni_1.0' --PERCENTAGE='0' --SEED='2' --SAVE_CHECKPOINT=True
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=1 python -u test.py --ORDER='random' --FOLDER='test_uni_1.0' --PERCENTAGE='0' --SEED='3' --SAVE_CHECKPOINT=True
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=1 python -u test.py --ORDER='random' --FOLDER='test_uni_1.0' --PERCENTAGE='0' --SEED='42' --SAVE_CHECKPOINT=True
PIDS[1]=$!


CUDA_VISIBLE_DEVICES=1 python -u test.py --ORDER='random' --FOLDER='test_uni_1.0' --PERCENTAGE='20' --SEED='0' --SAVE_CHECKPOINT=True  &
PIDS[2]=$!

CUDA_VISIBLE_DEVICES=1 python -u test.py --ORDER='random' --FOLDER='test_uni_1.0' --PERCENTAGE='20' --SEED='1' --SAVE_CHECKPOINT=True  &
PIDS[2]=$!

CUDA_VISIBLE_DEVICES=1 python -u test.py --ORDER='random' --FOLDER='test_uni_1.0' --PERCENTAGE='20' --SEED='2' --SAVE_CHECKPOINT=True  &
PIDS[2]=$!

CUDA_VISIBLE_DEVICES=1 python -u test.py --ORDER='random' --FOLDER='test_uni_1.0' --PERCENTAGE='20' --SEED='3' --SAVE_CHECKPOINT=True  &
PIDS[2]=$!

CUDA_VISIBLE_DEVICES=1 python -u test.py --ORDER='random' --FOLDER='test_uni_1.0' --PERCENTAGE='20' --SEED='42' --SAVE_CHECKPOINT=True  &
PIDS[2]=$!


CUDA_VISIBLE_DEVICES=1 python -u test.py --ORDER='random' --FOLDER='test_uni_1.0' --PERCENTAGE='30' --SEED='0' --SAVE_CHECKPOINT=True  &
PIDS[3]=$!

CUDA_VISIBLE_DEVICES=1 python -u test.py --ORDER='random' --FOLDER='test_uni_1.0' --PERCENTAGE='30' --SEED='1' --SAVE_CHECKPOINT=True  &
PIDS[3]=$!

CUDA_VISIBLE_DEVICES=1 python -u test.py --ORDER='random' --FOLDER='test_uni_1.0' --PERCENTAGE='30' --SEED='2' --SAVE_CHECKPOINT=True  &
PIDS[3]=$!

CUDA_VISIBLE_DEVICES=1 python -u test.py --ORDER='random' --FOLDER='test_uni_1.0' --PERCENTAGE='30' --SEED='3' --SAVE_CHECKPOINT=True  &
PIDS[3]=$!

CUDA_VISIBLE_DEVICES=1 python -u test.py --ORDER='random' --FOLDER='test_uni_1.0' --PERCENTAGE='30' --SEED='42' --SAVE_CHECKPOINT=True  &
PIDS[3]=$


CUDA_VISIBLE_DEVICES=2 python -u test.py --ORDER='random' --FOLDER='test_uni_1.0' --PERCENTAGE='40' --SEED='0' --SAVE_CHECKPOINT=True  &
PIDS[4]=$!

CUDA_VISIBLE_DEVICES=2 python -u test.py --ORDER='random' --FOLDER='test_uni_1.0' --PERCENTAGE='40' --SEED='1' --SAVE_CHECKPOINT=True  &
PIDS[4]=$!

CUDA_VISIBLE_DEVICES=2 python -u test.py --ORDER='random' --FOLDER='test_uni_1.0' --PERCENTAGE='40' --SEED='2' --SAVE_CHECKPOINT=True  &
PIDS[4]=$!

CUDA_VISIBLE_DEVICES=2 python -u test.py --ORDER='random' --FOLDER='test_uni_1.0' --PERCENTAGE='40' --SEED='3' --SAVE_CHECKPOINT=True  &
PIDS[4]=$!

CUDA_VISIBLE_DEVICES=2 python -u test.py --ORDER='random' --FOLDER='test_uni_1.0' --PERCENTAGE='40' --SEED='42' --SAVE_CHECKPOINT=True  &
PIDS[4]=$!


CUDA_VISIBLE_DEVICES=3 python -u test.py --ORDER='random' --FOLDER='test_uni_1.0' --PERCENTAGE='50' --SEED='0' --SAVE_CHECKPOINT=True  &
PIDS[5]=$!

CUDA_VISIBLE_DEVICES=3 python -u test.py --ORDER='random' --FOLDER='test_uni_1.0' --PERCENTAGE='50' --SEED='1' --SAVE_CHECKPOINT=True  &
PIDS[5]=$!

CUDA_VISIBLE_DEVICES=3 python -u test.py --ORDER='random' --FOLDER='test_uni_1.0' --PERCENTAGE='50' --SEED='2' --SAVE_CHECKPOINT=True  &
PIDS[5]=$!

CUDA_VISIBLE_DEVICES=3 python -u test.py --ORDER='random' --FOLDER='test_uni_1.0' --PERCENTAGE='50' --SEED='3' --SAVE_CHECKPOINT=True  &
PIDS[5]=$!

CUDA_VISIBLE_DEVICES=3 python -u test.py --ORDER='random' --FOLDER='test_uni_1.0' --PERCENTAGE='50' --SEED='42' --SAVE_CHECKPOINT=True  &
PIDS[5]=$!


trap "kill ${PIDS[*]}" SIGINT

wait
