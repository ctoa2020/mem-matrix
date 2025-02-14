#!/bin/bash -e

#0.1
CUDA_VISIBLE_DEVICES=3 python -u test.py --ORDER='random_nd' --FOLDER='test_nd_0.1' --PERCENTAGE='30' --SEED='0' --SAVE_CHECKPOINT=True
PIDS[3]=$!

CUDA_VISIBLE_DEVICES=3 python -u test.py --ORDER='random_nd' --FOLDER='test_nd_0.1' --PERCENTAGE='30' --SEED='1' --SAVE_CHECKPOINT=True
PIDS[3]=$!

CUDA_VISIBLE_DEVICES=3 python -u test.py --ORDER='random_nd' --FOLDER='test_nd_0.1' --PERCENTAGE='30' --SEED='2' --SAVE_CHECKPOINT=True
PIDS[3]=$!

CUDA_VISIBLE_DEVICES=3 python -u test.py --ORDER='random_nd' --FOLDER='test_nd_0.1' --PERCENTAGE='30' --SEED='3' --SAVE_CHECKPOINT=True
PIDS[3]=$!

CUDA_VISIBLE_DEVICES=3 python -u test.py --ORDER='random_nd' --FOLDER='test_nd_0.1' --PERCENTAGE='30' --SEED='42' --SAVE_CHECKPOINT=True
PIDS[3]=$!

#0.2
CUDA_VISIBLE_DEVICES=3 python -u test.py --ORDER='random_nd' --FOLDER='test_nd_0.2' --PERCENTAGE='30' --SEED='0' --SAVE_CHECKPOINT=True
PIDS[3]=$!

CUDA_VISIBLE_DEVICES=3 python -u test.py --ORDER='random_nd' --FOLDER='test_nd_0.2' --PERCENTAGE='30' --SEED='1' --SAVE_CHECKPOINT=True
PIDS[3]=$!

CUDA_VISIBLE_DEVICES=3 python -u test.py --ORDER='random_nd' --FOLDER='test_nd_0.2' --PERCENTAGE='30' --SEED='2' --SAVE_CHECKPOINT=True
PIDS[3]=$!

CUDA_VISIBLE_DEVICES=3 python -u test.py --ORDER='random_nd' --FOLDER='test_nd_0.2' --PERCENTAGE='30' --SEED='3' --SAVE_CHECKPOINT=True
PIDS[3]=$!

CUDA_VISIBLE_DEVICES=3 python -u test.py --ORDER='random_nd' --FOLDER='test_nd_0.2' --PERCENTAGE='30' --SEED='42' --SAVE_CHECKPOINT=True
PIDS[3]=$!


#0.5
CUDA_VISIBLE_DEVICES=3 python -u test.py --ORDER='random_nd' --FOLDER='test_nd_0.5' --PERCENTAGE='30' --SEED='0' --SAVE_CHECKPOINT=True
PIDS[3]=$!

CUDA_VISIBLE_DEVICES=3 python -u test.py --ORDER='random_nd' --FOLDER='test_nd_0.5' --PERCENTAGE='30' --SEED='1' --SAVE_CHECKPOINT=True
PIDS[3]=$!

CUDA_VISIBLE_DEVICES=3 python -u test.py --ORDER='random_nd' --FOLDER='test_nd_0.5' --PERCENTAGE='30' --SEED='2' --SAVE_CHECKPOINT=True
PIDS[3]=$!

CUDA_VISIBLE_DEVICES=3 python -u test.py --ORDER='random_nd' --FOLDER='test_nd_0.5' --PERCENTAGE='30' --SEED='3' --SAVE_CHECKPOINT=True
PIDS[3]=$!

CUDA_VISIBLE_DEVICES=3 python -u test.py --ORDER='random_nd' --FOLDER='test_nd_0.5' --PERCENTAGE='30' --SEED='42' --SAVE_CHECKPOINT=True
PIDS[3]=$!

#0.7
CUDA_VISIBLE_DEVICES=3 python -u test.py --ORDER='random_nd' --FOLDER='test_nd_0.7' --PERCENTAGE='30' --SEED='0' --SAVE_CHECKPOINT=True
PIDS[3]=$!

CUDA_VISIBLE_DEVICES=3 python -u test.py --ORDER='random_nd' --FOLDER='test_nd_0.7' --PERCENTAGE='30' --SEED='1' --SAVE_CHECKPOINT=True
PIDS[3]=$!

CUDA_VISIBLE_DEVICES=3 python -u test.py --ORDER='random_nd' --FOLDER='test_nd_0.7' --PERCENTAGE='30' --SEED='2' --SAVE_CHECKPOINT=True
PIDS[3]=$!

CUDA_VISIBLE_DEVICES=3 python -u test.py --ORDER='random_nd' --FOLDER='test_nd_0.7' --PERCENTAGE='30' --SEED='3' --SAVE_CHECKPOINT=True
PIDS[3]=$!

CUDA_VISIBLE_DEVICES=3 python -u test.py --ORDER='random_nd' --FOLDER='test_nd_0.7' --PERCENTAGE='30' --SEED='42' --SAVE_CHECKPOINT=True
PIDS[3]=$!

#1.0
CUDA_VISIBLE_DEVICES=3 python -u test.py --ORDER='random_nd' --FOLDER='test_nd_1.0' --PERCENTAGE='30' --SEED='0' --SAVE_CHECKPOINT=True
PIDS[3]=$!

CUDA_VISIBLE_DEVICES=3 python -u test.py --ORDER='random_nd' --FOLDER='test_nd_1.0' --PERCENTAGE='30' --SEED='1' --SAVE_CHECKPOINT=True
PIDS[3]=$!

CUDA_VISIBLE_DEVICES=3 python -u test.py --ORDER='random_nd' --FOLDER='test_nd_1.0' --PERCENTAGE='30' --SEED='2' --SAVE_CHECKPOINT=True
PIDS[3]=$!

CUDA_VISIBLE_DEVICES=3 python -u test.py --ORDER='random_nd' --FOLDER='test_nd_1.0' --PERCENTAGE='30' --SEED='3' --SAVE_CHECKPOINT=True
PIDS[3]=$!

CUDA_VISIBLE_DEVICES=3 python -u test.py --ORDER='random_nd' --FOLDER='test_nd_1.0' --PERCENTAGE='30' --SEED='42' --SAVE_CHECKPOINT=True
PIDS[3]=$!




trap "kill ${PIDS[*]}" SIGINT

wait
