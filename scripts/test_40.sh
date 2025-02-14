#!/bin/bash -e
CUDA_VISIBLE_DEVICES=1 python -u test.py --ORDER='random' --FOLDER='test_uni_0.2' --PERCENTAGE='0' --SEED='0' --SAVE_CHECKPOINT=True  &
PIDS[0]=$!
trap "kill ${PIDS[*]}" SIGINT

wait
