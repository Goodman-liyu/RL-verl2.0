#!/bin/bash

source ~/.bashrc
source activate zero
export LD_LIBRARY_PATH=/mnt/petrelfs/linhonglin/anaconda3/envs/zero/lib/python3.10/site-packages/nvidia/nvjitlink/lib/:$LD_LIBRARY_PATH
cd $(dirname $0)
echo pwd: $(pwd)
cd /mnt/petrelfs/linhonglin/rl/RL-verl2.0
DataName=countdown # 数据集名称:[kk,math,code,puzzle,zebra]

export N_GPUS=8
export BASE_MODEL=Qwen/Qwen2.5-7B-Instruct
export DATA_DIR=data/${DataName}
export ROLLOUT_TP_SIZE=8
export EXPERIMENT_NAME=${DataName}_7b_instruct_without_format
export VLLM_ATTENTION_BACKEND=XFORMERS
export LOG_FILE=log/${DataName}_7b_ins_without_format.log
export PROJECT_NAME=Nips

export SAVE_FREQ=10
export TEST_FREQ=10

pkill sft_lr
ray stop --force
ray start --head
# --port=$RAY_PORT --temp-dir=$TEMP_DIR
sleep 1

bash examples/grpo_trainer/countdown.sh

# for step in {10..200..10}
# do
#     python scripts/model_merger.py --local_dir=/mnt/petrelfs/linhonglin/rl/RL-verl2.0/checkpoints/Nips/${EXPERIMENT_NAME}/global_step_${step}/actor
# done