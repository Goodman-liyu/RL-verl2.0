#!/bin/bash
source activate zero

DataName=$1 # 数据集名称:[kk,math,code,puzzle,zebra]


export N_GPUS=8
export BASE_MODEL=Qwen/Qwen2.5-7B
export DATA_DIR=data/${DataName}
export ROLLOUT_TP_SIZE=8
export EXPERIMENT_NAME=${DataName}_7b_base_without_format
export VLLM_ATTENTION_BACKEND=XFORMERS
export LOG_FILE=log/${DataName}_7b_base_without_format.log
export PROJECT_NAME=Nips


bash examples/grpo_trainer/run_qwen2-7b.sh