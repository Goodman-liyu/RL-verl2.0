#!/bin/bash

source ~/.bashrc
source activate verl

DataName=zebra_v2 # 数据集名称:[kk,math,code,puzzle,zebra]

export N_GPUS=8
export BASE_MODEL=Qwen/Qwen2.5-7B
export DATA_DIR=data/${DataName}
export ROLLOUT_TP_SIZE=8
export EXPERIMENT_NAME=${DataName}_7b_base_without_format
export VLLM_ATTENTION_BACKEND=XFORMERS
export LOG_FILE=log/${DataName}_7b_base_without_format.log
export PROJECT_NAME=Nips

export SAVE_FREQ=100
export TEST_FREQ=50

pkill sft_lr
ray stop
ray start --head
#--port=$RAY_PORT --temp-dir=$TEMP_DIR
sleep 1

bash examples/grpo_trainer/run_qwen2-7b_zebra.sh

for step in {100..1000..100}
do
    python scripts/model_merger.py --local_dir=checkpoints/Nips/zebra_v2_7b_base_without_format/global_step_${step}/actor
done