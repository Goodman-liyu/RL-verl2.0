#!/bin/bash
source activate zero
export LD_LIBRARY_PATH=/mnt/petrelfs/linhonglin/anaconda3/envs/zero/lib/python3.10/site-packages/nvidia/nvjitlink/lib/:$LD_LIBRARY_PATH

# RAY_PORT=${1:-6379}  # 默认6379
# TEMP_DIR=${2:-/tmp/ray_$RAY_PORT}  # 为每个Ray实例使用不同的临时目录
cd /mnt/petrelfs/linhonglin/rl/RL-verl2.0
DataName=$1 # 数据集名称:[kk,math,code,puzzle,zebra]
export MODEL=$2

export N_GPUS=8
export BASE_MODEL=Qwen/Qwen2.5-7B
export DATA_DIR=data/base/${DataName}
export ROLLOUT_TP_SIZE=8
export EXPERIMENT_NAME=Qwen2.5-7B-${DataName}
export VLLM_ATTENTION_BACKEND=XFORMERS
export LOG_FILE=log/${DataName}_7b_base.log
export PROJECT_NAME=Demystify
if [[ "$MODEL" == "qwen" ]]; then
    export MODEL_NAME="Qwen/Qwen2.5-7B"
    export PPO_MICRO_BATCH_SIZE=8
    export PPO_MINI_BATCH_SIZE=256
    export TRAIN_BATCH_SIZE=1024
    export SPSIZE=1
    export MAX_TOKEN=18432
elif [[ "$MODEL" == "qwen_ins" ]]; then
    export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
    export PPO_MICRO_BATCH_SIZE=8
    export PPO_MINI_BATCH_SIZE=256
    export TRAIN_BATCH_SIZE=1024
    export SPSIZE=1
    export MAX_TOKEN=18432
elif [[ "$MODEL" == "qwen_math" ]]; then
    export MODEL_NAME="Qwen/Qwen2.5-Math-7B"
    export PPO_MICRO_BATCH_SIZE=8
    export PPO_MINI_BATCH_SIZE=256
    export SPSIZE=1
    export MAX_TOKEN=18432
elif [[ "$MODEL" == "llama" ]]; then
    export MODEL_NAME="/mnt/petrelfs/linhonglin/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/48d6d0fc4e02fb1269b36940650a1b7233035cbb" #"/mnt/hwfile/opendatalab/air/gaoxin/Meta-Llama-3.1-8B"
    # export MODEL_NAME="/mnt/hwfile/opendatalab/air/gaoxin/Meta-Llama-3.1-8B"
    # export BASE_MODEL="/mnt/hwfile/opendatalab/air/gaoxin/Meta-Llama-3.1-8B"
    # export EXTRA_LLM_ARGS="+actor_rollout_ref.rollout.engine_kwargs.swap_space=32"
    # export EXTRA_LLM_ARGS="+actor_rollout_ref.rollout.engine_kwargs.swap_space=32"
    # export EXTRA_LLM_ARGS="+actor_rollout_ref.rollout.enable_chunked_prefill=False"
    export PPO_MICRO_BATCH_SIZE=8
    export PPO_MINI_BATCH_SIZE=256
    export SPSIZE=2
    export MAX_TOKEN=14000
fi
ray stop --force
ray start --head
#--port=$RAY_PORT --temp-dir=$TEMP_DIR
sleep 1
# bash examples/grpo_trainer/math_7b.sh
bash examples/grpo_trainer/run_math_7b.sh
# bash examples/grpo_trainer/run_qwen2-7b.sh