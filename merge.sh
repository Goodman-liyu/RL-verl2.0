source activate zero
export LD_LIBRARY_PATH=/mnt/petrelfs/linhonglin/anaconda3/envs/zero/lib/python3.10/site-packages/nvidia/nvjitlink/lib/:$LD_LIBRARY_PATH
cd /mnt/petrelfs/linhonglin/rl/RL-verl2.0
for step in {20..20..60}
do
    # python scripts/model_merger.py --local_dir=/mnt/petrelfs/linhonglin/rl/RL-verl2.0/checkpoints/Nips/qwen_ins_deepscaler_bs_1024_1.0/global_step_${step}/actor
    # python scripts/model_merger.py --local_dir=/mnt/petrelfs/linhonglin/rl/RL-verl2.0/checkpoints/Nips/qwen_dpcd_bs_1024/global_step_${step}/actor
    python scripts/model_merger.py --local_dir=/mnt/petrelfs/linhonglin/rl/RL-verl2.0/checkpoints/Nips/qwen_ins_dpcd_bs_1024/global_step_${step}/actor
done