import pandas as pd

# 要读取的 Parquet 文件路径
input_path = "/mnt/petrelfs/linhonglin/rl/RL-verl2.0/data/base/deepscaler/train.parquet"

# 1. 读取整个 Parquet 文件到 DataFrame
df = pd.read_parquet(input_path)

# 2. 随机抽样 10,000 条（设置 random_state 以便结果可复现）
sampled_df = df.sample(n=10_000, random_state=42)

# 3. 将抽样后的结果保存到新的 Parquet 文件（可选）
output_path = "/mnt/petrelfs/linhonglin/rl/RL-verl2.0/data/base/dp10k/train.parquet"
sampled_df.to_parquet(output_path, index=False)

print(f"Completed sampling 10k rows, saved to: {output_path}")
