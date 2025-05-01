# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the olegbask/LogicPuzzleBaron dataset to parquet format
"""

import re
import os
import datasets
from collections import defaultdict
import json
from datasets import load_dataset, Dataset
from verl.utils.hdfs_io import copy, makedirs
import argparse
import random
from copy import deepcopy

INSTRUCTIONS = """
# Puzzle to Solve 

{puzzle}

# Instruction

Solve the above puzzle. First thinks about the reasoning process in the mind (enclosed in <think> </think>). Then present your reasoning and solution in the following json format:

{json_template}

Assistant: Let me solve this step by step.\n<think>
""".strip()

def extract_answer(text):
    matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
    return matches[-1].strip() if matches else None


def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data
    
def process(sample):
    data_source = "olegbask/LogicPuzzleBaron"
    prompt = sample["prompt"]
    labels = sample["labels"]
    categories = sample["categories"]
    solution_gt = sample["solution"]

    solution_wo_ans = deepcopy(solution_gt)
    for k, v in solution_wo_ans.items():
        for vk, vv in v.items():
            solution_wo_ans[k][vk] = "___"

    demo = json.dumps({
        "reasoning": "___",
        "solution": solution_wo_ans,
    })
    
    puzzle = prompt[prompt.find("### Puzzle to Solve ") + len("### Puzzle to Solve "):prompt.find("### Instruction")].strip()
    prompt = INSTRUCTIONS.format(puzzle=puzzle, json_template=demo)
    
    new_sample = {
        "data_source": data_source,
        "prompt": [{
            "role": "user",
            "content": prompt,
        }],
        "ability": "Puzzle",
        "reward_model": {
            "style": "rule",
            "labels": labels,
            "categories": categories,
            "ground_truth": json.dumps(solution_gt),
        }
    }
    return new_sample


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', default='annotation/logic_train_easy_deepseek-r1_annotated.jsonl')
    parser.add_argument('--local_dir', default='./data/base/logic_v3')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    data = load_jsonl(args.load_path)
    print(f"len(data) before filtering: {len(data)}")
    valid_data = []
    for sample in data:
        if sample["solution"] and sample["difficulty"] == "easy":
            try:
                n_category, n_house = [int(x) for x in sample["grid_size"].split("x")]
                
                assert len(sample["solution"]) == n_house, f"len(sample['solution']) = {len(sample['solution'])}"
                label_set = set()
                for label_list in sample["labels"]:
                    label_set.update(set(label_list))
                for i in range(1, n_house + 1):
                    assert len(sample["solution"][f"House {i}"]) == n_category, f"len(sample['solution'][f'House {i}']) = {len(sample['solution'][f'House {i}'])}"
                    for k, v in sample["solution"][f"House {i}"].items():
                        assert k in sample["categories"] and v in label_set, f"{k} {v}"

                    cell_cnt = 0
                    for k, v in sample["solution"].items():
                        cell_cnt += len(v)
                    assert cell_cnt == (n_category * n_house), f"{cell_cnt}"

            except Exception as e:
                print(f"Error: {e}")
                continue
            new_sample = {
                "id": sample["id"],
                "prompt": sample["prompt"],
                "labels": sample["labels"][:len(sample['categories'])],
                "categories": sample["categories"],
                "solution": sample["solution"]
            }
            valid_data.append(new_sample)
    data = valid_data
    print(f"len(data) after filtering: {len(data)}")

    random.shuffle(data)
    train_data_raw = data[:int(len(data) * 0.9)]
    test_data_raw = data[int(len(data) * 0.9):]

    train_data = [process(sample) for sample in train_data_raw]
    test_data = [process(sample) for sample in test_data_raw]

    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)
    
    for sample in train_dataset:
        print(sample)
        break
        
    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)