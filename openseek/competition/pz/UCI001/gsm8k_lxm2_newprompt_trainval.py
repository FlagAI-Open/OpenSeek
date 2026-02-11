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
Preprocess the GSM8k dataset to parquet format
"""

import argparse
import os
import re

import datasets
from glob import glob

# from verl.utils.hdfs_io import copy, makedirs


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/aipilot/ai-platform/datasets/openseek_data/sft_data/Big-Math-RL-Verified-Processed_pri-mid")
    parser.add_argument("--ms_base_dir", default="/aipilot/ai-platform/datasets/openseek_data/sft_data/Big-Math-RL-Verified-Processed")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--val_ratio", type=float, default=0.05, help="Validation split ratio (0-1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling/splitting")

    args = parser.parse_args()

    ms_base_dir = args.ms_base_dir

    train_candidates = glob(os.path.join(ms_base_dir, "**", "big-math-rl-verified-processed-train.arrow"), recursive=True)
    # test_candidates = glob(os.path.join(ms_base_dir, "**", "gsm8k-test.arrow"), recursive=True)

    assert len(train_candidates) > 0, f"未在 {ms_base_dir} 下找到 gsm8k-train.arrow"
    # assert len(test_candidates) > 0, f"未在 {ms_base_dir} 下找到 gsm8k-test.arrow"

    train_data_source = max(train_candidates, key=os.path.getmtime)
    # test_data_source = max(test_candidates, key=os.path.getmtime)

    print(f"[Info] train arrow: {train_data_source}")
    # print(f"[Info] test  arrow: {test_data_source}")

    from datasets import Dataset
    train_dataset = Dataset.from_file(train_data_source)
    # test_dataset = Dataset.from_file(test_data_source)

    print(f"[Info] train rows: {train_dataset.num_rows}")
    # print(f"[Info] test  rows: {test_dataset.num_rows}")

    assert train_dataset.num_rows > 0, "train 数据为空"
    # assert test_dataset.num_rows > 0, "test 数据为空"

    # train_dataset = dataset["train"]
    # test_dataset = dataset["test"]

    # instruction_following = 'Let\'s think step by step and output the final answer after "####".'
    # instruction_following = 'Please reason step by step.\nIn the last line, write the answer after "The answer is:" and don\'t include any other text.'
    instruction_following = 'Please reason step by step, and put your final answer within \\boxed{}.\nQuestion:\n'

    # 先进行过滤，避免在 map 中返回 None 导致错误
    allowed_sources = {"orca_math", "cn_k12", "gsm8k"}

    def _filter_source(ex):
        return ex.get("source") in allowed_sources

    def _is_float_convertible_solution(ex):
        val = ex.get("solution")
        if val is None:
            return False
        try:
            float(str(val).strip().replace(",", ""))
            return True
        except Exception:
            return False

    print(f"[Info] before filter rows: {train_dataset.num_rows}")
    train_dataset = train_dataset.filter(_filter_source)
    print(f"[Info] after source filter (source in {sorted(list(allowed_sources))}) rows: {train_dataset.num_rows}")
    train_dataset = train_dataset.filter(_is_float_convertible_solution)
    print(f"[Info] after float-convertible solution filter rows: {train_dataset.num_rows}")

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            # print(example)
            question_raw = example.pop("prompt")

            question = instruction_following + question_raw 

            answer_raw = example.pop("solution")
            solution = answer_raw
            data = {
                # "data_source": data_source,
                "data_source": "openai/gsm8k",
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                },
            }
            return data

        return process_fn

    # 首先执行统一的 map 处理，随后再进行随机划分
    processed_dataset = train_dataset.map(function=make_map_fn("trainval"), with_indices=True)
    print(f"[Info] processed rows: {processed_dataset.num_rows}")

    # 随机划分 train/validation（默认 10% 为验证集），保证可复现性
    split_result = processed_dataset.train_test_split(test_size=args.val_ratio, seed=args.seed, shuffle=True)
    train_dataset = split_result["train"]
    val_dataset = split_result["test"]

    # 将 extra_info.split 字段分别标注为 train/validation
    def _set_split_field(split_name):
        def _fn(ex):
            extra = dict(ex.get("extra_info", {}))
            extra["split"] = split_name
            return {"extra_info": extra}
        return _fn

    train_dataset = train_dataset.map(_set_split_field("train"))
    val_dataset = val_dataset.map(_set_split_field("test"))
    print(f"[Info] split rows -> train: {train_dataset.num_rows}, val: {val_dataset.num_rows}")

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    os.makedirs(local_dir, exist_ok=True)

    # 分别输出 train/validation 两个 parquet 文件
    base_name = "big-math-rl-verified-processed_orca_cnk12_gsm8k_newprompt_2"
    train_output = os.path.join(local_dir, f"{base_name}_train.parquet")
    val_output = os.path.join(local_dir, f"{base_name}_val.parquet")
    train_dataset.to_parquet(train_output)
    val_dataset.to_parquet(val_output)
    print(f"[Info] saved -> train: {train_output}\n[Info] saved -> val  : {val_output}")

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
