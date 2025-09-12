import argparse
import os
from modelscope.msdatasets import MsDataset

def main():
    """
    主函数，从 ModelScope 加载数据集，进行处理，并保存为 Parquet 文件。
    """
    parser = argparse.ArgumentParser(description="Convert Big-Math dataset from ModelScope to a verl-compatible PARQUET format.")
    # 我们仍然保留 output_file 参数，以便您可以指定输出路径
    parser.add_argument("--output_file", type=str, required=True, help="Path for the output PARQUET file (e.g., train.parquet).")
    args = parser.parse_args()

    # 数据集信息
    dataset_name = 'open-r1/Big-Math-RL-Verified-Processed'
    subset_name = 'all'
    split = 'train'
    data_source_name = "Big-Math" # 用于在数据中标记来源

    print(f"Loading dataset '{dataset_name}' from ModelScope...")
    
    # 1. 使用 MsDataset.load 直接加载数据集
    #    这一步就已经得到了一个结构化的数据集对象
    dataset = MsDataset.load(dataset_name, subset_name=subset_name, split=split)

    print(f"Loaded {len(dataset)} records. Starting preprocessing...")

    # 2. 定义处理函数，将原始数据格式映射到目标格式
    #    这个函数会被 .map() 方法应用到每一条记录上
    def process_fn(example, idx):
        # 从原始记录中提取需要的字段
        # 注意：这里的键名 ('prompt', 'solution' 等) 需要根据您数据集的实际列名来定
        # 请根据 'open-r1/Big-Math-RL-Verified-Processed' 数据集的实际情况调整
        problem_raw = example.get("prompt", "")
        answer_clean = example.get("solution", "") 
        domain = example.get("domain", [])
        solve_rate = example.get("llama8b_solve_rate", None)
        
        # 构建 prompt 内容
        instruction = r'Please reason step by step,and must put your final answer within \boxed{}.Question:'
        prompt_content = instruction+ " " + problem_raw

        # 构建 reward_model 字段
        reward_model_data = {
            "style": "rule",
            "ground_truth": str(answer_clean) # 确保是字符串
        }
        
        # 组装成最终的数据结构
        processed_data = {
            "data_source": 'hiyouga/geometry3k',
            "prompt": [
                {
                    "role": "user",
                    "content": prompt_content,
                }
            ],
            "ability": "math",
            "reward_model": reward_model_data,
            "extra_info": {
                "index": idx,
                "original_problem": problem_raw,
                "domain": domain,
                "llama8b_solve_rate": solve_rate,
            },
        }
        return processed_data

    # 3. 使用 .map() 方法应用处理函数
    #    MsDataset 的 .map() 实现通常非常稳健
    processed_dataset = dataset.map(function=process_fn, with_indices=True)

    print("Preprocessing complete.")

    # 确保输出目录存在
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    # 4. 将处理好的数据集直接保存为 Parquet 文件
    print(f"Saving output to '{args.output_file}'...")
    processed_dataset.to_parquet(args.output_file)
    # processed_dataset.to_json(args.output_file, lines=True, force_ascii=False)
    
    print("Conversion finished successfully!")


if __name__ == "__main__":
    main()