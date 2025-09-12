from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, GenerationConfig, TextStreamer
import time
import torch
import json
import os
from datetime import datetime
from collections import OrderedDict

torch.manual_seed(0)

checkpoint_paths = []

checkpoint_paths.append(r"./models/OpenSeek_Small_v1")
checkpoint_paths.append(r"./data/OpenSeek-1.4B-A0.4B/checkpoint-1000")
checkpoint_paths.append(r"./data/OpenSeek-1.4B-A0.4B/checkpoint-2000")
checkpoint_paths.append(r"./data/OpenSeek-1.4B-A0.4B/checkpoint-3000")
checkpoint_paths.append(r"./data/OpenSeek-1.4B-A0.4B/checkpoint-4000")
checkpoint_paths.append(r"./data/OpenSeek-1.4B-A0.4B/checkpoint-5000")
checkpoint_paths.append(r"./data/OpenSeek-1.4B-A0.4B/checkpoint-5000")
checkpoint_paths.append(r"./data/OpenSeek-1.4B-A0.4B/checkpoint-6000")
checkpoint_paths.append(r"./data/OpenSeek-1.4B-A0.4B/checkpoint-7000")


# merge这些模型
print("Merging models...")

def merge_checkpoints(checkpoint_paths, output_path, merge_method="average"):
    """
    合并多个checkpoints
    
    Args:
        checkpoint_paths: 要合并的checkpoint路径列表
        output_path: 合并后模型的保存路径
        merge_method: 合并方法，支持 "average", "last", "weighted_average"
    """
    print(f"Using merge method: {merge_method}")
    
    # 加载第一个模型作为基础
    base_model = AutoModelForCausalLM.from_pretrained(checkpoint_paths[0], trust_remote_code=True)
    base_config = AutoConfig.from_pretrained(checkpoint_paths[0], trust_remote_code=True)
    base_tokenizer = AutoTokenizer.from_pretrained(checkpoint_paths[0], trust_remote_code=True)
    
    if merge_method == "average":
        # 平均合并所有checkpoints
        print("Averaging weights from all checkpoints...")
        merged_state_dict = OrderedDict()
        weight_counts = OrderedDict()  # 记录每个权重参与合并的次数
        
        # 初始化merged_state_dict和weight_counts
        for key in base_model.state_dict():
            merged_state_dict[key] = torch.zeros_like(base_model.state_dict()[key])
            weight_counts[key] = 0
        
        # 累加所有模型的权重
        for i, checkpoint_path in enumerate(checkpoint_paths):
            print(f"Processing checkpoint {i+1}/{len(checkpoint_paths)}: {checkpoint_path}")
            
            model = AutoModelForCausalLM.from_pretrained(checkpoint_path, trust_remote_code=True)
            model_state_dict = model.state_dict()
            
            # 只合并匹配的权重
            matched_keys = 0
            total_keys = len(model_state_dict)
            for key in base_model.state_dict():
                if key in model_state_dict:
                    
                    # 检查形状是否匹配
                    if merged_state_dict[key].shape == model_state_dict[key].shape:
                        merged_state_dict[key] += model_state_dict[key]
                        weight_counts[key] += 1
                        matched_keys += 1
                    else:
                        print(f"Warning: Shape mismatch for key {key}: base {merged_state_dict[key].shape} vs model {model_state_dict[key].shape}")
                else:
                    print(f"Warning: Key {key} not found in model {checkpoint_path}")
            
            print(f"  Matched {matched_keys}/{len(base_model.state_dict())} keys from base model")
            print(f"  Model has {total_keys} total keys")
            
            # 释放内存
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # 计算平均值（只对参与合并的权重求平均）
        for key in merged_state_dict:
            if weight_counts[key] > 0:
                merged_state_dict[key] /= weight_counts[key]
                print(f"Key {key}: averaged over {weight_counts[key]} models")
            else:
                # 如果某个权重没有参与任何合并，使用base_model的权重
                merged_state_dict[key] = base_model.state_dict()[key].clone()
                print(f"Key {key}: using base model weight (no matches found)")
    
    elif merge_method == "weighted_average":
        # 加权平均合并（后面的checkpoint权重更高）
        print("Weighted averaging weights from all checkpoints...")
        merged_state_dict = OrderedDict()
        weight_sums = OrderedDict()  # 记录每个权重的总权重
        
        # 初始化merged_state_dict和weight_sums
        for key in base_model.state_dict():
            merged_state_dict[key] = torch.zeros_like(base_model.state_dict()[key])
            weight_sums[key] = 0.0
        
        # 计算权重（线性递增）
        weights = [i+1 for i in range(len(checkpoint_paths))]
        total_weight = sum(weights)
        weights = [w/total_weight for w in weights]
        
        print(f"Weights: {weights}")
        
        # 加权累加所有模型的权重
        for i, (checkpoint_path, weight) in enumerate(zip(checkpoint_paths, weights)):
            print(f"Processing checkpoint {i+1}/{len(checkpoint_paths)}: {checkpoint_path} (weight: {weight:.3f})")
            
            model = AutoModelForCausalLM.from_pretrained(checkpoint_path, trust_remote_code=True)
            model_state_dict = model.state_dict()
            
            # 只合并匹配的权重
            matched_keys = 0
            total_keys = len(model_state_dict)
            for key in base_model.state_dict():
                if key in model_state_dict:
                    
                    # 检查形状是否匹配
                    if merged_state_dict[key].shape == model_state_dict[key].shape:
                        merged_state_dict[key] += model_state_dict[key] * weight
                        weight_sums[key] += weight
                        matched_keys += 1
                    else:
                        print(f"Warning: Shape mismatch for key {key}: base {merged_state_dict[key].shape} vs model {model_state_dict[key].shape}")
                else:
                    print(f"Warning: Key {key} not found in model {checkpoint_path}")
            
            print(f"  Matched {matched_keys}/{len(base_model.state_dict())} keys from base model")
            print(f"  Model has {total_keys} total keys")
            
            # 释放内存
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # 最终加权平均结果打印
        for key in merged_state_dict:
            if weight_sums[key] > 0:
                print(f"Key {key}: weighted averaged over {weight_sums[key]:.3f} total weight")
            else:
                merged_state_dict[key] = base_model.state_dict()[key].clone()
                print(f"Key {key}: using base model weight (no matches found)")
    
    elif merge_method == "last":
        # 只使用最后一个checkpoint
        print("Using the last checkpoint...")
        last_model = AutoModelForCausalLM.from_pretrained(checkpoint_paths[-1], trust_remote_code=True)
        merged_state_dict = last_model.state_dict()
        del last_model
    
    else:
        raise ValueError(f"Unknown merge method: {merge_method}")
    
    # 加载合并后的权重到基础模型
    base_model.load_state_dict(merged_state_dict)
    
    # 保存合并后的模型
    print(f"Saving merged model to {output_path}")
    os.makedirs(output_path, exist_ok=True)
    base_model = base_model.to(torch.bfloat16)
    base_model.save_pretrained(output_path)
    base_tokenizer.save_pretrained(output_path)
    base_config.save_pretrained(output_path)
    
    # 保存合并信息
    merge_info = {
        "merge_method": merge_method,
        "merged_checkpoints": checkpoint_paths,
        "merge_time": datetime.now().isoformat(),
        "num_checkpoints": len(checkpoint_paths)
    }
    
    with open(os.path.join(output_path, "merge_info.json"), "w", encoding="utf-8") as f:
        json.dump(merge_info, f, indent=2, ensure_ascii=False)
    
    print(f"Merge completed! Model saved to {output_path}")
    return base_model, base_tokenizer

# 执行合并
if checkpoint_paths:
    print(f"Total checkpoints to merge: {len(checkpoint_paths)}")
    
    # 方法1: 平均合并
    output_path_avg = "./models/OpenSeek_Small_v1-merged-average"
    merged_model_avg, merged_tokenizer_avg = merge_checkpoints(
        checkpoint_paths, 
        output_path_avg, 
        merge_method="average"
    )
    
    # # 方法2: 加权平均合并
    # output_path_weighted = "./models/OpenSeek_Small_v1-merged-weighted"
    # merged_model_weighted, merged_tokenizer_weighted = merge_checkpoints(
    #     checkpoint_paths, 
    #     output_path_weighted, 
    #     merge_method="weighted_average"
    # )
    
    # # 方法3: 使用最后一个checkpoint
    # output_path_last = "./models/OpenSeek_Small_v1-merged-last"
    # merged_model_last, merged_tokenizer_last = merge_checkpoints(
    #     checkpoint_paths, 
    #     output_path_last, 
    #     merge_method="last"
    # )
    
    # print("All merge methods completed!")
    # print(f"Average merge: {output_path_avg}")
    # print(f"Weighted average merge: {output_path_weighted}")
    # print(f"Last checkpoint: {output_path_last}")

else:
    print("No checkpoints found to merge!")