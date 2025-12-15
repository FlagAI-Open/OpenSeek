#!/usr/bin/env python3
"""
诊断训练卡住的问题
检查数据、环境、GPU 等可能的问题
"""

import os
import sys
from pathlib import Path

def check_data_files():
    """检查数据文件是否存在"""
    print("=" * 80)
    print("1. 检查数据文件")
    print("=" * 80)
    
    data_dir = os.environ.get("OPENSEEK_NANOCHAT_DATA_DIR") or os.environ.get("NANOCHAT_BASE_DIR")
    if not data_dir:
        print("❌ 未设置 OPENSEEK_NANOCHAT_DATA_DIR 或 NANOCHAT_BASE_DIR")
        print("   请设置环境变量:")
        print("   export OPENSEEK_NANOCHAT_DATA_DIR=~/.cache/openseek_nanochat")
        return False
    
    print(f"✓ 数据目录: {data_dir}")
    
    parquet_dir = os.path.join(data_dir, "parquet_shards")
    if not os.path.exists(parquet_dir):
        print(f"❌ Parquet 目录不存在: {parquet_dir}")
        return False
    
    print(f"✓ Parquet 目录存在: {parquet_dir}")
    
    # 列出 parquet 文件
    try:
        parquet_files = [
            f for f in os.listdir(parquet_dir)
            if f.endswith('.parquet') and not f.endswith('.tmp')
        ]
        
        if not parquet_files:
            print(f"❌ 未找到 parquet 文件在: {parquet_dir}")
            print("   请先运行数据转换:")
            print("   python -m examples.nanochat_exp.dataset --dataset BAAI/OpenSeek-Pretrain-Data-Examples")
            return False
        
        print(f"✓ 找到 {len(parquet_files)} 个 parquet 文件")
        
        # 检查文件大小
        total_size = 0
        for f in parquet_files[:5]:  # 只检查前5个
            filepath = os.path.join(parquet_dir, f)
            size = os.path.getsize(filepath)
            total_size += size
            print(f"  - {f}: {size / 1024 / 1024:.2f} MB")
        
        if len(parquet_files) > 5:
            print(f"  ... 还有 {len(parquet_files) - 5} 个文件")
        
        if total_size == 0:
            print("⚠️  警告: 文件大小可能为 0，数据可能未正确转换")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 检查 parquet 文件时出错: {e}")
        return False

def check_tokenizer():
    """检查 tokenizer 是否存在"""
    print("\n" + "=" * 80)
    print("2. 检查 Tokenizer")
    print("=" * 80)
    
    data_dir = os.environ.get("OPENSEEK_NANOCHAT_DATA_DIR") or os.environ.get("NANOCHAT_BASE_DIR")
    if not data_dir:
        print("❌ 未设置数据目录")
        return False
    
    tokenizer_dir = os.path.join(data_dir, "tokenizer")
    tokenizer_json = os.path.join(tokenizer_dir, "tokenizer.json")
    tokenizer_pkl = os.path.join(tokenizer_dir, "tokenizer.pkl")
    
    if os.path.exists(tokenizer_json):
        print(f"✓ 找到 tokenizer.json: {tokenizer_json}")
        
        # 检查文件大小
        size = os.path.getsize(tokenizer_json)
        print(f"  文件大小: {size / 1024:.2f} KB")
        
        if size < 1000:
            print("⚠️  警告: tokenizer.json 文件很小，可能未正确训练")
            return False
        
        return True
    elif os.path.exists(tokenizer_pkl):
        print(f"✓ 找到 tokenizer.pkl: {tokenizer_pkl}")
        return True
    else:
        print(f"❌ 未找到 tokenizer 文件")
        print(f"   期望位置: {tokenizer_dir}/tokenizer.json 或 tokenizer.pkl")
        print("   请运行:")
        print("   python -m examples.nanochat_exp.tok_train --data-dir <parquet_dir>")
        return False

def check_gpu():
    """检查 GPU 是否可用"""
    print("\n" + "=" * 80)
    print("3. 检查 GPU")
    print("=" * 80)
    
    try:
        import torch
        print(f"✓ PyTorch 版本: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA 可用")
            print(f"  CUDA 版本: {torch.version.cuda}")
            print(f"  GPU 数量: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name}")
                print(f"    显存: {props.total_memory / 1024**3:.2f} GB")
        else:
            print("⚠️  CUDA 不可用，将使用 CPU（训练会很慢）")
            return False
        
        return True
    except ImportError:
        print("❌ PyTorch 未安装")
        return False
    except Exception as e:
        print(f"❌ 检查 GPU 时出错: {e}")
        return False

def check_data_loading():
    """测试数据加载"""
    print("\n" + "=" * 80)
    print("4. 测试数据加载")
    print("=" * 80)
    
    try:
        # 设置环境变量
        data_dir = os.environ.get("OPENSEEK_NANOCHAT_DATA_DIR") or os.environ.get("NANOCHAT_BASE_DIR")
        if data_dir:
            os.environ["NANOCHAT_BASE_DIR"] = data_dir
        
        # 尝试导入数据加载器
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from examples.nanochat_exp.dataset import list_parquet_files, parquets_iter_batched
        
        parquet_paths = list_parquet_files()
        print(f"✓ 找到 {len(parquet_paths)} 个 parquet 文件")
        
        if not parquet_paths:
            print("❌ 没有 parquet 文件可加载")
            return False
        
        # 尝试加载第一个批次
        print("  尝试加载第一个批次...")
        batch_count = 0
        for batch in parquets_iter_batched(split="train", start=0, step=1):
            batch_count += 1
            print(f"  ✓ 成功加载批次 {batch_count}: {len(batch)} 个文档")
            if batch_count >= 2:  # 只测试前2个批次
                break
        
        if batch_count == 0:
            print("❌ 无法加载任何批次，数据可能有问题")
            return False
        
        print(f"✓ 数据加载正常（测试了 {batch_count} 个批次）")
        return True
        
    except Exception as e:
        print(f"❌ 数据加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_environment():
    """检查环境变量"""
    print("\n" + "=" * 80)
    print("5. 检查环境变量")
    print("=" * 80)
    
    env_vars = [
        "OPENSEEK_NANOCHAT_DATA_DIR",
        "NANOCHAT_BASE_DIR",
        "PYTHONPATH",
        "CUDA_VISIBLE_DEVICES",
    ]
    
    all_ok = True
    for var in env_vars:
        value = os.environ.get(var)
        if value:
            print(f"✓ {var} = {value}")
        else:
            if var in ["OPENSEEK_NANOCHAT_DATA_DIR", "NANOCHAT_BASE_DIR"]:
                print(f"⚠️  {var} 未设置（可能使用默认值）")
            else:
                print(f"  {var} 未设置")
    
    return True

def main():
    print("\n" + "=" * 80)
    print("训练诊断工具")
    print("=" * 80)
    print()
    
    results = {
        "数据文件": check_data_files(),
        "Tokenizer": check_tokenizer(),
        "GPU": check_gpu(),
        "环境变量": check_environment(),
        "数据加载": check_data_loading(),
    }
    
    print("\n" + "=" * 80)
    print("诊断结果总结")
    print("=" * 80)
    
    all_ok = True
    for name, ok in results.items():
        status = "✓ 通过" if ok else "❌ 失败"
        print(f"{name}: {status}")
        if not ok:
            all_ok = False
    
    print()
    if all_ok:
        print("✓ 所有检查通过！如果训练仍然卡住，可能是以下原因:")
        print("  1. 分布式训练同步问题（多 GPU 时）")
        print("  2. 数据加载速度慢（首次加载需要时间）")
        print("  3. Wandb 初始化问题（尝试禁用 wandb）")
        print("  4. 网络问题（如果使用 streaming 模式）")
        print()
        print("建议:")
        print("  - 查看训练输出，看卡在哪一步")
        print("  - 尝试单 GPU 训练: NPROC_PER_NODE=1 bash run_openseek_exp.sh")
        print("  - 检查系统资源: htop, nvidia-smi")
    else:
        print("❌ 发现问题，请先解决上述问题再运行训练")
    
    print()

if __name__ == "__main__":
    main()
