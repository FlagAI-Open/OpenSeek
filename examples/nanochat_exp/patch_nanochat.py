#!/usr/bin/env python3
"""
自动修改 nanochat 的 base_train.py 以使用 OpenSeek dataloader 和 tokenizer。

这个脚本会备份原始文件，然后修改导入语句：
1. 将 dataloader 导入替换为 OpenSeek 的版本
2. 将 tokenizer 导入替换为 OpenSeek 的版本（使用 HuggingFace tokenizers，无需 rustbpe）
"""

import os
import sys
import shutil
from pathlib import Path

def patch_base_train(nanochat_path: str, backup: bool = True):
    """
    修改 nanochat 的 base_train.py 以使用 OpenSeek dataloader。
    
    Args:
        nanochat_path: nanochat 目录路径
        backup: 是否创建备份文件
    """
    base_train_path = Path(nanochat_path) / "scripts" / "base_train.py"
    
    if not base_train_path.exists():
        print(f"错误: 找不到文件 {base_train_path}")
        return False
    
    # 读取文件内容
    with open(base_train_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否已经修改过
    has_openseek_dataloader = "from examples.nanochat_exp.dataloader import" in content
    has_openseek_tokenizer = "from examples.nanochat_exp.tokenizer import" in content
    
    if has_openseek_dataloader and has_openseek_tokenizer:
        print(f"文件 {base_train_path} 已经完全修改过（dataloader 和 tokenizer），跳过")
        return True
    elif has_openseek_dataloader:
        print(f"文件 {base_train_path} 已部分修改（dataloader），需要更新 tokenizer")
        # 继续执行以更新 tokenizer
    elif has_openseek_tokenizer:
        print(f"文件 {base_train_path} 已部分修改（tokenizer），需要更新 dataloader")
        # 继续执行以更新 dataloader
    
    # 创建备份
    if backup:
        backup_path = base_train_path.with_suffix('.py.backup')
        shutil.copy2(base_train_path, backup_path)
        print(f"已创建备份: {backup_path}")
    
    # 替换 dataloader 导入语句（如果还没有替换）
    if not has_openseek_dataloader:
        old_dataloader_import = "from nanochat.dataloader import tokenizing_distributed_data_loader, tokenizing_distributed_data_loader_with_state"
        new_dataloader_import = "from examples.nanochat_exp.dataloader import tokenizing_distributed_data_loader, tokenizing_distributed_data_loader_with_state"
        
        if old_dataloader_import in content:
            content = content.replace(old_dataloader_import, new_dataloader_import)
            print(f"已替换 dataloader 导入语句")
        else:
            # 尝试只替换其中一个
            if "from nanochat.dataloader import tokenizing_distributed_data_loader_with_state" in content:
                content = content.replace(
                    "from nanochat.dataloader import tokenizing_distributed_data_loader_with_state",
                    "from examples.nanochat_exp.dataloader import tokenizing_distributed_data_loader_with_state"
                )
                print(f"已替换 tokenizing_distributed_data_loader_with_state 导入")
            if "from nanochat.dataloader import tokenizing_distributed_data_loader" in content:
                content = content.replace(
                    "from nanochat.dataloader import tokenizing_distributed_data_loader",
                    "from examples.nanochat_exp.dataloader import tokenizing_distributed_data_loader"
                )
                print(f"已替换 tokenizing_distributed_data_loader 导入")
    else:
        print(f"dataloader 导入已存在，跳过")
    
    # 替换 tokenizer 导入语句（使用 HuggingFace tokenizers，无需 rustbpe）
    # 匹配各种可能的导入形式
    import re
    
    # 如果还没有替换 tokenizer，进行替换
    if not has_openseek_tokenizer:
        # 检查是否有 nanochat.tokenizer 的导入
        has_nanochat_tokenizer = "from nanochat.tokenizer import" in content
        
        # 替换 "from nanochat.tokenizer import get_tokenizer"（各种形式）
        # 包括可能同时导入 get_token_bytes 的情况
        patterns = [
            # 匹配 "from nanochat.tokenizer import get_tokenizer, get_token_bytes"
            (r'from nanochat\.tokenizer import\s+(get_tokenizer|get_token_bytes|get_tokenizer\s*,\s*get_token_bytes|get_token_bytes\s*,\s*get_tokenizer)(\s+as\s+\w+)?',
             lambda m: f"from examples.nanochat_exp.tokenizer import get_tokenizer, get_token_bytes{m.group(2) or ''}"),
            # 匹配 "from nanochat.tokenizer import get_tokenizer"
            (r'from nanochat\.tokenizer import get_tokenizer(\s+as\s+\w+)?',
             lambda m: f"from examples.nanochat_exp.tokenizer import get_tokenizer{m.group(1) or ''}"),
        ]
        
        tokenizer_replaced = False
        for pattern, replacement in patterns:
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                tokenizer_replaced = True
                print(f"已替换 tokenizer 导入语句")
                break
        
        if not tokenizer_replaced and has_nanochat_tokenizer:
            print(f"警告: 检测到 nanochat.tokenizer 导入，但格式不匹配。尝试通用替换...")
            # 尝试更通用的替换 - 替换整个导入行
            content = re.sub(
                r'from nanochat\.tokenizer import\s+([^\n]+)',
                lambda m: f"from examples.nanochat_exp.tokenizer import {m.group(1)}",
                content
            )
            print(f"已尝试通用替换 tokenizer 导入")
    else:
        print(f"tokenizer 导入已存在，跳过")
    
    # 如果直接使用 get_tokenizer() 但没有导入，添加导入
    # 或者如果仍然使用 nanochat.tokenizer，替换它
    if "get_tokenizer()" in content or "get_tokenizer(" in content:
        if "from examples.nanochat_exp.tokenizer import" not in content:
            # 如果还有 nanochat.tokenizer 的导入，先替换它
            if "from nanochat.tokenizer import" in content:
                # 已经在上面的 pattern1 中处理了，这里确保替换
                content = re.sub(
                    r'from nanochat\.tokenizer import get_tokenizer(\s+as\s+\w+)?',
                    lambda m: f"from examples.nanochat_exp.tokenizer import get_tokenizer{m.group(1) or ''}",
                    content
                )
                print(f"已替换剩余的 tokenizer 导入语句")
            else:
                # 如果没有导入，添加导入
                lines = content.split('\n')
                insert_pos = 0
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    if stripped.startswith('from') or stripped.startswith('import'):
                        insert_pos = i + 1
                    elif stripped and not stripped.startswith('#') and insert_pos > 0:
                        break
                
                # 插入导入语句
                lines.insert(insert_pos, "from examples.nanochat_exp.tokenizer import get_tokenizer")
                content = '\n'.join(lines)
                print(f"已添加 tokenizer 导入语句")
    
    # 写入修改后的内容
    with open(base_train_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"成功修改文件: {base_train_path}")
    return True

def restore_base_train(nanochat_path: str):
    """恢复备份文件"""
    base_train_path = Path(nanochat_path) / "scripts" / "base_train.py"
    backup_path = base_train_path.with_suffix('.py.backup')
    
    if not backup_path.exists():
        print(f"错误: 找不到备份文件 {backup_path}")
        return False
    
    shutil.copy2(backup_path, base_train_path)
    print(f"已恢复文件: {base_train_path}")
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="修改 nanochat base_train.py 以使用 OpenSeek dataloader")
    parser.add_argument(
        "--nanochat-path",
        type=str,
        required=True,
        help="nanochat 目录路径"
    )
    parser.add_argument(
        "--restore",
        action="store_true",
        help="恢复原始文件（从备份）"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="不创建备份文件"
    )
    
    args = parser.parse_args()
    
    if args.restore:
        success = restore_base_train(args.nanochat_path)
    else:
        success = patch_base_train(args.nanochat_path, backup=not args.no_backup)
    
    sys.exit(0 if success else 1)
