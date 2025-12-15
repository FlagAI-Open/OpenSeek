#!/bin/bash

# OpenSeek nanochat experiment runner
# This script runs nanochat training using OpenSeek datasets

set -e

# Default settings
export OMP_NUM_THREADS=1
export OPENSEEK_NANOCHAT_DATA_DIR="${OPENSEEK_NANOCHAT_DATA_DIR:-$HOME/.cache/openseek_nanochat}"
mkdir -p "$OPENSEEK_NANOCHAT_DATA_DIR"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENSEEK_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Try to find nanochat and add to PYTHONPATH
NANOCHAT_DIR=""
# Check common locations for nanochat
for dir in "$(dirname "$OPENSEEK_ROOT")/nanochat" "$HOME/nanochat" "/opt/nanochat"; do
    if [ -d "$dir" ] && [ -f "$dir/nanochat/__init__.py" ] 2>/dev/null; then
        NANOCHAT_DIR="$dir"
        break
    fi
done

# Add nanochat and OpenSeek to PYTHONPATH if found
if [ -n "$NANOCHAT_DIR" ]; then
    export PYTHONPATH="$NANOCHAT_DIR:${PYTHONPATH}"
    echo "Found nanochat at: $NANOCHAT_DIR"
fi
export PYTHONPATH="$OPENSEEK_ROOT:${PYTHONPATH}"

echo "=========================================="
echo "OpenSeek nanochat Experiment Runner"
echo "=========================================="
echo "OpenSeek root: $OPENSEEK_ROOT"
echo "Data directory: $OPENSEEK_NANOCHAT_DATA_DIR"
if [ -n "$NANOCHAT_DIR" ]; then
    echo "Nanochat directory: $NANOCHAT_DIR"
fi
echo "=========================================="
echo

# Check if nanochat is available
if ! python -c "import nanochat" 2>/dev/null; then
    echo "Error: nanochat is not installed or not in Python path"
    echo "Please install nanochat or add it to your PYTHONPATH"
    echo "You can clone nanochat from: https://github.com/karpathy/nanochat"
    echo ""
    echo "Suggested locations:"
    echo "  - $(dirname "$OPENSEEK_ROOT")/nanochat"
    echo "  - $HOME/nanochat"
    exit 1
fi

# Check and install required dependencies
echo "检查依赖项..."
MISSING_DEPS=()

# Check for tokenizers (required by nanochat)
if ! python -c "import tokenizers" 2>/dev/null; then
    MISSING_DEPS+=("tokenizers>=0.22.0")
fi

# Check for torch
if ! python -c "import torch" 2>/dev/null; then
    MISSING_DEPS+=("torch>=2.8.0")
fi

# Check for datasets
if ! python -c "import datasets" 2>/dev/null; then
    MISSING_DEPS+=("datasets>=4.0.0")
fi

# Check for wandb (required by nanochat)
if ! python -c "import wandb" 2>/dev/null; then
    MISSING_DEPS+=("wandb>=0.21.3")
fi

# Check for tiktoken (required by nanochat)
if ! python -c "import tiktoken" 2>/dev/null; then
    MISSING_DEPS+=("tiktoken>=0.11.0")
fi

# Check for HuggingFace tokenizers library (easy-to-install alternative to rustbpe)
echo "检查 HuggingFace tokenizers 库..."
if ! python3 -c "from tokenizers import Tokenizer" 2>/dev/null; then
    echo "警告: tokenizers 库未安装"
    echo "OpenSeek 使用 HuggingFace tokenizers 库作为 rustbpe 的替代（更容易安装，无需 Rust 编译）。"
    echo ""
    MISSING_DEPS+=("tokenizers>=0.22.0")
else
    echo "✓ tokenizers 库可用（无需 rustbpe，安装更简单）"
fi

if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo "发现缺失的依赖项:"
    for dep in "${MISSING_DEPS[@]}"; do
        echo "  - $dep"
    done
    echo ""
    read -p "是否自动安装缺失的依赖项？(Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        echo "正在安装依赖项..."
        for dep in "${MISSING_DEPS[@]}"; do
            echo "安装 $dep..."
            pip install "$dep" || {
                echo "错误: 安装 $dep 失败"
                echo "请手动安装: pip install $dep"
                exit 1
            }
        done
        echo "依赖项安装完成"
    else
        echo "跳过依赖项安装。请手动安装:"
        for dep in "${MISSING_DEPS[@]}"; do
            echo "  pip install $dep"
        done
        exit 1
    fi
else
    echo "所有必需的依赖项已安装"
fi


# Step 1: Convert OpenSeek dataset to parquet format
echo "Step 1: Converting OpenSeek dataset to parquet format..."
echo "----------------------------------------------------------"

# Check if parquet files already exist
PARQUET_DIR="$OPENSEEK_NANOCHAT_DATA_DIR/parquet_shards"
if [ -d "$PARQUET_DIR" ] && [ "$(ls -A $PARQUET_DIR/*.parquet 2>/dev/null)" ]; then
    echo "Parquet files already exist in $PARQUET_DIR"
    read -p "Do you want to regenerate them? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping dataset conversion..."
        SKIP_CONVERSION=true
    fi
fi

# Check if user wants to skip download and use streaming mode
if [ "${SKIP_DOWNLOAD:-false}" = "true" ]; then
    echo "跳过下载，将使用 streaming 模式直接从 HuggingFace 加载数据"
    USE_STREAMING=true
fi

if [ "${SKIP_CONVERSION:-false}" != "true" ]; then
    # Check if OpenSeek dataset is available
    DATASET_PATH="$OPENSEEK_NANOCHAT_DATA_DIR"
    DATASET_NAME="BAAI/OpenSeek-Pretrain-Data-Examples"
    
    # 设置 HuggingFace 镜像端点（使用 hf-mirror.com）
    export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
    echo "使用 HuggingFace 镜像: $HF_ENDPOINT"
    
    # 检查数据集是否已下载
    HAS_DATA_FILES=false
    if [ -d "$DATASET_PATH" ]; then
        # 检查是否包含数据集文件
        if ls "$DATASET_PATH"/*.arrow 2>/dev/null | grep -q . || \
           ls "$DATASET_PATH"/*.parquet 2>/dev/null | grep -q . || \
           ls "$DATASET_PATH"/*.jsonl 2>/dev/null | grep -q .; then
            HAS_DATA_FILES=true
        fi
    fi
    
    if [ "$HAS_DATA_FILES" = false ]; then
        echo "OpenSeek-Pretrain-Data-Examples 未找到或数据文件不完整"
        
        # 如果使用 streaming 模式，跳过下载
        if [ "${USE_STREAMING:-false}" = "true" ]; then
            echo "使用 streaming 模式，跳过本地下载"
            echo "数据集将在转换时直接从 HuggingFace 流式加载"
        elif [ "${SKIP_DOWNLOAD:-false}" = "true" ]; then
            echo "跳过下载步骤"
        else
            # 检查 huggingface-cli 是否可用
            if command -v huggingface-cli &> /dev/null; then
                echo "使用 huggingface-cli 下载数据集..."
                # 确保目录存在
                mkdir -p "$DATASET_PATH"
                
                # 使用 huggingface-cli 下载数据集
                echo "开始下载数据集: $DATASET_NAME"
                echo "保存到: $DATASET_PATH"
                
                if huggingface-cli download \
                    --repo-type dataset \
                    --resume-download \
                    "$DATASET_NAME" \
                    --local-dir "$DATASET_PATH" \
                    --local-dir-use-symlinks False; then
                    echo "数据集下载完成"
                else
                    echo "huggingface-cli 下载失败，将使用 Python 自动下载..."
                    USE_PYTHON_DOWNLOAD=true
                fi
            else
                echo "huggingface-cli 未找到，将使用 Python 自动下载数据集..."
                USE_PYTHON_DOWNLOAD=true
            fi
        fi
        
        # 如果 huggingface-cli 不可用或下载失败，使用 Python 下载
        if [ "${USE_PYTHON_DOWNLOAD:-false}" = "true" ]; then
            echo "使用 Python 的 datasets 库下载数据集..."
            echo "注意: 这需要安装 datasets 和 huggingface_hub 库"
            echo "如果未安装，请运行: pip install datasets huggingface_hub"
            
            # 创建一个临时 Python 脚本来下载数据集（支持超时、重试和进度监控）
            TEMP_DOWNLOAD_SCRIPT=$(mktemp)
            cat > "$TEMP_DOWNLOAD_SCRIPT" << 'PYTHON_EOF'
import os
import sys
import time
import signal
import threading
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError

dataset_name = sys.argv[1]
local_dir = sys.argv[2]
hf_endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
max_retries = int(os.environ.get("HF_DOWNLOAD_MAX_RETRIES", "3"))
retry_delay = int(os.environ.get("HF_DOWNLOAD_RETRY_DELAY", "5"))
timeout_seconds = int(os.environ.get("HF_DOWNLOAD_TIMEOUT", "300"))  # 5分钟超时
progress_timeout = int(os.environ.get("HF_DOWNLOAD_PROGRESS_TIMEOUT", "120"))  # 2分钟无进度超时

print(f"从 HuggingFace 下载数据集: {dataset_name}")
print(f"镜像端点: {hf_endpoint}")
print(f"保存到: {local_dir}")
print(f"最大重试次数: {max_retries}")
print(f"重试延迟: {retry_delay}秒")
print(f"下载超时: {timeout_seconds}秒")
print(f"进度超时: {progress_timeout}秒（无进度时自动重试）")
print("\n提示: 如果下载速度太慢或卡住，可以:")
print("  1. 按 Ctrl+C 中断下载（已下载的文件会保留）")
print("  2. 重新运行脚本会自动恢复下载")
print("  3. 或使用 streaming 模式（不需要完整下载）")
print("  4. 设置环境变量: export USE_STREAMING=true")
print()

os.makedirs(local_dir, exist_ok=True)
os.environ["HF_ENDPOINT"] = hf_endpoint

# 进度监控变量
last_progress_time = time.time()
download_completed = False
download_error = None

def check_progress():
    """检查下载进度，如果长时间无进度则触发超时"""
    global last_progress_time, download_completed, download_error
    while not download_completed:
        time.sleep(30)  # 每30秒检查一次
        if download_completed:
            break
        elapsed = time.time() - last_progress_time
        if elapsed > progress_timeout:
            print(f"\n警告: 下载已超过 {progress_timeout} 秒无进度，可能已卡住")
            print("建议:")
            print("  1. 按 Ctrl+C 中断并重新运行（支持断点续传）")
            print("  2. 或使用 streaming 模式: export USE_STREAMING=true")
            # 不强制退出，让用户决定

# 尝试下载，支持重试
for attempt in range(1, max_retries + 1):
    try:
        print(f"开始下载（尝试 {attempt}/{max_retries}）...")
        last_progress_time = time.time()
        download_completed = False
        
        # 启动进度监控线程
        progress_thread = threading.Thread(target=check_progress, daemon=True)
        progress_thread.start()
        
        # 使用 snapshot_download 下载整个数据集仓库
        # 注意：新版本的 huggingface_hub 已经移除了 resume_download 和 local_dir_use_symlinks 参数
        snapshot_download(
            repo_id=dataset_name,
            repo_type="dataset",
            local_dir=local_dir,
        )
        
        download_completed = True
        print(f"\n数据集下载完成: {local_dir}")
        sys.exit(0)
        
    except KeyboardInterrupt:
        print("\n\n下载被用户中断")
        download_completed = True
        print(f"已下载的文件保存在: {local_dir}")
        print("重新运行脚本可以继续下载（支持断点续传）")
        print("\n或者使用 streaming 模式跳过下载:")
        print("  export USE_STREAMING=true")
        print("  bash run_openseek_exp.sh")
        sys.exit(130)
    except HfHubHTTPError as e:
        download_completed = True
        if attempt < max_retries:
            print(f"\n下载失败（尝试 {attempt}/{max_retries}）: {e}")
            print(f"等待 {retry_delay} 秒后重试...")
            time.sleep(retry_delay)
            download_completed = False
        else:
            print(f"\n下载失败（已重试 {max_retries} 次）: {e}")
            raise
    except Exception as e:
        download_completed = True
        print(f"\n下载失败: {e}")
        print("\n请确保已安装必要的库:")
        print("  pip install huggingface_hub")
        print("\n如果已安装但仍失败，请检查:")
        print("  1. 网络连接是否正常")
        print("  2. 数据集名称是否正确")
        print("  3. 是否有足够的磁盘空间")
        print("  4. 镜像站点是否可访问")
        print("\n建议:")
        print("  - 尝试使用 streaming 模式（不需要完整下载）:")
        print("    export USE_STREAMING=true")
        print("    bash run_openseek_exp.sh")
        print("  - 或手动下载数据集到: " + local_dir)
        sys.exit(1)
PYTHON_EOF
            
            if python "$TEMP_DOWNLOAD_SCRIPT" "$DATASET_NAME" "$DATASET_PATH"; then
                echo "数据集下载完成"
                rm -f "$TEMP_DOWNLOAD_SCRIPT"
            else
                DOWNLOAD_EXIT_CODE=$?
                rm -f "$TEMP_DOWNLOAD_SCRIPT"
                
                echo ""
                echo "=========================================="
                echo "下载失败（退出码: $DOWNLOAD_EXIT_CODE）"
                echo "=========================================="
                echo ""
                
                # 如果用户中断（Ctrl+C），退出码是 130
                if [ "$DOWNLOAD_EXIT_CODE" = "130" ]; then
                    echo "下载被用户中断"
                    echo "已下载的文件保存在: $DATASET_PATH"
                    echo ""
                    echo "选项："
                    echo "  1. 重新运行脚本继续下载（支持断点续传）"
                    echo "  2. 使用 streaming 模式跳过下载:"
                    echo "     export USE_STREAMING=true"
                    echo "     bash $0"
                    exit 130
                fi
                
                # 检查是否已有部分文件下载
                if [ -d "$DATASET_PATH" ] && [ "$(ls -A $DATASET_PATH 2>/dev/null)" ]; then
                    echo "检测到部分文件已下载，但下载不完整"
                    echo "已下载文件位置: $DATASET_PATH"
                    echo ""
                fi
                
                echo "下载失败的可能原因："
                echo "  1. 网络连接问题（无法访问 HuggingFace 镜像）"
                echo "  2. 镜像站点响应慢或超时"
                echo "  3. 磁盘空间不足"
                echo ""
                echo "建议解决方案："
                echo "  方案 1（推荐）: 使用 streaming 模式，无需下载完整数据集"
                echo "    export USE_STREAMING=true"
                echo "    bash $0"
                echo ""
                echo "  方案 2: 手动下载数据集"
                echo "    export HF_ENDPOINT=https://hf-mirror.com"
                echo "    huggingface-cli download --repo-type dataset \\"
                echo "      --resume-download $DATASET_NAME \\"
                echo "      --local-dir $DATASET_PATH"
                echo ""
                echo "  方案 3: 检查网络连接后重试"
                echo "    bash $0"
                echo ""
                
                # 如果用户没有明确禁用自动切换，询问是否使用 streaming 模式
                if [ "${AUTO_USE_STREAMING_ON_FAILURE:-true}" = "true" ] && [ "${USE_STREAMING:-false}" != "true" ]; then
                    echo "是否自动切换到 streaming 模式？(Y/n)"
                    read -t 10 -n 1 -r REPLY || REPLY="y"
                    echo
                    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                        echo "自动切换到 streaming 模式..."
                        export USE_STREAMING=true
                        # 继续执行，使用 streaming 模式
                    else
                        echo "保持当前模式，退出脚本"
                        exit 1
                    fi
                else
                    exit 1
                fi
            fi
        fi
    else
        echo "使用本地数据集: $DATASET_PATH"
    fi
    
    # 转换数据集为 parquet 格式
    echo "开始转换数据集为 parquet 格式..."
    # 对于示例数据集，默认使用 -1（处理所有数据，不限制 shards 数量）
    # 可以通过环境变量 NUM_SHARDS 自定义，例如：export NUM_SHARDS=10
    
    # 构建转换命令
    CONVERSION_CMD="python -m examples.nanochat_exp.dataset --dataset"
    
    # 如果使用 streaming 模式，使用数据集名称而不是本地路径
    if [ "${USE_STREAMING:-false}" = "true" ]; then
        CONVERSION_CMD="$CONVERSION_CMD \"$DATASET_NAME\" --streaming"
        echo "使用 streaming 模式（不需要完整下载数据集）"
    else
        CONVERSION_CMD="$CONVERSION_CMD \"$DATASET_PATH\""
    fi
    
    CONVERSION_CMD="$CONVERSION_CMD --num-shards \"${NUM_SHARDS:--1}\""
    
    # 添加 rows-per-shard 参数（默认 2048）
    ROWS_PER_SHARD="${ROWS_PER_SHARD:-2048}"
    CONVERSION_CMD="$CONVERSION_CMD --rows-per-shard \"$ROWS_PER_SHARD\""
    echo "每个 parquet shard 的行数限制: $ROWS_PER_SHARD"
    
    eval $CONVERSION_CMD
fi

echo
echo "Step 2: Training with nanochat..."
echo "----------------------------------------------------------"

# Set nanochat data directory to use our parquet files
export NANOCHAT_BASE_DIR="$OPENSEEK_NANOCHAT_DATA_DIR"

# Check if tokenizer exists and validate it
TOKENIZER_DIR="$OPENSEEK_NANOCHAT_DATA_DIR/tokenizer"
TOKENIZER_EXISTS=false
TOKENIZER_VALID=false
TOKENIZER_VOCAB_SIZE=0

if [ -f "$TOKENIZER_DIR/tokenizer.json" ]; then
    TOKENIZER_EXISTS=true
    echo "检测到 tokenizer.json 文件"
    
    # Check vocab size using Python
    echo "检查 tokenizer 词汇表大小..."
    TOKENIZER_JSON_PATH="$TOKENIZER_DIR/tokenizer.json"
    VOCAB_SIZE_CHECK=$(python3 << PYTHON_EOF
import sys
import os
import json

tokenizer_path = "$TOKENIZER_JSON_PATH"
try:
    if not os.path.exists(tokenizer_path):
        print("0")
        sys.exit(0)
    
    from tokenizers import Tokenizer as HFTokenizer
    tokenizer = HFTokenizer.from_file(tokenizer_path)
    
    vocab_size = None
    # Try multiple methods to get vocab size
    if hasattr(tokenizer, 'get_vocab_size'):
        vocab_size = tokenizer.get_vocab_size()
    elif hasattr(tokenizer, 'get_vocab'):
        vocab = tokenizer.get_vocab()
        if vocab:
            vocab_size = len(vocab)
    elif hasattr(tokenizer, 'model'):
        model = tokenizer.model
        if hasattr(model, 'get_vocab_size'):
            vocab_size = model.get_vocab_size()
        elif hasattr(model, 'vocab') and model.vocab:
            vocab_size = len(model.vocab)
        elif hasattr(model, 'merges') and model.merges:
            vocab_size = 256 + len(model.merges) + 4
    
    if vocab_size:
        print(vocab_size)
    else:
        print("0")
except Exception as e:
    print("0")
PYTHON_EOF
)
    
    if [ -n "$VOCAB_SIZE_CHECK" ] && [ "$VOCAB_SIZE_CHECK" != "0" ]; then
        TOKENIZER_VOCAB_SIZE=$VOCAB_SIZE_CHECK
        echo "Tokenizer 词汇表大小: $TOKENIZER_VOCAB_SIZE"
        
        # Check if vocab size is valid (should be > 10, ideally > 1000)
        if [ "$TOKENIZER_VOCAB_SIZE" -gt 10 ]; then
            TOKENIZER_VALID=true
            echo "✓ Tokenizer 看起来正常"
        else
            echo "⚠️  警告: Tokenizer 词汇表大小只有 $TOKENIZER_VOCAB_SIZE，可能未正确训练"
            echo "   只有特殊 tokens，无法用于实际训练"
        fi
    else
        echo "⚠️  警告: 无法读取 tokenizer 词汇表大小"
    fi
elif [ -f "$TOKENIZER_DIR/tokenizer.pkl" ]; then
    TOKENIZER_EXISTS=true
    echo "检测到 tokenizer.pkl 文件（nanochat 格式）"
    TOKENIZER_VALID=true  # Assume valid if pkl exists
fi

if [ "$TOKENIZER_EXISTS" = false ] || [ "$TOKENIZER_VALID" = false ]; then
    echo ""
    echo "警告: Tokenizer 文件未找到"
    echo "Tokenizer 目录: $TOKENIZER_DIR"
    echo ""
    if [ "$TOKENIZER_EXISTS" = true ] && [ "$TOKENIZER_VALID" = false ]; then
        echo "⚠️  警告: Tokenizer 文件存在但词汇表大小异常（只有 $TOKENIZER_VOCAB_SIZE tokens）"
        echo ""
        echo "这通常表示 tokenizer 未正确训练，只有特殊 tokens。"
        echo "需要重新训练 tokenizer 才能用于实际训练。"
        echo ""
        echo "请运行以下命令重新训练 tokenizer:"
        echo ""
        echo "  python -m examples.nanochat_exp.tok_train --force --data-dir $OPENSEEK_NANOCHAT_DATA_DIR/parquet_shards"
        echo ""
        read -p "是否现在重新训练 tokenizer？(y/N): " -n 1 -r
        echo
        FORCE_RETRAIN=true
    else
        echo "nanochat 需要 tokenizer 文件才能运行训练。"
        echo "请运行以下命令生成 tokenizer（使用 HuggingFace tokenizers，无需 rustbpe）:"
        echo ""
        echo "  python -m examples.nanochat_exp.tok_train --data-dir $OPENSEEK_NANOCHAT_DATA_DIR/parquet_shards"
        echo ""
        echo "或者如果您有现有的 tokenizer，请将其放在:"
        echo "  $TOKENIZER_DIR/tokenizer.pkl 或 $TOKENIZER_DIR/tokenizer.json"
        echo ""
        read -p "是否现在运行 tok_train.py 生成 tokenizer？(y/N): " -n 1 -r
        echo
        FORCE_RETRAIN=false
    fi
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # 使用 OpenSeek 的 tok_train.py（使用 HuggingFace tokenizers，无需 rustbpe）
        if [ -f "$OPENSEEK_ROOT/examples/nanochat_exp/tok_train.py" ]; then
            # 检查 tokenizer 库是否可用（优先使用 HuggingFace tokenizers）
            echo "检查 tokenizer 库..."
            TOKENIZER_OK=false
            if python3 -c "from tokenizers import Tokenizer" 2>/dev/null; then
                TOKENIZER_OK=true
                echo "✓ HuggingFace tokenizers 库可用（推荐，无需 Rust 编译）"
            elif python3 -c "import rustbpe; hasattr(rustbpe, 'Tokenizer')" 2>/dev/null; then
                TOKENIZER_OK=true
                echo "✓ rustbpe 模块可用（备用方案）"
            else
                echo "✗ 未找到可用的 tokenizer 库"
                echo ""
                echo "OpenSeek 推荐使用 HuggingFace tokenizers（更容易安装，无需 Rust 编译）。"
                echo "如果 nanochat 的 tok_train.py 需要 rustbpe，您也可以选择安装 rustbpe。"
                echo ""
                read -p "是否安装 HuggingFace tokenizers 库（推荐）？(Y/n): " -n 1 -r
                echo
                if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                    echo "安装 tokenizers 库..."
                    if pip install tokenizers>=0.22.0; then
                        TOKENIZER_OK=true
                        echo "✓ tokenizers 库安装完成"
                    else
                        echo "警告: tokenizers 库安装失败"
                    fi
                fi
                
                # 如果 tokenizers 不可用，检查是否需要 rustbpe（用于 nanochat 的 tok_train.py）
                if [ "$TOKENIZER_OK" != true ]; then
                    echo ""
                    echo "注意: nanochat 的 tok_train.py 可能需要 rustbpe。"
                    echo "如果您需要使用 tok_train.py，请安装 rustbpe（需要 Rust 编译）。"
                    echo "否则，OpenSeek 的数据加载器可以使用 tokenizers 库。"
                    read -p "是否继续尝试运行 tok_train.py（可能需要 rustbpe）？(y/N): " -n 1 -r
                    echo
                    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                        echo "跳过 tokenizer 生成。"
                        echo "提示: OpenSeek 的数据加载器可以使用现有的 tokenizer.json 文件，"
                        echo "      无需运行 tok_train.py。"
                        exit 0
                    fi
                fi
            fi
            
            if [ "$TOKENIZER_OK" = true ]; then
                echo ""
                echo "使用 OpenSeek 的 tok_train.py 生成 tokenizer（使用 HuggingFace tokenizers，无需 rustbpe）..."
                cd "$OPENSEEK_ROOT"
                # 确保 PYTHONPATH 包含 OpenSeek
                export PYTHONPATH="$OPENSEEK_ROOT:${PYTHONPATH}"
                # 确保 NANOCHAT_BASE_DIR 已设置
                export NANOCHAT_BASE_DIR="$OPENSEEK_NANOCHAT_DATA_DIR"
                
                # 检查数据文件是否存在
                if [ ! -d "$OPENSEEK_NANOCHAT_DATA_DIR/parquet_shards" ] || [ -z "$(ls -A $OPENSEEK_NANOCHAT_DATA_DIR/parquet_shards/*.parquet 2>/dev/null)" ]; then
                    echo "错误: 未找到 parquet 数据文件"
                    echo "请先运行数据转换步骤"
                    echo "数据目录: $OPENSEEK_NANOCHAT_DATA_DIR/parquet_shards"
                    exit 1
                fi
                
                echo "使用数据目录: $OPENSEEK_NANOCHAT_DATA_DIR"
                FORCE_FLAG=""
                if [ "$FORCE_RETRAIN" = true ]; then
                    FORCE_FLAG="--force"
                    echo "使用 --force 选项强制重新训练 tokenizer"
                fi
                python -m examples.nanochat_exp.tok_train $FORCE_FLAG --data-dir "$OPENSEEK_NANOCHAT_DATA_DIR/parquet_shards" || {
                    TOKENIZER_EXIT_CODE=$?
                    echo ""
                    echo "错误: 生成 tokenizer 失败（退出码: $TOKENIZER_EXIT_CODE）"
                    echo ""
                    echo "可能的原因:"
                    echo "1. tokenizers 库未正确安装"
                    echo "2. 数据文件不存在或路径不正确"
                    echo "3. 内存不足（生成 tokenizer 需要大量内存）"
                    echo ""
                    echo "请检查:"
                    echo "1. 确保已安装 tokenizers: pip install tokenizers>=0.22.0"
                    echo "2. 确保数据文件存在: ls $OPENSEEK_NANOCHAT_DATA_DIR/parquet_shards/*.parquet"
                    echo "3. 检查可用内存: free -h"
                    echo "4. 手动运行: python -m examples.nanochat_exp.tok_train --data-dir $OPENSEEK_NANOCHAT_DATA_DIR/parquet_shards"
                    exit 1
                }
                echo ""
                echo "✓ Tokenizer 生成完成（使用 HuggingFace tokenizers，无需 rustbpe）"
            else
                echo "错误: tokenizers 库不可用，无法生成 tokenizer"
                echo "请安装 tokenizers 库: pip install tokenizers>=0.22.0"
                exit 1
            fi
        else
            echo "错误: 找不到 tok_train.py 脚本"
            exit 1
        fi
    else
        echo "跳过 tokenizer 生成。请手动生成 tokenizer 后再运行训练。"
        exit 1
    fi
else
    echo "✓ Tokenizer 文件已找到: $TOKENIZER_DIR"
fi
echo

# Number of GPUs to use
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

# Model depth (d4 = small test model, d20 = ~$100 tier, d26 = ~$300 tier, d32 = ~$800 tier)
DEPTH="${DEPTH:-4}"

# Wandb run name (optional)
WANDB_RUN="${WANDB_RUN:-openseek_d${DEPTH}}"
WANDB_PROJECT="${WANDB_PROJECT:-openseek-nanochat}"

echo "Configuration:"
echo "  Model depth: $DEPTH"
echo "  GPUs: $NPROC_PER_NODE"
echo "  Wandb project: $WANDB_PROJECT"
echo "  Wandb run: $WANDB_RUN"
echo "  注意: nanochat 使用默认的 loss 输出频率（通常在训练循环中每步输出）"
echo

# Check if nanochat is available and patch if needed
if [ -n "$NANOCHAT_DIR" ] && [ -f "$NANOCHAT_DIR/scripts/base_train.py" ]; then
    echo "检测到 nanochat 目录: $NANOCHAT_DIR"
    
    # Check if already patched (check both dataloader and tokenizer)
    BASE_TRAIN_FILE="$NANOCHAT_DIR/scripts/base_train.py"
    if [ -f "$BASE_TRAIN_FILE" ]; then
        if grep -q "from examples.nanochat_exp.dataloader import" "$BASE_TRAIN_FILE" 2>/dev/null && \
           grep -q "from examples.nanochat_exp.tokenizer import" "$BASE_TRAIN_FILE" 2>/dev/null; then
            echo "✓ nanochat base_train.py 已经修改过（dataloader 和 tokenizer），跳过"
        elif grep -q "from examples.nanochat_exp.dataloader import" "$BASE_TRAIN_FILE" 2>/dev/null; then
            echo "检测到 dataloader 已修改，但 tokenizer 未修改，需要更新 tokenizer 导入..."
            echo "自动修改 nanochat base_train.py 以使用 OpenSeek tokenizer..."
            if python3 -m examples.nanochat_exp.patch_nanochat --nanochat-path "$NANOCHAT_DIR" 2>&1; then
                echo "✓ 成功更新 tokenizer 导入"
            else
                echo "警告: 自动修改失败，请手动修改:"
                echo "  在 $BASE_TRAIN_FILE 中，将:"
                echo "    from nanochat.tokenizer import get_tokenizer"
                echo "  改为:"
                echo "    from examples.nanochat_exp.tokenizer import get_tokenizer"
            fi
        else
            echo "自动修改 nanochat base_train.py 以使用 OpenSeek dataloader 和 tokenizer..."
            if python3 -m examples.nanochat_exp.patch_nanochat --nanochat-path "$NANOCHAT_DIR" 2>&1; then
                echo "✓ 成功修改 nanochat base_train.py"
            else
                echo "警告: 自动修改失败，请手动修改:"
                echo "  在 $BASE_TRAIN_FILE 中，将:"
                echo "    from nanochat.dataloader import tokenizing_distributed_data_loader, tokenizing_distributed_data_loader_with_state"
                echo "    from nanochat.tokenizer import get_tokenizer"
                echo "  改为:"
                echo "    from examples.nanochat_exp.dataloader import tokenizing_distributed_data_loader, tokenizing_distributed_data_loader_with_state"
                echo "    from examples.nanochat_exp.tokenizer import get_tokenizer"
            fi
        fi
    else
        echo "警告: 找不到 $BASE_TRAIN_FILE"
        echo "请确保 nanochat 目录正确: $NANOCHAT_DIR"
    fi
    
    echo
    echo "准备开始训练..."
    
    # 检测 torchrun 是否可用
    TORCHRUN_CMD=""
    if command -v torchrun &> /dev/null; then
        TORCHRUN_CMD="torchrun"
    elif python3 -m torch.distributed.run --help &> /dev/null; then
        TORCHRUN_CMD="python3 -m torch.distributed.run"
    else
        echo "警告: 未找到 torchrun 命令，且 PyTorch 似乎未安装"
        echo "请确保:"
        echo "  1. 已安装 PyTorch: pip install torch"
        echo "  2. 或者激活包含 PyTorch 的虚拟环境"
        echo "  3. 或者 torchrun 在 PATH 中"
        echo
        echo "训练命令（需要先安装 PyTorch）:"
        echo "  cd $NANOCHAT_DIR"
        echo "  torchrun --standalone --nproc_per_node=$NPROC_PER_NODE \\"
        echo "    -m scripts.base_train -- --depth=$DEPTH --run=$WANDB_RUN"
        echo
        echo "或者使用 Python 模块方式:"
        echo "  cd $NANOCHAT_DIR"
        echo "  python3 -m torch.distributed.run --standalone --nproc_per_node=$NPROC_PER_NODE \\"
        echo "    -m scripts.base_train -- --depth=$DEPTH --run=$WANDB_RUN"
        echo
        read -p "是否现在开始训练？(y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "错误: 无法启动训练，请先安装 PyTorch"
            exit 1
        else
            echo "跳过训练。请先安装 PyTorch 后再运行训练。"
            exit 0
        fi
    fi
    
    echo "训练命令:"
    echo "  cd $NANOCHAT_DIR"
    echo "  $TORCHRUN_CMD --standalone --nproc_per_node=$NPROC_PER_NODE \\"
    echo "    -m scripts.base_train -- --depth=$DEPTH --run=$WANDB_RUN"
    echo
    read -p "是否现在开始训练？(y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "开始训练..."
        cd "$NANOCHAT_DIR"
        
        # 设置环境变量以显示详细错误信息
        export TORCHELASTIC_ERROR_FILE="/tmp/torchelastic_error.log"
        export TORCHELASTIC_ERROR_RANGE="0-255"
        
        # 确保 PYTHONPATH 包含 OpenSeek 路径（在 cd 之后仍然有效）
        export PYTHONPATH="$OPENSEEK_ROOT:$NANOCHAT_DIR:${PYTHONPATH}"
        echo "PYTHONPATH: $PYTHONPATH"
        
        # 先测试导入，以便获得更清晰的错误信息
        echo "测试导入..."
        if ! python3 -c "
import sys
import os
# 确保 OpenSeek 路径在 sys.path 中
openseek_root = '$OPENSEEK_ROOT'
if openseek_root not in sys.path:
    sys.path.insert(0, openseek_root)
print(f'OpenSeek root in path: {openseek_root in sys.path}')
print(f'Python path: {sys.path[:3]}...')

try:
    from examples.nanochat_exp.dataloader import tokenizing_distributed_data_loader, tokenizing_distributed_data_loader_with_state
    print('✓ dataloader 导入成功')
except Exception as e:
    print(f'✗ dataloader 导入失败: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    import scripts.base_train
    print('✓ base_train 导入成功')
except Exception as e:
    print(f'✗ base_train 导入失败: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
" 2>&1; then
            echo "导入测试通过，开始训练..."
            echo ""
            echo "=========================================="
            echo "开始训练（带 Loss 监控）"
            echo "=========================================="
            echo ""
            echo "提示: 训练监控脚本会自动提取并高亮显示 Loss 值"
            echo "      所有训练输出都会正常显示，Loss 会被额外格式化显示"
            echo ""
            
            # 可选：运行诊断（如果设置了 RUN_DIAGNOSE 环境变量）
            if [ "${RUN_DIAGNOSE:-false}" = "true" ]; then
                echo "运行训练前诊断..."
                python3 -m examples.nanochat_exp.diagnose_training || {
                    echo "⚠️  诊断发现问题，但继续训练..."
                }
                echo ""
            fi
            
            echo "如果训练卡住，可以："
            echo "  1. 运行诊断脚本: python -m examples.nanochat_exp.diagnose_training"
            echo "  2. 检查数据文件: ls -lh $OPENSEEK_NANOCHAT_DATA_DIR/parquet_shards/ | head"
            echo "  3. 尝试单 GPU 训练: NPROC_PER_NODE=1 bash run_openseek_exp.sh"
            echo "  4. 查看 GPU 状态: nvidia-smi"
            echo "  5. 检查进程: ps aux | grep python"
            echo ""
            
            # 设置 wandb 环境变量
            export WANDB_RUN="$WANDB_RUN"
            export WANDB_PROJECT="$WANDB_PROJECT"
            
            # 运行训练，捕获错误，并使用训练监控脚本格式化输出
            # 确保 PYTHONPATH 在 torchrun 命令中可用
            # 使用 unbuffered Python 输出以确保实时显示
            if env PYTHONPATH="$PYTHONPATH" PYTHONUNBUFFERED=1 WANDB_RUN="$WANDB_RUN" WANDB_PROJECT="$WANDB_PROJECT" \
                $TORCHRUN_CMD --standalone --nproc_per_node=$NPROC_PER_NODE \
                -m scripts.base_train -- --depth=$DEPTH --run=$WANDB_RUN 2>&1 | \
                python3 -u -m examples.nanochat_exp.training_monitor; then
                echo ""
                echo "=========================================="
                echo "训练完成"
                echo "=========================================="
            else
                TRAIN_EXIT_CODE=$?
                echo ""
                echo "=========================================="
                echo "训练失败（退出码: $TRAIN_EXIT_CODE）"
                echo "=========================================="
                echo ""
                echo "可能的解决方案:"
                echo "1. 运行诊断脚本检查问题:"
                echo "   python -m examples.nanochat_exp.diagnose_training"
                echo ""
                echo "2. 检查依赖项是否完整安装"
                echo "3. 检查数据文件是否存在: $OPENSEEK_NANOCHAT_DATA_DIR/parquet_shards/"
                echo "4. 检查 GPU 是否可用: nvidia-smi"
                echo "5. 查看详细错误日志（如果存在）: $TORCHELASTIC_ERROR_FILE"
                echo ""
                echo "6. 如果训练卡住（无输出），可能的原因："
                echo "   - 数据加载慢（首次加载需要时间）"
                echo "   - 分布式训练同步问题（多 GPU 时）"
                echo "   - Wandb 初始化慢（网络问题）"
                echo "   建议："
                echo "     - 尝试单 GPU: NPROC_PER_NODE=1 bash run_openseek_exp.sh"
                echo "     - 禁用 wandb: export WANDB_MODE=disabled"
                echo "     - 检查数据: ls -lh $OPENSEEK_NANOCHAT_DATA_DIR/parquet_shards/ | head"
                echo ""
                echo "可以尝试直接运行 Python 脚本查看详细错误:"
                echo "  cd $NANOCHAT_DIR"
                echo "  python3 -m scripts.base_train -- --depth=$DEPTH --run=$WANDB_RUN"
                exit $TRAIN_EXIT_CODE
            fi
        else
            echo "导入测试失败，请检查错误信息 above"
            exit 1
        fi
    else
        echo "跳过训练。您可以稍后手动运行训练命令。"
    fi
else
    echo "未找到 nanochat 目录，无法自动启动训练"
    echo
    echo "要开始训练，请:"
    echo "1. 确保 nanochat 在 PYTHONPATH 中"
    echo "2. 修改 nanochat/scripts/base_train.py，将:"
    echo "     from nanochat.dataloader import tokenizing_distributed_data_loader, tokenizing_distributed_data_loader_with_state"
    echo "   改为:"
    echo "     from examples.nanochat_exp.dataloader import tokenizing_distributed_data_loader, tokenizing_distributed_data_loader_with_state"
    echo "3. 运行训练命令:"
    echo "   export OPENSEEK_NANOCHAT_DATA_DIR=\"$OPENSEEK_NANOCHAT_DATA_DIR\""
    echo "   export NANOCHAT_BASE_DIR=\"$OPENSEEK_NANOCHAT_DATA_DIR\""
    echo "   cd <nanochat_directory>"
    echo ""
    echo "   如果 torchrun 可用:"
    echo "     torchrun --standalone --nproc_per_node=$NPROC_PER_NODE \\"
    echo "       -m scripts.base_train -- --depth=$DEPTH --run=$WANDB_RUN"
    echo ""
    echo "   或者使用 Python 模块方式:"
    echo "     python3 -m torch.distributed.run --standalone --nproc_per_node=$NPROC_PER_NODE \\"
    echo "       -m scripts.base_train -- --depth=$DEPTH --run=$WANDB_RUN"
fi

