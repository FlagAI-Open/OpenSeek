#!/bin/bash
# =========================================================
# FlagOS OpenSeek 赛道三 - TOPGO团队
# 一键运行脚本
# =========================================================

set -e

echo "========================================"
echo "FlagOS OpenSeek 赛道三 - TOPGO团队"
echo "========================================"

# 步骤0: 设置国内镜像（解决HuggingFace连接问题）
export HF_ENDPOINT=https://hf-mirror.com

# 步骤1: 安装依赖
echo "[1/4] 安装依赖..."
pip install openai tqdm requests -q

# 步骤2: 配置API
echo "[2/4] 配置API..."
if [ -z "$QWEN_API_BASE" ]; then
    export QWEN_API_BASE="http://localhost:8000/v1/"
    echo "使用默认API端点: $QWEN_API_BASE"
fi
if [ -z "$QWEN_API_KEY" ]; then
    export QWEN_API_KEY="EMPTY"
fi

# 步骤3: 创建输出目录
echo "[3/4] 创建输出目录..."
mkdir -p outputs

# 步骤4: 运行任务
echo "[4/4] 运行标注任务..."

cd src

# 询问运行哪些任务
echo ""
echo "请选择要运行的任务:"
echo "  1-8: 运行单个任务"
echo "  all: 运行所有任务"
read -p "请输入选项: " choice

if [ "$choice" = "all" ]; then
    # 运行所有任务
    for i in {1..8}; do
        echo ""
        echo "运行任务 $i..."
        python main.py --task_id $i --max_input_length 10000 --log_path_prefix ../outputs/ --device ascend
    done
elif [[ "$choice" -ge 1 && "$choice" -le 8 ]]; then
    # 运行单个任务
    echo ""
    echo "运行任务 $choice..."
    python main.py --task_id $choice --max_input_length 10000 --log_path_prefix ../outputs/ --device ascend
else
    echo "无效选项: $choice"
    exit 1
fi

cd ..

# 查看结果
echo ""
echo "========================================"
echo "运行完成！"
echo "========================================"
echo ""
echo "输出文件:"
ls -lh outputs/*.jsonl 2>/dev/null || echo "暂无输出文件"

echo ""
echo "如需打包下载:"
echo "  zip -r results.zip outputs/"
