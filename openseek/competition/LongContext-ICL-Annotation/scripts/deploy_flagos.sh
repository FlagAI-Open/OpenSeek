#!/usr/bin/env bash
# 使用 FlagScale 启动本地 FlagOS OpenAI 兼容服务（默认端口 9010）
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FLAGSCALE_DIR="${FLAGSCALE_DIR:-$REPO_ROOT/FlagScale}"
CONFIG_NAME="${CONFIG_NAME:-llm_config_flagos}"
ACTION="${1:-run}"

if [[ ! -d "$FLAGSCALE_DIR" ]]; then
  echo "[deploy_flagos] 未找到 FlagScale 目录: $FLAGSCALE_DIR"
  echo "请先克隆: git clone https://github.com/FlagOpen/FlagScale.git \"$FLAGSCALE_DIR\""
  echo "并按 FlagScale 文档完成 Ascend/NVIDIA 环境安装（见 readme.md）。"
  exit 1
fi

if [[ ! -f "$REPO_ROOT/configs/$CONFIG_NAME.yaml" ]]; then
  echo "[deploy_flagos] 缺少配置: $REPO_ROOT/configs/$CONFIG_NAME.yaml"
  exit 1
fi

# 将配置链接到 FlagScale 上级目录，与官方 run.py --config-path .. 用法一致
ln -sf "$REPO_ROOT/configs/$CONFIG_NAME.yaml" "$FLAGSCALE_DIR/../$CONFIG_NAME.yaml"

cd "$FLAGSCALE_DIR"
echo "[deploy_flagos] cwd=$PWD action=$ACTION config=../$CONFIG_NAME.yaml"
python run.py --config-path .. --config-name "$CONFIG_NAME" "action=$ACTION"
