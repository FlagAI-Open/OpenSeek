# FlagOS Trailblazers 推理提交包

本目录是推理复现包，只保留运行推理所需代码、模型服务脚本、环境依赖、官方输入数据副本和相关 Markdown 说明文档。

## 目录结构

```text
.
├── src/                         # Task1-Task8 推理、路由、候选生成与打包辅助代码
├── FlagScale/                   # Qwen3-4B 服务启动所需 FlagScale 代码
├── configs/llm_config.yaml      # FlagScale 服务配置
├── scripts/
│   ├── start_service.sh         # 启动本地 OpenAI-compatible 服务
│   ├── stop_service.sh          # 停止服务
│   └── run_task8_pytorch_passk.sh
├── input/                       # 官方输入数据副本
├── docs/
│   ├── 代码方案与复现步骤.md
│   └── 合规自查.md
├── requirements.txt
├── run_full_inference.sh        # 全量推理入口
├── make_submission.py           # 生成提交 zip
└── verify_submission.py         # 校验提交格式
```

## 环境准备

建议使用 Linux/Ascend NPU 环境和 Python 3.11。模型权重不放在本目录中，默认读取 `models/Qwen3-4B`，也可以显式指定。

```bash
cd /path/to/competition_official_submission_package_20260520
pip install -r requirements.txt

export FLAGSCALE_MODEL_PATH=/path/to/models/Qwen3-4B
export API_BASE=http://127.0.0.1:2026
export PYTHON_BIN=python3
```

## 启动模型服务

```bash
bash scripts/start_service.sh
curl http://127.0.0.1:2026/v1/models
```

停止服务：

```bash
bash scripts/stop_service.sh
```

## 全量推理

```bash
STAMP=official_full_$(date +%Y%m%d_%H%M%S) \
FRESH_RUN=1 \
CLIENT_CONCURRENCY=2 \
ROUTER_MAX_WORKERS=4 \
T7_MAX_WORKERS=4 \
T8_MAX_WORKERS=16 \
bash run_full_inference.sh
```

输出目录：

```text
outputs/${STAMP}/
outputs/upload_ready_${STAMP}/
outputs/upload_ready_${STAMP}.zip
```

`outputs/upload_ready_${STAMP}.zip` 可直接上传平台。

## 快速自检

```bash
STAMP=smoke20_$(date +%Y%m%d_%H%M%S) \
FRESH_RUN=1 \
MAX_TEST_SAMPLES=20 \
TASK8_SAMPLE_COUNT=20 \
T7_BANK_SAMPLES=100 \
T7_VAL_SAMPLES=20 \
T7_GRID_PRESET=probe \
T7_TEST_CONFIGS=1 \
T7_MAIN_CONFIGS=1 \
T8_REPEATS=4 \
T8_REPAIR_ROUNDS=0 \
T8_HARD_REPAIR_ROUNDS=0 \
CLIENT_CONCURRENCY=1 \
ROUTER_MAX_WORKERS=2 \
T7_MAX_WORKERS=2 \
T8_MAX_WORKERS=2 \
bash run_full_inference.sh
```

## 默认分支

```bash
TASK2_BRANCH=legacy_router
```

Task1/3/4 默认仍为 solver debug loop：

```bash
TASK134_BRANCH=solver_debug_loop
```

如需运行迁入的 Task1/3/4 长上下文 many-shot 分支：

```bash
TASK134_BRANCH=longctx_manyshot bash run_full_inference.sh
```

## 提交校验

```bash
python verify_submission.py \
  --submission-dir outputs/upload_ready_${STAMP} \
  --zip-path outputs/upload_ready_${STAMP}.zip
```

校验内容包括 8 个 JSONL 文件名、官方固定行数、字段格式、Task8 `code`/`prediction` 一致性，以及 zip 内部结构。
