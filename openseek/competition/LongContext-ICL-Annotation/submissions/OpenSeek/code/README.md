# AI-LAB OpenSeek Submission

面向 OpenSeek `LongContext-ICL-Annotation` 赛题的可复现参赛工程。

本项目基于 `Qwen3-4B` 构建统一的 8 任务自动标注流程，核心链路包括：

- 长上下文示例检索与压缩
- `front-back` 证据重排
- 多协议首轮推理
- 一致性与置信度判断
- 低置信样本复核
- 提交校验与统一打包

## 环境要求

- Python 3.10+
- 可用的 Qwen3-4B 权重目录
- 官方比赛数据目录
- 若采用正式部署路径，使用 FlagScale 作为模型加载与服务框架

安装依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 目录说明

```text
AI-LAB/
├── configs/         运行配置
├── data/            数据目录约定与缓存目录
├── prompts/         各任务提示词协议
├── scripts/         运行、校验与打包脚本
├── src/             主流程实现
├── tests/           基础回归测试
└── requirements.txt 环境依赖
```

## 数据准备

将官方数据导入到 `data/raw/openseek/`：

```bash
./scripts/import_official_data.sh /path/to/LongContext-ICL-Annotation/data
```

## 快速运行

本地 smoke：

```bash
./scripts/run_infer.sh configs/local_smoke.yaml
./scripts/run_eval.sh configs/local_smoke.yaml
./scripts/package_submission.sh configs/local_smoke.yaml
```

标准运行：

```bash
./scripts/run_infer.sh configs/base.yaml
./scripts/run_eval.sh configs/base.yaml
./scripts/package_submission.sh configs/base.yaml
```

## FlagScale 部署路径

若采用 FlagScale 服务化推理，先按实际模型路径修改
`configs/flagscale_serve.template.yaml`，然后启动服务：

```bash
flagscale run -p configs -n flagscale_serve.template -a run \
  serve.0.engine_args.model=/path/to/Qwen3-4B \
  serve.0.engine_args.port=2026 \
  +serve.0.engine_args.max_model_len=32768 \
  +serve.0.engine_args.max_num_seqs=1 \
  serve.0.engine_args.gpu_memory_utilization=0.88
```

随后运行正式配置：

```bash
./scripts/run_infer.sh configs/openbayes_flagscale_full.yaml
./scripts/run_eval.sh configs/openbayes_flagscale_full.yaml
./scripts/package_submission.sh configs/openbayes_flagscale_full.yaml
```

## 结果校验

任务 8 静态检查：

```bash
python3 scripts/check_task8_predictions.py outputs/predictions/openseek-8-v1.jsonl
```

任务 8 运行代理检查：

```bash
python3 scripts/check_task8_runtime.py outputs/predictions/openseek-8-v1.jsonl
```

完整发布预检：

```bash
make release-check
```

## 关键文件

- 主流程：`src/ai_lab/pipeline.py`
- 官方数据读取：`src/ai_lab/adapters/official_reader.py`
- 检索与重排：`src/ai_lab/retrieval/`
- 决策与投票：`src/ai_lab/decision/`
- 输出解析：`src/ai_lab/output_parser.py`
- 提交校验：`src/ai_lab/submit/validate_submission.py`

## 当前交付物

在本次 OpenSeek 官方仓库提交中，对应交付物位于上一级目录：

- 最终预测包：`../submission.zip`
- 最终代码包：`../源代码-OpenSeek.zip`
- 正式技术报告 PDF：`../技术报告-OpenSeek.pdf`
