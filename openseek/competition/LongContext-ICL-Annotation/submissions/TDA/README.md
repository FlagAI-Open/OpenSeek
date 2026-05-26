# LongContext-ICL-Annotation 代码提交包

本目录为 FlagOS/OpenSeek「长上下文场景中大模型自动数据标注」赛道的代码提交包，对应最终提交 `submission_0520v11.zip`，榜单分数 81.87。

## 目录结构

- `src/`：方案代码、FlagScale 推理配置、推理与优化脚本。
- `data/`：组委会提供的 8 个官方数据集文件。
- `outputs/final_submission/`：最终预测结果、最终提交 zip、分数摘要与运行摘要。
- `scripts/`：提交包校验与重新打包脚本。
- `docs/技术报告-TDA.pdf`：最终技术报告。
- `requirements.txt`：Python 依赖说明。

本包不包含 Qwen3-4B 模型权重。请按赛题说明从官方 HuggingFace/ModelScope/Modelers 链接下载模型，并放置到本目录同级或按实际路径修改 `src/llm_config.yaml`。

## 环境准备

建议使用 Python 3.10，并根据本机 CUDA 环境安装 PyTorch、FlagScale 与 vLLM。随后安装本包依赖：

```bash
pip install -r requirements.txt
```

## 启动 Qwen3-4B 推理服务

使用 FlagScale 作为模型加载与运行入口。示例命令如下，模型路径可按实际环境调整：

```bash
cd /path/to/FlagScale
python run.py --config-path /path/to/LongContext-ICL-Annotation-code-20260520-final/src \
  --config-name llm_config action=run \
  serve.0.engine_args.model=/path/to/Qwen3-4B
```

服务默认监听 `http://127.0.0.1:2026/v1/completions`，服务名为 `../Qwen3-4B`。

## 运行与复现

基础推理入口示例：

```bash
python src/main.py --task_id 5 --tokenizer_path /path/to/Qwen3-4B --log_path_prefix outputs/
```

最终方案中的规则校验、任务特化优化与提交包生成逻辑主要位于 `src/competition_optimize.py` 及相关系统化分析脚本中。最终提交结果已保存在 `outputs/final_submission/submission_0520v11.zip`。

## 校验最终提交

```bash
python scripts/validate_submission.py outputs/final_submission/submission_0520v11.zip
```

若需要从 8 个 JSONL 重新打包：

```bash
python scripts/make_submission_zip.py
python scripts/validate_submission.py outputs/final_submission/submission_repacked.zip
```

## 合规说明

本方案仅使用赛题指定的 Qwen3-4B 模型、FlagScale 推理框架和组委会提供的官方数据集；未使用外部模型、外部数据、自建数据或模型微调。最终预测覆盖全部 8 个任务，共 3666 条测试样本。
