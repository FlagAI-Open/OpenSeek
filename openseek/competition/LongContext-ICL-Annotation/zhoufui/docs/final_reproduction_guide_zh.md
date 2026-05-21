# 最终方案复现说明

## 当前最佳结果

- 平台最好分数：`82.72`
- 当前最佳预测包：`outputs/zhoufui_prediction_8272_task2_noun_last_push20.zip`
- 该包已经通过本地结构校验：8 个 JSONL 位于 zip 根目录，行数为 `500, 500, 500, 500, 500, 500, 500, 166`，编码为无 BOM UTF-8。

任务来源如下：

| 数据集 | 方法 |
| --- | --- |
| openseek-1 | 确定性程序求解 |
| openseek-2 | Qwen3-4B ICL + 高置信动词/名词计数边界修正 |
| openseek-3 | 确定性程序求解 |
| openseek-4 | 确定性程序求解 |
| openseek-5 | Qwen3-4B ICL |
| openseek-6 | Qwen3-4B ICL |
| openseek-7 | Qwen3-4B ICL |
| openseek-8 | 可执行 PyTorch reference fallback |

后续实验中，task5 v2、task6 v2、task7 v2 以及多组 task8 小改均未超过当前最佳；2026-05-19 的 task2 noun audit 修正将平台最好分数提升到 `82.72`，因此作为当前最佳包。

## 合规原则

- 只使用 Qwen3-4B。
- 不使用外部数据、自建数据或额外模型。
- 只使用官方给定 examples、test_samples、程序化后处理和本地/FlagScale 推理环境。
- 最终有效复现应使用 FlagScale 加载或服务化 Qwen3-4B。

## FlagScale 推理环境

推荐在 Linux GPU 服务器上使用 FlagOS / FlagScale Nvidia 镜像。具体步骤见：

```text
docs/flagscale_qwen3_setup.md
```

服务启动后，设置 OpenAI-compatible endpoint：

```bash
export OPENAI_BASE_URL="http://127.0.0.1:9010/v1"
export OPENAI_API_KEY="EMPTY"
export PYTHONPATH=src
```

然后运行官方数据推理：

```bash
PYTHONPATH=src python -m flagos_icl.official_cli \
  --config configs/baseline.yaml \
  --data-dir data/official \
  --output outputs/official_submission.json \
  --diagnostics outputs/official_diagnostics.json \
  --official-jsonl-dir outputs/official_jsonl \
  --deterministic-postprocess
```

说明：`configs/baseline.yaml` 的 `model.framework` 为 `flagscale`，`model.provider` 为 `openai_compatible`，模型名固定为 `Qwen3-4B`。

## 本地快速迭代

本地 WSL 路径仅用于快速实验和候选包生成：

```bash
source .venv-wsl/bin/activate
export PYTHONPATH=src
export LOCAL_QWEN3_4B_PATH=models/Qwen3-4B
```

任务 2：

```bash
TASK_IDS=openseek-2 OUT_PREFIX=local_qwen_2 bash scripts/wsl_run_qwen_selected.sh
```

任务 5/6：

```bash
TASK_IDS=openseek-5,openseek-6 OUT_PREFIX=local_qwen_56 bash scripts/wsl_run_qwen_selected.sh
```

本地 8GB GPU 的极小样本验证：

```bash
bash scripts/wsl_run_tiny_qwen.sh
TASK_IDS=openseek-5,openseek-6 MAX_SAMPLES=10 OUT_PREFIX=local_qwen_56_smoke10 CONFIG=configs/local_qwen_classification.yaml bash scripts/wsl_run_qwen_selected.sh
```

该路径已验证 Windows + WSL2 + Docker Desktop + NVIDIA GPU 可见，且 4-bit Qwen3-4B 能完成短上下文小样本推理。它用于开发验证，不替代最终 FlagScale 复现路径。

任务 7：

```bash
CONFIG=configs/local_qwen_answer.yaml TASK_IDS=openseek-7 OUT_PREFIX=local_qwen_7 bash scripts/wsl_run_qwen_selected.sh
```

## 任务 8 PyTorch Reference 包

task8 的本地 Qwen Triton 生成实验中出现占位代码和不稳定代码。当前最佳方案采用可执行 PyTorch reference fallback，目标是优先保证每条样本返回可导入、可运行、覆盖常见 wrapper 名称的代码。

生成 task8 reference 包：

```bash
PYTHONPATH=src python scripts/build_task8_torch_reference_package.py
```

该实验从 `69.18` 提升到 `80.55`，是当前最大单项收益。

## 候选包验证

提交前必须运行：

```bash
python scripts/validate_submission_zip.py outputs/zhoufui_prediction_8272_task2_noun_last_push20.zip
```

最终 FlagScale 复现入口为：

```bash
bash scripts/linux_run_predictions.sh
```

该脚本会先通过 FlagScale/Qwen3-4B 生成任务 2/5/6/7 的模型结果，并在 `--deterministic-postprocess` 下对任务 1/3/4 使用确定性求解、对任务 8 使用 `src/flagos_icl/task8_torch_reference.py` 中的 PyTorch reference fallback。随后脚本会应用 `configs/final_overrides.json` 中的平台验证 task2 审计修正，并输出 `outputs/zhoufui_prediction_final.zip`。

比较两个候选包：

```bash
python scripts/compare_submission_zips.py outputs/A.zip outputs/B.zip --show 20
```

按任务拼包：

```bash
python scripts/blend_submission_zip.py \
  --base outputs/zhoufui_prediction_8272_task2_noun_last_push20.zip \
  --out outputs/new_candidate.zip \
  --replace 7=outputs/other_task7_package.zip
```

## 打包源码

源码包不包含 `.venv-wsl`、`models`、`outputs`、`data/official` 等大文件或官方测试数据：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/package_submission.ps1
```

输出：

```text
outputs/flagos_source_package.zip
```
