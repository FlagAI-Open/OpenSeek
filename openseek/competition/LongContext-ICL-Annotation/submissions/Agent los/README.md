# OpenSeek 长上下文自动标注方案

队名：不想熬夜等结果

本目录包含 FlagOS / OpenSeek 赛道三“长上下文场景中大模型自动数据标注”的预测结果、推理代码与技术报告。

## 目录说明

- `code/main.py`：通用推理入口，支持按任务 ID 运行、并发推理和断点续传。
- `code/method.py`：通用 prompt 构造、示例选择、模型请求与答案解析。
- `code/api_test.py`：本地模型服务连通性测试。
- `code/debug_*.py`：任务专用增强脚本，用于更稳定地生成任务 2 至任务 8 的结果。
- `openseek-*-v1.jsonl`：8 个测试集预测结果。
- `submission_Jin_Final_Strict.zip`：预测结果提交压缩包。
- `技术报告-不想熬夜等结果.pdf`：技术报告。

## 环境准备

官方运行环境建议使用 Ascend 910C，并通过 FlagScale 加载 Qwen3-4B。推理脚本默认访问本地 OpenAI-compatible Chat Completions 服务：

```bash
http://127.0.0.1:8000/v1/chat/completions
```

安装 Python 依赖：

```bash
pip install -r requirements.txt
```

## 连通性测试

启动模型服务后执行：

```bash
cd code
python api_test.py
```

如果接口正常，会返回 Qwen3-4B 的回复和 token 用量信息。

## 推理运行

通用入口示例：

```bash
cd code
python main.py --task_id 1 --max_input_length 32000 --tokenizer_path /root/OpenSeek/openseek/competition/Qwen3-4B --workers 4
```

任务专用脚本示例：

```bash
python debug_two.py
python debug_three.py
python debug_four.py
python debug_five.py
python debug_six.py
python debug_seven.py
python debug_seven2.py
python debug_eight.py
python debug_eight2.py
```

各脚本会读取官方数据目录中的任务文件，并将结果写入 `../outputs/`。脚本带有断点续传逻辑，重复运行时会跳过已完成样本。

## 结果格式

每个 jsonl 文件每行一个 JSON 对象：

```json
{"test_sample_id": "sample-id", "prediction": "answer"}
```

提交前建议检查：

- 是否包含 8 个 jsonl 文件。
- 每行是否为合法 JSON。
- `test_sample_id` 和 `prediction` 是否非空。
- 压缩包内文件名是否与平台要求一致。

## 开源计划

2026 年 5 月 21 日至 2026 年 5 月 31 日期间，将技术报告和完整代码提交至 GitHub OpenSeek 官方开源项目，并将 PR 链接回填至 FlagOS 赛事平台。
