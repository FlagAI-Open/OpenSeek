# 快速开始指引

## 1. 环境安装

首先，确保您的 Python 环境版本符合要求（建议 Python 3.9+），然后安装项目所需的依赖库：

```
pip install -r requirements.txt
```

注：您可以根据实际显卡驱动版本或特殊需求手动调整 torch 等核心库的版本。

## 2. 模型下载与验证
运行 model.py 脚本以同步 Qwen3-4B 模型权重。

默认使用 ModelScope 国内源，确保下载速度。

脚本运行结束后，若看到模型成功进行逻辑回复，即代表下载与加载测试完毕。

自定义路径：如需修改存储位置，请在 model.py 中调整 model_dir 参数。

```
python model.py
```

## 3. 部署推理服务 (vLLM)
在控制台启动推理后端。请将 --model 参数替换为您在第 2 步中实际的模型存储路径：

```
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/your/Qwen3-4B \
    --served-model-name qwen3-4b \
    --port 2026 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    --max-model-len 32768
```
启动成功标志：当控制台输出 Application startup complete. 时，表示 API 服务已就绪。

## 4. 执行自动化标注任务
保持推理服务控制台开启，新建一个控制台窗口并切换至源代码目录：

```
cd code/src
```
单任务运行 (以题目一为例)：
```
python main.py --task_id 1
```
全任务一键运行 (依次执行 8 道赛题)：
```
for i in {1..8}; do python main.py --task_id $i; done
```
## 项目结构说明
main.py: 标注任务的主入口，负责数据加载、版本控制及结果存证。

method.py: 核心逻辑层。包含 SSD 结构化提示词构建、Qwen 专属 Token 计算以及 count_answer 容错提取算法。

model.py: 模型下载与本地推理推演脚本。

requirements.txt: 项目环境依赖清单。

api_test.py: 用于快速校验 vLLM 服务联通性的测试工具。

## 标注结果输出
所有标注结果将自动存储于项目根目录下的 outputs/ 文件夹中：

.jsonl 文件：存储结构化标注结果（ID + Prediction）。

_responses.txt 文件：完整留存模型原始回复（含思维链），用于实验复盘与 Bug 溯源。

FlagOS 赛事技术方案 - 2026