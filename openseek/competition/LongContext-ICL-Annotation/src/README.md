## 目录结构

- `src/`：源码目录，包含推理入口、任务策略、客户端配置和依赖清单
- `data/`：官方数据集
- `submission_results.zip`：已测试好的提交结果压缩包
- `README.md`：使用说明

## 运行环境

建议使用 Python 3.10 及以上版本。

安装依赖：

```bash
pip install -r src/requirements.txt
```

当前最小依赖仅包含：

- `requests`

## 模型与合规说明

本方案遵循赛事合规要求，核心逻辑均基于官方指定的 `Qwen3-4B` 模型进行设计与优化，未接入任何外部模型或非公开数据集。

在模型底层支撑方面，本方案依托于官方提供的 `FlagScale` 推理平台（或等效的 `FlagScale + Qwen3-4B` 算力环境）。模型的加载及服务端推理由 `FlagScale` 框架提供核心支持。

为保证工程复现的简洁性与跨平台一致性，本项目通过标准化 OpenAI 接口与 `FlagScale` 推理服务通信。代码专注于 ICL 标注策略、动态上下文组织及结果自修复逻辑的实现，模型的底层生命周期与服务化部署建议参考 `FlagScale` 官方文档与环境预设。

## 配置说明

模型服务接口配置位于 `src/llm_config.yaml`。

默认配置如下：

```yaml
api:
  base_url: "http://localhost:30000/v1"
  model: "Qwen3-4B"
  api_key: ""
```

如果服务启用了鉴权，可通过 `api_key` 进行配置。

为确保方案成功复现，请在运行前确认已准备好提供以下标准化接口的 `FlagScale` 推理环境：

- `POST /completions`
- `POST /chat/completions`
- `GET /models`

## 快速开始

1. **环境准备**：
   ```bash
   pip install -r src/requirements.txt
   ```

2. **配置服务**：
   在 `src/llm_config.yaml` 中配置推理服务的 `base_url`；如果服务要求鉴权，同时填写 `api_key`。

3. **执行推理**：
   ```bash
   cd src
   # 默认运行 Task 1-8 全量任务
   python main.py
   
   # 单任务运行 (例如 Task 8)
   python main.py --task_id 8
   
   # 冒烟测试 (每任务仅运行前 5 条)
   python main.py --limit 5
   ```

## 输出产物

本目录中已包含测试完成后的提交文件 `submission_results.zip`，可直接作为赛事平台提交结果使用。

如在复现环境中重新执行推理，程序会在 `submission_results/` 目录下生成：
- **JSONL 文件**：`openseek-1-v1.jsonl` 至 `openseek-8-v1.jsonl`
- **打包文件**：`result.zip`（由当前运行自动生成，包含本次推理得到的全部任务结果）
