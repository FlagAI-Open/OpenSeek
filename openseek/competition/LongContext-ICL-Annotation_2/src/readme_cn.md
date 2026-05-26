# 超长长上下文场景中LLM自动数据标注挑战赛

---

## 消息
<!-- BEGIN NEWS -->
- **[2026-01-20] `发布`：** 赛事信息已在 **Kaggle** 正式上线。详情见：[FlagOS Open Computing Global Challenge](https://www.kaggle.com/competitions/flag-os-open-computing-global-challenge).
- **[2026-01-06] `发布`：** 由 **众智 FlagOS 社区**、**北京智源人工智能研究院（BAAI）** 与 **CCF ODTC** 联合主办的综合性大赛 **FlagOS 开放计算全球挑战赛** 正式发布。详情见：  
  [FlagOS开放计算全球挑战赛- AI赛事通 | 数据算法赛](https://www.competehub.dev/zh/competitions/modelscope180)
<!-- END NEWS -->

---


## 快速开始
### 1. 环境

```bash
openai
torch
flagScale
```

### 2. 下载模型权重
```bash
hf download Qwen/Qwen3-4B --local-dir Qwen3-4B
# or
modelscope download --model Qwen/Qwen3-4B 
```
### 3. 长文本配置
在`Qwen3-4B/config.json`将原有配置替换为：
```json
"rope_scaling": {
    "rope_type": "yarn",
    "factor": 4.0,
    "original_max_position_embeddings": 32768
}
```
### 4. 模型部署

请根据实际需求，配置 `llm_config.yaml` 文件。启动配置  

```bash
cd FlagScale
python run.py --config-path .. --config-name llm_config action=run
```

在模型服务启动后，可通过以下方式测试本地 API：

```bash
python api_test.py
```

如需停止服务，请执行：

```bash
python run.py --config-path .. --config-name llm_config action=stop
```

### 5. 运行/改进基线方法（Baseline）

启动如下命令开始模型标注
```bash
python main.py
```

实现新的标注方法，请修改`method.py`文件。你可以在该文件中：  
* 定义新的指令模板、
* 定义新的上下文示例选择策略
* 定义新的模型推理、标注方案
* 添加自定义后处理逻辑
