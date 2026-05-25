# LongContext-ICL-Annotation

Large Language Models Automatic Data Annotation under Long-Context Scenarios.

---

## News
<!-- BEGIN NEWS -->
- **[2026-01-20] `Release`:** The competition is now officially live on **Kaggle**. See details: [FlagOS Open Computing Global Challenge](https://www.kaggle.com/competitions/flag-os-open-computing-global-challenge).
- **[2026-01-06] `Release`:** The comprehensive competition **FlagOS Open Computing Global Challenge** was officially announced, co-hosted by the **FlagOS Community**, the **Beijing Academy of Artificial Intelligence (BAAI)**, and **CCF ODTC**. See details:  
  [FlagOS开放计算全球挑战赛- AI赛事通 | 数据算法赛](https://www.competehub.dev/zh/competitions/modelscope180)
<!-- END NEWS -->

---

## Quick Start

### 1. Environment Setup

```bash
openai
torch
flagScale
```

### 2. Download Model Weights

```bash
hf download Qwen/Qwen3-4B --local-dir Qwen3-4B
# or
modelscope download --model Qwen/Qwen3-4B
```

### 3. Long-Context Configuration

In `Qwen3-4B/config.json`, replace the original configuration with the following settings:

```json
"rope_scaling": {
    "rope_type": "yarn",
    "factor": 4.0,
    "original_max_position_embeddings": 32768
}
```

### 4. Model Deployment

Configure the `llm_config.yaml` file according to your actual requirements. Then start the service with:

```bash
cd FlagScale
python run.py --config-path .. --config-name llm_config action=run
```

After the model service is launched, you can test the local API using:

```bash
python api_test.py
```

To stop the service, run:

```bash
python run.py --config-path .. --config-name llm_config action=stop
```

### 5. Run or Extend the Baseline Method

Start the baseline annotation pipeline with:

```bash
python main.py
```

To implement a new annotation method, modify the `method.py` file. Within this file, you may:

- Define new instruction or prompt templates
- Design new context example selection strategies
- Implement alternative model inference and annotation pipelines
- Add custom post-processing logic
