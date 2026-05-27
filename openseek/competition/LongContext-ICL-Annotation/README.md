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
hf download Qwen/Qwen3-4B --local-dir ~/.cache/modelscope/hub/models/Qwen/Qwen3-4B
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
git clone https://gitee.com/flagos-ai/FlagScale
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
cd src
python main.py --task_id=0 --task_step=0
```

To implement a new annotation method, modify the `method.py` file. Within this file, you may:

- Define new instruction or prompt templates
- Design new context example selection strategies
- Implement alternative model inference and annotation pipelines
- Add custom post-processing logic


The table below is the release compatibility matrix for vLLM Ascend release.
```csv
vLLM Ascend	    vLLM	      Python	          CANN        PyTorch/torch_npu       Triton Ascend   Recommend
v0.14.0rc1	    v0.14.1	    >= 3.10, < 3.12	  8.5.0	      2.9.0  / 2.9.0	        3.2.0           Y
v0.13.0	        v0.13.0	    >= 3.10, < 3.12	  8.5.0	      2.9.0  / 2.8.0.post2	  3.2.0           N
v0.13.0rc2	    v0.13.0	    >= 3.10, < 3.12	  8.5.0	      2.8.0  / 2.8.0.post1	  3.2.0           N
v0.13.0rc1	    v0.13.0	    >= 3.10, < 3.12	  8.3.RC2	    2.8.0  / 2.8.0	        3.2.0           Y
v0.12.0rc1	    v0.12.0	    >= 3.10, < 3.12	  8.3.RC2	    2.8.0  / 2.8.0	        3.2.0           Y
v0.11.0	        v0.11.0	    >= 3.9 , < 3.12	  8.3.RC2	    2.7.1  / 2.7.1.post1	  3.2.0           N
v0.11.0rc3	    v0.11.0	    >= 3.9,  < 3.12	  8.3.RC2	    2.7.1  / 2.7.1.post1	  3.2.0           N
v0.11.0rc2	    v0.11.0	    >= 3.9,  < 3.12	  8.3.RC2	    2.7.1  / 2.7.1	        3.2.0           Y
v0.11.0rc1	    v0.11.0	    >= 3.9,  < 3.12	  8.3.RC1	    2.7.1  / 2.7.1	        3.2.0           Y
```
