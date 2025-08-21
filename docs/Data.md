# 📚 Data

## 1. Data Source Preparation
The pre-training dataset is mainly composed of collected and selected open source datasets.

### English Common Crawl
- https://data.commoncrawl.org/contrib/Nemotron/Nemotron-CC/index.html
- https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu

### Chinese Common Crawl
- https://huggingface.co/datasets/BAAI/CCI3-HQ
- https://huggingface.co/datasets/opencsg/Fineweb-Edu-Chinese-V2.1

### Other Domains
#### Wiki & Books & Arixv
- English: https://huggingface.co/datasets/allenai/dolma
- Chinese: Self-built Chinese encyclopedia, books, and literature data

#### Math
- https://huggingface.co/datasets/OpenCoder-LLM/opc-fineweb-math-corpus
- https://huggingface.co/datasets/EleutherAI/proof-pile-2
- https://huggingface.co/datasets/HuggingFaceTB/finemath

#### Code
- https://huggingface.co/datasets/OpenCoder-LLM/opc-fineweb-code-corpus
- https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus
- https://huggingface.co/datasets/bigcode/the-stack-v2

## 2. Data Synthesis
- **Preliminary Reasoning Data Synthesis**: semantically segment, summarize, organize CoT process, and summarize queries on the original pre-trained documents. take {Query, CoT process, Original document} as one training sample.
- **Labeling system construction**: build labeling system by domain (code, math, general knowledge, etc.) to balance data diversity.
- **Synthesized Data Quality Evaluation and Filtering**: Evaluate the quality of synthesized data based on rules, models, etc., and screen out low-quality data.
- **Synthesis Pipeline Optimization**: Optimize the existing synthesis prompt or synthesis pipeline, re-synthesize based on the first version of reasoning data, etc. to increase the quality of reasoning data.

## 3. Data Preprocessing

### Deduplication
- **Global Fuzzy Deduplication Based on MiniHash**
  - https://github.com/huggingface/datatrove/blob/main/examples/minhash_deduplication.py
- **Exact Substring Deduplication**
  - https://github.com/google-research/deduplicate-text-datasets

### Rule-based Filtering
Developed based on the data-juicer tool https://github.com/modelscope/data-juicer, the main rules include:
- Document character length
- Average sentence character length in documents
- Traditional Chinese to Simplified Chinese conversion
- Sensitive word and safety word filtering

### Quality Classifier
- Chinese quality classifier based on education level estimation
- English quality classifier based on multiple education level estimations


