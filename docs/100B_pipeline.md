## OpenSeek-Baseline Training Configuration

- **GPU Memory:** 40GB
- **Model Size:** OpenSeek-Small-1.4B

------

## Experiment Pipeline

### Training Preparation

1. **Download Data:**
   - Download the **[OpenSeek-Pretrain-100B dataset](https://huggingface.co/datasets/BAAI/OpenSeek-Pretrain-100B)** from Huggingface.
2. **Download Code:**
   - `git clone https://github.com/FlagAI-Open/OpenSeek.git`
3. **Modify Data Configuration:**
   - Adjust data weights and paths.
4. **Modify Training Hyperparameter Configuration:**
   - Adjust training hyperparameters. **Important hyperparameters:** Learning Rate (lr), Batch Size (global_batch_size).
5. **Other Training Configurations:**
   - **exp_name:** Specify the output path (logs, Wandb, model weights).
   - **nnodes:** Number of training nodes.
   - **nproc_per_node:** Number of GPUs per machine.
   - **hostfile:** List of IPs for all training machines (can be set to null for local training).
6. **Start Training:**
   - `bash openseek/training/run_exp.sh start 3b`

### Performance Validation

1. **Upload Training Wandb**
2. View Stored Model Weights:
   - Typically, model weights are stored in: `${exp_name}/checkpoints`
   - `exp_name` is the output path specified in step 5.
3. Convert Megatron Model Weights to HF Model Weights:
   - `OpenSeek/flagscale/tools/checkpoint/run.sh`
   - Specify input and output paths.
4. Evaluation:
   - `OpenSeek/pretraining/tools/eval/lighteval/run_lighteval_v3_mgpu.sh`

------

## OpenSeek-Pretrain-100B Dataset Details

The following describes the sources of each dataset. Some dataset names have "high/mid/low" suffixes, indicating samples with lower/medium/higher perplexity as inferred by Qwen2.5-3B.

| **Domain**               | **Dataset Name**                             | **Description**                                              |
| ------------------------ | -------------------------------------------- | ------------------------------------------------------------ |
| English Webpage          | Nemotron-CC-high-actual-actual               | Original text from the Nemotron dataset. Labeled as high quality by the Nemotron quality classifier. |
| English Webpage          | Nemotron-CC-medium-actual-actual             | Original text from the Nemotron dataset. Labeled as medium quality by the Nemotron quality classifier. |
| English Webpage          | Nemotron-CC-high-synthetic-distill           | More refined and clearer text derived from the Nemotron dataset through rewriting. The original document was labeled as high quality by the Nemotron quality classifier. |
| English Webpage          | Nemotron-CC-high-synthetic-extract_knowledge | Text with higher knowledge density obtained by filtering out low-information segments from the original Nemotron dataset text. The original document was labeled as high quality by the Nemotron quality classifier. |
| English Webpage          | Nemotron-CC-high-synthetic-diverse_qa_pairs  | Question-answer pairs extracted from the facts contained in the original Nemotron dataset text. The original document was labeled as high quality by the Nemotron quality classifier. |
| English Webpage          | Nemotron-CC-high-synthetic-knowledge_list    | Summarized and listed knowledge points from the original Nemotron dataset documents. The original document was labeled as high quality by the Nemotron quality classifier. |
| English Webpage          | Nemotron-CC-high-synthetic-wrap              | Original Nemotron dataset text rewritten in a Wiki style. The original document was labeled as high quality by the Nemotron quality classifier. |
| English Webpage          | Nemotron-CC-low-synthetic-wrap               | Original Nemotron dataset text rewritten in a Wiki style. The original document was labeled as low quality by the Nemotron quality classifier. |
| Chinese Webpage          | zh_cc                                        | Chinese web dataset collected by OpenSeek, including sources like CCI3-HQ and OpenCSG. |
| Code                     | code                                         | Code dataset collected by OpenSeek, including sources like OpenCoder, SmoLLMCorpus, and StarCoder. |
| Mathematics              | math                                         | Mathematics dataset collected by OpenSeek, including sources like Finemath and Pile-Proof-2. |
| Books                    | books                                        | Book dataset collected by OpenSeek, including sources like Dolma and Dolmino. |
| Encyclopedia             | wiki                                         | Encyclopedia dataset collected by OpenSeek, including sources like Dolma and Dolmino. |
| Papers                   | arxiv                                        | Paper dataset collected by OpenSeek, including sources like Dolma and Pile-Proof-2. |
| Q&A                      | stack                                        | General common sense Q&A dataset collected by OpenSeek, primarily from StackExchange forum data in Dolma. |
| Synthetic Reasoning Data | cot_synthesis_CC                             | Reasoning data synthesized by OpenSeek from web texts. Original texts are primarily from Nemotron-CC data. |
| Synthetic Reasoning Data | cot_synthesis_code                           | Reasoning data synthesized by OpenSeek from code texts. Original texts are primarily from OpenCoder data. |
| Synthetic Reasoning Data | cot_synthesis_math                           | Reasoning data synthesized by OpenSeek from mathematical texts. Original texts are primarily from FineMath and Pile-Proof-2 data. |
| Synthetic Reasoning Data | cot_synthesis_wiki                           | Reasoning data synthesized by OpenSeek from encyclopedia texts. Original texts are primarily from Dolmino data. |
| Synthetic Reasoning Data | cot_synthesis_arxiv                          | Reasoning data synthesized by OpenSeek from paper texts. Original texts are primarily from Pile-Proof-2 and Dolma data. |
| Synthetic Reasoning Data | cot_synthesis_OpenSource                     | Open-source reasoning data instructions collected by OpenSeek. |
| Biological Papers        | pes2o                                        | Biological paper data collected by OpenSeek.                 |

**Detailed Dataset Introductions/Download Links:**

- **Nemotron-CC:** https://data.commoncrawl.org/contrib/Nemotron/Nemotron-CC/index.html
- **CCI3-HQ:** https://huggingface.co/datasets/BAAI/CCI3-Data
- **OpenCSG:** https://huggingface.co/datasets/opencsg/chinese-fineweb-edu
- **OpenCoder:** https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage1
- **SmoLLMCorpus:** https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus
- **StarCoder:** https://huggingface.co/datasets/bigcode/starcoderdata
- **FineMath:** https://huggingface.co/datasets/HuggingFaceTB/finemath
- **Pile-Proof-2:** https://huggingface.co/datasets/EleutherAI/proof-pile-2
- **Dolma:** https://huggingface.co/datasets/allenai/dolma
- **Dolmino:** https://huggingface.co/datasets/allenai/dolmino-mix-1124