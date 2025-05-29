## OpenSeek-Baseline 训练配置

* **显存：** 40G
* **模型尺寸：** OpenSeek-Small-1.4B

---

## 实验流水线

### 训练准备

1.  **下载数据：**
    * 从Huggingface下载 **[OpenSeek-Pretrain-100B 数据集](https://huggingface.co/datasets/BAAI/OpenSeek-Pretrain-100B)**。
2.  **下载代码：**
    * `git clone https://github.com/FlagAI-Open/OpenSeek.git`
3.  **修改数据配置：**
    * 修改数据权重和路径。
4.  **修改训练超参配置：**
    * 修改训练超参数。**重要超参数：** Learning Rate (lr)， Batch Size (global\_batch\_size)。
5.  **其他训练配置：**
    * **exp\_name：** 制定输出路径（日志, Wandb, 模型权重）。
    * **nnodes：** 训练节点数量。
    * **nproc\_per\_node：** 每台机器包含的卡数量。
    * **hostfile：** 所有训练机器的IP列表 (本地训练可设置为null)。
6.  **启动训练：**
    * `bash openseek/training/run_exp.sh start 3b`

### 效果验证

1.  **上传训练Wandb**
2.  **查看存储的模型权重：**
    * 通常，模型权重被存储在：`${exp_name}/checkpoints`
    * `exp_name` 是在第5步指定的输出路径。
3.  **转换Megatron模型权重到HF模型权重：**
    * `OpenSeek/flagscale/tools/checkpoint/run.sh`
    * 指定输入及输出路径。
4.  **评测：**
    * `OpenSeek/pretraining/tools/eval/lighteval/run_lighteval_v3_mgpu.sh`

---

## OpenSeek-Pretrain-100B 数据集详细介绍

各数据集来源介绍，部分数据名带有high/mid/low后缀。分别是我们用Qwen2.5-3B推理困惑度较低/中等/较高的样本集合。

| 领域         | 数据名                                        | 介绍                                                         |
| :----------- | :-------------------------------------------- | :----------------------------------------------------------- |
| 英文网页     | Nemotron-CC-high-actual-actual                | Nemotron数据集的原始文本。Nemotron质量分类器标注为高质量的部分。 |
| 英文网页     | Nemotron-CC-medium-actual-actual              | Nemotron数据集的原始文本。Nemotron质量分类器标注为中等质量的部分。 |
| 英文网页     | Nemotron-CC-high-synthetic-distill            | Nemotron数据集通过改写得到的更加精炼、清晰的文本。Nemotron质量分类器标注原始文档为高质量的部分。 |
| 英文网页     | Nemotron-CC-high-synthetic-extract\_knowledge | Nemotron数据集从原始文本中过滤信息含量较低的片段，得到的知识点更加密集的文本。Nemotron质量分类器标注原始文档为高质量的部分。 |
| 英文网页     | Nemotron-CC-high-synthetic-diverse\_qa\_pairs | Nemotron数据集基于原始文本包含的事实，提取出的问答对。Nemotron质量分类器标注原始文档为高质量的部分。 |
| 英文网页     | Nemotron-CC-high-synthetic-knowledge\_list    | Nemotron数据集从原始文档中总结、罗列的知识点。Nemotron质量分类器标注原始文档为高质量的部分。 |
| 英文网页     | Nemotron-CC-high-synthetic-wrap               | Nemotron数据集按照wiki风格改写的原始文本。Nemotron质量分类器标注原始文档为高质量的部分。 |
| 英文网页     | Nemotron-CC-low-synthetic-wrap                | Nemotron数据集按照wiki风格改写的原始文本。Nemotron质量分类器标注原始文档为低质量的部分。 |
| 中文网页     | zh\_cc                                        | OpenSeek搜集的中文网页数据集。包括CCI3-HQ、OpenCSG等来源。   |
| 代码         | code                                          | OpenSeek搜集的代码数据集。包括OpenCoder、SmollmCorpus、StarCoder等来源。 |
| 数学         | math                                          | OpenSeeks搜集的数学数据集。包括Finemath、Pile-Proof-2等来源。 |
| 书籍         | books                                         | OpenSeek搜集的书籍类数据集。包括Dolma、Dolmino等来源。       |
| 百科         | wiki                                          | OpenSeek搜集的百科类数据集。包括Dolma、Dolmino等来源。       |
| 论文         | arxiv                                         | OpenSeek搜集的论文类数据集。包括Dolma、Pile-Proof-2等来源。  |
| 问答         | stack                                         | OpenSeek搜集的通用常识类数据集。主要来自Dolma中的StackExchange论坛数据。 |
| 合成推理数据 | cot\_synthesis\_CC                            | OpenSeek从网页文本中合成的推理数据。原始文本主要来自Nemotron-CC数据。 |
| 合成推理数据 | cot\_synthesis\_code                          | OpenSeek从代码文本中合成的推理数据。原始文本主要来自OpenCoder数据。 |
| 合成推理数据 | cot\_synthesis\_math                          | OpenSeek从数学文本中合成的推理数据。原始文本主要来自FineMath、Pile-Proof-2数据。 |
| 合成推理数据 | cot\_synthesis\_wiki                          | OpenSeek从百科文本中合成的推理数据。原始文本主要来自Dolmino数据。 |
| 合成推理数据 | cot\_synthesis\_arxiv                         | OpenSeek从论文文本中合成的推理数据。原始文本主要来自Pile-Proof-2、Dolma数据。 |
| 合成推理数据 | cot\_synthesis\_OpenSource                    | OpenSeek搜集的开源推理数据指令数据。                         |
| 生物论文     | pes2o                                         | OpenSeek搜集的生物论文数据。                                 |

**各数据集的详细介绍/下载链接：**

* **Nemotron-CC：** [https://data.commoncrawl.org/contrib/Nemotron/Nemotron-CC/index.html](https://data.commoncrawl.org/contrib/Nemotron/Nemotron-CC/index.html)
* **CCI3-HQ：** [https://huggingface.co/datasets/BAAI/CCI3-Data](https://huggingface.co/datasets/BAAI/CCI3-Data)
* **OpenCSG：** [https://huggingface.co/datasets/opencsg/chinese-fineweb-edu](https://huggingface.co/datasets/opencsg/chinese-fineweb-edu)
* **OpenCoder：** [https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage1](https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage1)
* **SmoLLMCorpus：** [https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus)
* **StarCoder：** [https://huggingface.co/datasets/bigcode/starcoderdata](https://huggingface.co/datasets/bigcode/starcoderdata)
* **FineMath：** [https://huggingface.co/datasets/HuggingFaceTB/finemath](https://huggingface.co/datasets/HuggingFaceTB/finemath)
* **Pile-Proof-2：** [https://huggingface.co/datasets/EleutherAI/proof-pile-2](https://huggingface.co/datasets/EleutherAI/proof-pile-2)
* **Dolma：** [https://huggingface.co/datasets/allenai/dolma](https://huggingface.co/datasets/allenai/dolma)
* **Dolmino：** [https://huggingface.co/datasets/allenai/dolmino-mix-1124](https://huggingface.co/datasets/allenai/dolmino-mix-1124)



