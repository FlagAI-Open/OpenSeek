# CCI4.0-ZH-HQ-Classifiers

CCI4.0中文语料质量标注脚本，用于对CCI4.0中文语料库进行质量评分。该脚本实现了多个质量分类器的推理，用于全面的文本质量评估。

## 概述

该脚本专门用于对CCI4.0中文语料库进行质量评分。考虑到单一分类器在识别高质量预训练文档时召回率有限，我们参考了Nemotron-CC处理英文Common Crawl的方法，构建了三个独立的质量分类器来重新评估中文预训练数据。

该推理代码结合了三个质量分类器：基于Qwen、DeepSeek得到的训练数据训练的两个Roberta分类器和一个基于Fasttext的文本质量分类器，同时我们也在推理代码中加入了一个Fineweb-edu-classifier分类器，为中文文本数据提供全面的质量评估。每个分类器通过一定的集成策略为最终质量分数做出贡献。

## 质量分类器训练

我们使用两个大语言模型对中文样本进行质量标注：
- 基于Qwen2.5-72B和DeepSeek-V3分别构建了两个46万条样本的训练集，通过参数调优开发了两个独立的质量分类器
- 实现了一个基于FastText的分类器，该分类器在指令格式数据和高分帖子组合上训练，并经过了多轮迭代排除一些不相关因素的影响

## 功能特点

- 模型支持：
  - 基于Qwen2.5-72B的质量分类器
  - 基于DeepSeek-V3的质量分类器
  - 基于FastText的质量分类器
  - 领域分类器
- 批量处理能力
- 全面的文本预处理
- 处理状态跟踪日志系统
- 支持单文本和批量处理
- 支持JSONL输入/输出格式

## 环境要求

- Python 3.6+
- PyTorch
- Transformers
- FastText
- jsonlines
- jieba

## 使用说明

安装所需依赖：

```bash
pip install torch transformers fasttext jsonlines jieba 
```

### 下载对应的几个质量分类器

共4个质量打分模型
- Qwen2.5-72B based quality classifier —— Scorer 1
- DeepSeek-V3 based quality classifier —— Scorer 2
- FastText-based quality classifier    —— Scorer 3
- FineWeb-Edu classifier               —— Scorer 4

### 基本用法

```bash
python data_infer.py     --roberta-model-path1 "/path/to/roberta_model (Scorer 1)" \
                         --roberta-model-path2 "/path/to/roberta_model (Scorer 2)" \
                         --deberta-model-path "/path/to/deberta_model (Scorer 4)" \
                         --fasttext-model-path "/path/to/fasttext_model (Scorer 3)" \
                         --input-file-path "/path/to/input1.jsonl" "/path/to/input2.jsonl" \
                         --output-file-dir "/path/to/output" \
                         --batch-size 16 \
                         --model "roberta-qwen" "roberta-deepseek" "deberta" "fasttext" \
                         --cuda-device 1
                         --stopwords-path "/path/to/stopwords"
```

### 输入格式

输入应为JSONL文件，每行包含一个至少具有"text"字段的JSON对象：

```json
{"text": "您的文本内容"}
```

### 输出格式

输出将是一个JSONL文件，包含原始内容以及每个模型的质量评分：

```json
{
    "text": "原始文本",
    "quality": {
        "fasttext_400k_train_1gram": *,
        "fasttext_400k_train_1gram_normal": *,
        "roberta_qwen": *,
        "roberta_qwen_normal" : *,
        "roberta_deepseek" : *,
        "roberta_deepseek_normal" : *,
        "fine_web" : *,
        "fine_web_normal" : *
    }
}
```

## 模型详情

### 基于Qwen2.5-72B的分类器
- 使用Qwen2.5-72B作为基础模型
- 在46万质量标注样本上训练
- 提供质量分类分数

### 基于DeepSeek-V3的分类器
- 基于DeepSeek-V3架构
- 在46万质量标注样本上训练
- 支持序列分类

### 基于FastText的分类器
- 实现FastText文本分类
- 在指令格式数据上训练
- 包含文本预处理

### FineWeb-Edu classifier
