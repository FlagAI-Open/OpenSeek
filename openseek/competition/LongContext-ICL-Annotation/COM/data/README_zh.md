# 数据集说明

本仓库提供 **LLM Automatic Data Annotation** 的官方数据集。

这些数据集专门用于评估大语言模型（LLMs）在 超长上下文设置 下，使用 In-context Learning（ICL）范式进行 自动数据标注 的能力。

---

## 概览

- 大多数任务要求 **最小 ICL 上下文长度为 30K tokens**，接近 **Qwen3-4B** 的标准上下文限制，以评估长上下文理解、提示工程以及示例选择策略。
- 任务 **openseek-8** 配置了 **更短的最小上下文长度（15K tokens）** 和 **更小的测试集**，以反映**算子生成**的独特挑战。
- 所有数据集均以 **固定且标准化的测试划分** 发布，以确保提交之间的公平比较与可复现性。
- 任务集合覆盖 **多样的领域与推理类型**，包括符号推理、语言学分析、自然语言推断、代码相关任务以及开放式生成。

| Task ID | task name | Minimum ICL context | Test sample number | 
| --- | --- | --- | --- | 
| openseek-1 | closest_integers |  30K  | 500 | 
| openseek-2 | count_nouns_verbs |  30K  | 500 | 
| openseek-3 | collatz_conjecture |  30K  | 500 | 
| openseek-4 | conala_concat_strings |  30K  | 500 | 
| openseek-5 | semeval_2018_task1_tweet_sadness_detection |  30K  | 500 | 
| openseek-6 | mnli_same_genre_classification |  30K  | 500 | 
| openseek-7 | jeopardy_answer_generation_all |  30K  | 500 | 
| openseek-8 | kernel_genernation |  16K  | 166 | 

---

## 数据结构

数据集以 `JSON` 格式组织，每个任务对应一个独立的 `.json` 文件。以下是数据结构的简要说明：

- `task_id`: 任务的唯一标识符。
- `task_name`: 任务的简短、便于理解的人类可读名称。
- `Definition`: 对模型应执行内容的详细描述。
- `examples`: 用于理解任务格式的演示样本（不一定用于计分）。每个示例包含：`id`、`input` 和 `output`。
- `test_samples`: 参赛者需要预测的样本。标签/真实值被隐藏。每个测试样本包含：`id` 和 `input`。
- `License`: 数据集许可证名称和/或许可证文本的 URL，用于说明允许的使用方式与再分发规则。

---

## 使用说明

- 参赛者必须使用**按原样提供的官方数据集**，不得更改测试划分或标签，以用于排行榜评测。
- 任何预处理步骤、上下文构建策略或示例选择机制，都应在随附的技术报告中清晰描述。
- 所有实验结果必须能够使用本仓库中的数据集**完全复现**。