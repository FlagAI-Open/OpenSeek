# 复现路径

[English Version](README.md)

本文档面向**主办方/复现者**，提供从零开始复现全部 8 个 task 推理结果的完整步骤。

---

## 1. 拉取代码并准备目录

```bash
git clone https://github.com/FlagAI-Open/OpenSeek.git
cd OpenSeek/openseek/competition/LongContext-ICL-Annotation
# 本提交位于 COM/ 目录下
cd COM
```

---

## 2. 安装环境

复现者可根据自己的环境情况灵活调整：

```bash
# conda 环境创建（可选，已有 Python 3.11 环境可跳过）
conda init bash && source ~/.bashrc
conda create -n flagscale python=3.11.11 -y
conda activate flagscale

# 安装 Python 依赖
pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple/

# 下载 Qwen3-4B 模型权重
modelscope download --model Qwen/Qwen3-4B --cache_dir ./models
```

下载完成后，模型权重位于 `COM/models/Qwen/Qwen3-4B`。

---

## 3. 配置路径

编辑 `src/common/paths.py`，修改以下变量以匹配本机环境：

```python
# 复现时改这几行 ↓↓↓
COM_ROOT = '/your/path/to/COM'                        # COM 目录绝对路径
MODEL_DIR = COM_ROOT + '/models/Qwen/Qwen3-4B'       # 模型权重位置
VLLM_MODEL_ID = '../models/Qwen/Qwen3-4B'            # 与 llm_config.yaml 中 model 字段保持一致
```

其余路径（`DATA_DIR` / `FINAL_OUTPUT_DIR` / `SRC_DIR` 等）均由 `COM_ROOT` 自动拼接，无需修改。

可执行自检确认路径正确：

```bash
cd src && python -m common.paths
```

---

## 4. 启动模型服务

```bash
git clone https://github.com/FlagOpen/FlagScale.git
cd FlagScale
python run.py --config-path ../env --config-name llm_config action=run
```

服务启动后监听 `localhost:2026`，提供 OpenAI 兼容 API。

使用 API 连通性测试验证部署：

```bash
cd ..  # 回到 COM/
python env/api_test.py
```

若返回正常响应，说明模型服务已就绪。

---

## 5. 准备数据

`data/` 目录为官方比赛数据，文件命名遵循：

```
data/
├── openseek-1_closest_integers.json
├── openseek-2_count_nouns_verbs.json
├── openseek-3_collatz_conjecture.json
├── openseek-4_conala_concat_strings.json
├── openseek-5_semeval_2018_task1_tweet_sadness_detection.json
├── openseek-6_mnli_same_genre_classification.json
├── openseek-7_jeopardy_answer_generation_all.json
└── openseek-8_kernel_generation.json
```

---

## 5.1 （可选）重新生成 CoT 数据 & 规整 Task 8 数据

> 以下步骤**非必须**——我们已将生成好的 CoT 数据（`src/taskN/cot_data/`）和规整后的 Task 8 数据（`src/task8/normalized_data/`）随代码一起提交。如需从头复现，可执行以下命令（需模型服务已启动）。

```bash
# 生成各任务 CoT（Task 1/2/3/4/6 各约 1~3 小时）
cd src/task1 && python generate_cot.py && cd ../..
cd src/task2 && python generate_cot.py && cd ../..
cd src/task3 && python generate_cot.py && cd ../..
cd src/task4 && python generate_cot.py && cd ../..
cd src/task6 && python generate_cot.py && cd ../..

# 规整 Task 8 原始数据（几秒即可完成）
cd src/task8 && python normalize_dataset.py && cd ../..
```

---

## 6. 运行全部 task

```bash
bash scripts/run_all.sh          # 运行全部 8 个 task
bash scripts/run_all.sh 1 3 5    # 选择性运行指定 task
bash scripts/run_all.sh 3-7      # 范围运行
```

**运行时间参考**（单卡 4090D）：

| Task | 总耗时 | 均速 | 说明 |
|------|--------|------|------|
| Task 1 | ~2h 14min | 16.1s/样本 | 单轮推理 |
| Task 2 | ~1h 30min | 10.8s/样本 | 单轮推理 |
| Task 3 | ~2h 11min | 15.7s/样本 | 单轮推理 |
| Task 4 | ~1h 50min | 13.2s/样本 | 单轮推理 |
| Task 5 | ~3h 19min | 23.8s/样本 | 5 轮投票 |
| Task 6 | ~4h 31min | 32.5s/样本 | 3 轮投票 |
| Task 7 | ~6h 59min | 50.3s/样本 | 三策略 + 验证器 + 重调用 |
| Task 8 | ~4h 26min | 96.0s/样本 | 六阶段 pipeline |

---

## 7. 校验输出

```bash
bash scripts/verify_outputs.sh
```

最终 8 份预测文件位于 `outputs/openseek-{1..8}-v1.jsonl`。

---

## 8. 停止模型服务

```bash
cd FlagScale
python run.py --config-path ../env --config-name llm_config action=stop
```