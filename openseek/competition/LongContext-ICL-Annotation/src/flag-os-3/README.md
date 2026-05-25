# FlagOS LongContext ICL Annotation

超长长上下文场景下 LLM 自动数据标注挑战赛 - 基线实现代码

## 关于项目

本项目是 **FlagOS 开放计算全球挑战赛** - "超长长上下文场景中 LLM 自动数据标注" 赛道的官方基线实现。该赛事由 **众智 FlagOS 社区**、**北京智源人工智能研究院（BAAI）** 与 **CCF ODTC** 联合主办。

## 项目特点

- ✅ 基于 Qwen3-4B 长上下文模型的端到端标注基线
- ✅ 支持 8 个不同领域的标注任务（数学推理、语言分析、NLI、代码生成等）
- ✅ 集成 nanobot 智能标注框架
- ✅ 支持多线程并行标注
- ✅ 内置多种 ICL 示例选择策略

---

## 任务说明

本项目包含 8 个标注任务，每个任务都需要超长上下文支持：

| 任务ID | 任务名称 | 最小上下文 | 测试样本数 | 任务描述 |
|--------|---------|-----------|-----------|---------|
| 1 | closest_integers | 30K | 500 | 数学：最近整数推理 |
| 2 | count_nouns_verbs | 30K | 500 | 语言学：名词动词计数 |
| 3 | collatz_conjecture | 30K | 500 | 数学：考拉兹猜想验证 |
| 4 | conala_concat_strings | 30K | 500 | 代码：字符串拼接生成 |
| 5 | tweet_sadness_detection | 30K | 500 | 情感：悲伤情绪检测 |
| 6 | mnli_same_genre_classification | 30K | 500 | NLI：自然语言推断 |
| 7 | jeopardy_answer_generation | 30K | 500 | 问答：Jeopardy 答案生成 |
| 8 | kernel_generation | 16K | 166 | 代码：算子/内核生成 |

---

## 项目结构

```
.
├── install.sh              # 安装脚本
├── run.sh                  # 运行脚本（由install.sh生成）
├── run.sh__               # 运行脚本模板
├── requirements.txt        # Python 依赖
├── config/                # 配置目录
│   ├── config.json__      # 配置模板
│   └── workspace/         # nanobot 工作目录
├── data/                  # 任务数据集
│   ├── openseek-1_closest_integers.json
│   ├── openseek-2_count_nouns_verbs.json
│   ├── openseek-3_collatz_conjecture.json
│   ├── openseek-4_conala_concat_strings.json
│   ├── openseek-5_semeval_2018_task1_tweet_sadness_detection.json
│   ├── openseek-6_mnli_same_genre_classification.json
│   ├── openseek-7_jeopardy_answer_generation_all.json
│   └── openseek-8_kernel_generation.json
├── data_processed/        # 预处理后的数据（带标签、缓存）
├── outputs/               # 标注结果输出目录
├── nanobot/               # nanobot AI 框架（已预装）
└── LongContext-ICL-Annotation/
    └── src/              # 核心标注代码
        ├── main.py       # 主程序入口
        ├── method.py     # 标注方法实现
        ├── rules.py      # 任务规则定义
        ├── sample_selector_base.py  # 示例选择策略
        ├── build_sadness_representative_cache.py
        ├── tag_task2.py  # 任务2预处理脚本
        ├── tag_task7_examples.py  # 任务7预处理脚本
        └── merge_task8_results.py  # 结果合并脚本
```

---

## 快速开始

### 1 准备工作

#### 1.1 环境要求

- Python 3.11+
- transformers
- tqdm
- 完整依赖见 `requirements.txt`

#### 1.2 文件与模型
 
- 部署 Qwen3-4B 模型并启动 vLLM 推理服务：
- 赛题数据存放于./data
- 清理 ./outputs以及data_processed

### 2. 安装

```bash
# 修改 install.sh 中的以下变量，适配你的环境：
#
# __MODEL_NAME__   # 模型名称，用于配置标识
# __MODEL_URL__    # vLLM 推理服务地址，如 http://127.0.0.1:9010/v1/
# __MODEL_PATH__   # 模型权重文件路径，如 /path/to/Qwen3-4B/
# __WORK_DIR__     # 工作目录，默认使用当前目录（`pwd`）
#

# 安装 Python 依赖并生成配置
bash install.sh
```

安装脚本会自动：
- 安装所有 Python 依赖包
- 以可编辑模式安装 nanobot 框架
- 生成配置文件 `config/config.json`
- 生成运行脚本 `run.sh`

### 3. 运行标注

```bash
# 运行完整标注流程
bash run.sh
```

### 4. 任务执行流程

`run.sh` 执行以下步骤：

```bash
# 1. 构建任务4,5示例缓存
python build_sadness_representative_cache.py
python build_concat_cache.py

# 2. 任务2预处理（词性标注）
python tag_task2.py

# 3. 任务7预处理（标签生成）
python tag_task7_examples.py --num_examples 1000

# 4. 运行指定任务的标注（默认任务8）
for ...

# 5. 合并任务8结果
python merge_task8_results.py
```

---

## 配置说明

### 模型配置（config/config.json）

```json
{
  "agents": {
    "defaults": {
      "workspace": "./config/workspace",
      "model": "Qwen3-4B-ascend-flagos",
      "provider": "auto",
      "maxTokens": 20000,
      "contextWindowTokens": 32768,
      "temperature": 0.7,
      "maxToolIterations": 200
    }
  },
  "providers": {
    "vllm": {
      "apiBase": "http://127.0.0.1:9010/v1/",
      "timeout": 1800
    }
  }
}
```

## 评估与结果

标注结果将保存在 `outputs/` 目录下。

** 注意 **
对于任务8，由于需要进行多轮校验，尽管不会出现上下文超长问题，但可能会有LLM timeout等问题，对任务实现逻辑进行了优化，每次从上一版本文件中读取，只标注当前标注为空的结果，因此任务8需要运行多次。最后执行文件合并。

## 部署环境
硬件环境基于比赛方提供的算力资源，包括两张NPU 910B(64G)显卡，基于flagscale，采用pipeline并行模式部署模型，32并发，启用KVCache。具体配置文件见examples(需要修改模型路径)：
启动命令为：
```shell
flagscale serve qwen3
```

```yaml
root@f9145b39470e:~# more /workspace/examples/qwen3/conf/serve.yaml
defaults:
- _self_
- serve: 4b

experiment:
  exp_name: qwen3_4b
  exp_dir: outputs/${experiment.exp_name}
  task:
    type: serve
    backend: vllm
  runner:
    hostfile: null
    deploy:
      use_fs_serve: false
  envs:
    CUDA_VISIBLE_DEVICES: 0,1
    CUDA_DEVICE_MAX_CONNECTIONS: 1

action: run

hydra:
  run:
    dir: ${experiment.exp_dir}/hydra

root@f9145b39470e:~# more /workspace/examples/qwen3/conf/serve.yaml
defaults:
- _self_
- serve: 4b

experiment:
  exp_name: qwen3_4b
  exp_dir: outputs/${experiment.exp_name}
  task:
    type: serve
    backend: vllm
  runner:
    hostfile: null
    deploy:
      use_fs_serve: false
  envs:
    CUDA_VISIBLE_DEVICES: 0,1
    CUDA_DEVICE_MAX_CONNECTIONS: 1

action: run

hydra:
  run:
    dir: ${experiment.exp_dir}/hydra
```



## 致谢

感谢 FlagOS 社区、智源研究院、CCF ODTC 对本赛事的支持！
