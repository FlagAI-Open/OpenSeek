# H20 迁移与首轮提交指南

本文档用于把当前 `LongContext-ICL-Annotation` baseline 从本机迁移到 H20 服务器，并尽快产出第一版可提交结果。

## 1. 结论先说

如果你现在用的是本机这张 `RTX 4060 Laptop GPU 8GB`，迁到 H20 基本是更合适的。

这场比赛更看重：

- 长上下文推理是否稳定
- 显存是否足够支撑 `Qwen3-4B + 长上下文 + vLLM/FlagScale`
- 模型服务是否能长时间稳定运行
- 批量生成 `8` 个任务结果时是否容易 OOM
- 调 prompt 和示例选择时的迭代速度

对这类工作负载，H20 这类数据中心 Hopper 系列卡通常明显优于 8GB 的移动端 4060。

## 2. 为什么 H20 更适合这场比赛

### 2.1 关键不是“理论算力”，而是显存和工程稳定性

本赛题要求：

- 固定模型：`Qwen3-4B`
- 长上下文 ICL
- `8` 个任务批量推理
- 使用 `FlagScale` 启动模型服务

在消费级 8GB 显卡上，很容易碰到：

- 上下文一长就爆显存
- vLLM 服务参数必须压得很保守
- batch 稍微大一点就不稳定
- 调 prompt 的试错成本很高

H20 的优势主要体现在：

- 更大的显存余量
- 更高的显存带宽
- 更适合长期稳定跑服务
- 更适合反复迭代实验

### 2.2 对比赛来说，“能稳定跑完”比“偶尔能跑”更重要

当前 baseline 是按任务逐条生成预测，再输出 `8` 个 `jsonl` 文件并打包成一个 `zip` 提交。  
如果 GPU 在边缘状态运行，最常见的问题不是完全不能跑，而是：

- 某个任务中途挂掉
- 服务偶发超时
- 一次运行太慢，导致调参周期过长
- 需要不断压缩上下文预算，影响效果

H20 更大的价值在于降低这些“工程性失败”的概率。

## 3. 什么时候 H20 不一定直接提分

H20 更强，不代表迁移后一定立刻提分。真正决定成绩的仍然是：

- 示例选择策略
- prompt 结构
- 长上下文组织方式
- 输出格式约束
- 后处理与错误恢复

所以正确预期是：

- H20 会明显改善可运行性、稳定性和迭代速度
- H20 不会自动替你提升方法本身的质量

## 4. 当前代码基础

本地已经准备好的关键文件：

- `src/main.py`
- `src/method.py`
- `src/run_all_tasks.py`
- `src/README_local.md`
- `src/llm_config.yaml`

其中我们已经做过这些改造：

- 修掉了原始 baseline 里写死的本地路径
- 支持批量跑 `1-8` 号任务
- 支持自动打包 `zip`
- 支持通过环境变量切换 tokenizer 路径和服务地址
- 没有 tokenizer 时也能做流程烟测

## 5. 建议的 H20 服务器目录规划

推荐目录约定：

```bash
/workspace/flagos/
  ├─ OpenSeek/
  ├─ FlagScale/
  ├─ models/
  │   └─ Qwen3-4B/
  └─ outputs/
```

## 6. 迁移步骤

### 6.1 代码同步

把下面两个仓库同步到 H20 机器：

- `OpenSeek`
- `FlagScale`

如果你用 git：

```bash
cd /workspace/flagos
git clone https://github.com/FlagAI-Open/OpenSeek.git
git clone https://github.com/FlagOpen/FlagScale.git
```

如果你要保留当前本地改过的 baseline，请把这些文件一起带过去：

- `OpenSeek/openseek/competition/LongContext-ICL-Annotation/src/main.py`
- `OpenSeek/openseek/competition/LongContext-ICL-Annotation/src/method.py`
- `OpenSeek/openseek/competition/LongContext-ICL-Annotation/src/run_all_tasks.py`
- `OpenSeek/openseek/competition/LongContext-ICL-Annotation/src/README_local.md`
- `OpenSeek/openseek/competition/LongContext-ICL-Annotation/bootstrap_h20.sh`

### 6.2 Conda 环境与依赖

#### 6.2.1 推荐环境版本

如果你准备尽量贴近当前 `FlagScale` 仓库里的 CUDA inference 依赖，推荐：

- Linux
- Python `3.12`
- CUDA `12.8` 对应的软件栈

这样做的原因是当前仓库里提供的 vLLM wheel 是：

- `vllm-0.13.0+fl.0.1.cu128`
- 对应 `cp312`

也就是说，**如果你想尽量少踩依赖坑，优先使用 `Python 3.12`。**

#### 6.2.2 系统层准备

```bash
sudo apt update
sudo apt install -y git curl wget unzip build-essential
```

先确认这些基础工具存在：

```bash
git --version
nvidia-smi
python3 --version
```

#### 6.2.3 创建 conda 环境

```bash
conda create -n openseek-h20 python=3.12 -y
conda activate openseek-h20
python -V
pip install -U pip setuptools wheel
```

#### 6.2.4 推荐安装顺序

```bash
cd /workspace/flagos

pip install -r ./FlagScale/requirements/common.txt

pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
  torch==2.9.0 \
  torchaudio==2.9.0 \
  torchvision==0.24.0

pip install "vllm @ https://resource.flagos.net/repository/flagos-pypi-hosted/packages/vllm/0.13.0%2Bfl.0.1.cu128.g72506c983/vllm-0.13.0%2Bfl.0.1.cu128.g72506c983-cp312-cp312-linux_x86_64.whl"

pip install transformers==4.57.6 openai==2.29.0

pip install -e ./FlagScale
```

#### 6.2.5 这些依赖是必须的

最低需要：

- `torch`
- `vllm`
- `transformers`
- `requests`
- `tqdm`
- `openai`
- `hydra-core`
- `pyyaml`
- `FlagScale`

其中：

- `torch + vllm` 是启动模型服务的关键
- `transformers` 用于 tokenizer 和 prompt 长度检查
- `openai` 主要保留 Ascend/OpenAI-compatible 接口兼容

#### 6.2.6 安装完成后的验证

```bash
python - <<'PY'
import torch, transformers, requests, openai
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu_count:", torch.cuda.device_count())
    print("gpu0:", torch.cuda.get_device_name(0))
print("transformers:", transformers.__version__)
print("requests:", requests.__version__)
print("openai:", openai.__version__)
PY
```

再额外确认：

```bash
python -c "import flagscale; print('flagscale ok')"
```

#### 6.2.7 如果环境装不上

优先排查：

- Python 版本是不是 `3.12`
- 机器 CUDA 栈是不是和 `cu128` 兼容
- 驱动是否足够新
- `torch` 和 `vllm` 是否来自同一套 CUDA 版本

如果机器不是 `cu128` 路线，不建议硬套上面的 wheel。  
这时应使用服务器侧已经验证过的 `torch/vllm` 组合，再回头适配 `FlagScale`。

### 6.3 下载模型

```bash
hf download Qwen/Qwen3-4B --local-dir /workspace/flagos/models/Qwen3-4B
```

如需长上下文 YaRN 配置，按比赛仓库 README 修改 `config.json`。

### 6.4 FlagScale 服务配置

当前 `src/llm_config.yaml` 已经支持通过环境变量读取模型路径、监听地址、端口和 `CUDA_VISIBLE_DEVICES`。  
如果你按下面的环境变量方式启动，通常不需要手动改配置文件。

默认配置等价于：

```yaml
serve:
- serve_id: vllm_model
  engine: vllm
  engine_args:
    model: /workspace/flagos/models/Qwen3-4B
    host: 0.0.0.0
    port: 2026
    gpu_memory_utilization: 0.9
```

第一轮建议保守一些：

- `gpu_memory_utilization: 0.85`

等确认稳定后再往上调。

### 6.5 设置环境变量

```bash
export OPENSEEK_TOKENIZER_PATH=/workspace/flagos/models/Qwen3-4B
export OPENSEEK_MODEL_NAME=/workspace/flagos/models/Qwen3-4B
export OPENSEEK_VLLM_HOST=0.0.0.0
export OPENSEEK_VLLM_PORT=2026
export OPENSEEK_VLLM_URL=http://127.0.0.1:2026/v1/completions
export OPENSEEK_CONTEXT_BUDGET=8192
```

说明：

- `OPENSEEK_CONTEXT_BUDGET` 是示例选择时的预算，不是模型理论最大上下文
- 第一轮建议先保守，跑通后再扩
- 如果你想改服务端口或绑定地址，优先改 `OPENSEEK_VLLM_HOST` 和 `OPENSEEK_VLLM_PORT`

### 6.6 启动模型服务

```bash
cd /workspace/flagos/FlagScale
python run.py \
  --config-path ../OpenSeek/openseek/competition/LongContext-ICL-Annotation/src \
  --config-name llm_config \
  action=run
```

### 6.7 测试服务是否正常

```bash
cd /workspace/flagos/OpenSeek/openseek/competition/LongContext-ICL-Annotation/src
python api_test.py
```

如果这里不通，先排查：

- 服务是否真的启动成功
- `port 2026` 是否一致
- 模型路径是否正确
- 权重是否完整
- `vllm` 和 CUDA 版本是否兼容

### 6.8 单任务烟测

```bash
cd /workspace/flagos/OpenSeek/openseek/competition/LongContext-ICL-Annotation/src
python main.py \
  --task_id 1 \
  --sample_limit 5 \
  --tokenizer_path /workspace/flagos/models/Qwen3-4B
```

先观察：

- 是否能持续返回结果
- 生成内容里是否出现 `<label>...</label>`
- 输出文件里 `prediction` 是否不再是 `null`

### 6.9 跑首个完整提交包

```bash
cd /workspace/flagos/OpenSeek/openseek/competition/LongContext-ICL-Annotation/src
python run_all_tasks.py \
  --tokenizer_path /workspace/flagos/models/Qwen3-4B \
  --output_dir ../outputs/first_submission
```

成功后你会得到：

- `openseek-1-v*.jsonl`
- `openseek-2-v*.jsonl`
- ...
- `openseek-8-v*.jsonl`
- `baseline_submission.zip`
- `run_summary.json`

## 7. 第一轮上 H20 时的推荐策略

不要一上来就追求“最强配置”，建议这样做：

### 第一轮目标

- 先完整跑通所有 `8` 个任务
- 先得到第一份真实可提交的 `zip`
- 先确认输出格式合法

### 第一轮不要急着改的东西

- 不要一开始就把上下文预算拉满
- 不要一开始就做复杂多轮 agent
- 不要一开始就混入额外后处理

### 第一轮最值得改的东西

- `method.py` 里的示例选择策略
- 不同任务使用不同 prompt 模板
- 对无 `<label>` 输出做重试
- 对异常长答案做清洗和回退

## 8. H20 上最常见的坑

### 8.1 不是显卡不够，而是软件栈不匹配

最常见问题通常来自：

- 驱动版本不对
- CUDA 版本不对
- `torch` 版本和 CUDA 不匹配
- `vllm` 版本和 `transformers` 版本冲突

### 8.2 不是服务起不来，而是模型路径错了

优先检查：

- `llm_config.yaml` 中 `model` 路径
- 环境变量中的 `OPENSEEK_TOKENIZER_PATH`
- 权重目录是否完整

### 8.3 不是 H20 不行，而是 prompt 太激进

如果长上下文下结果异常：

- 先缩小 `OPENSEEK_CONTEXT_BUDGET`
- 减少 `examples_limit`
- 检查输出是否经常没有 `<label>`

## 9. 对“是不是算力更好”的最终判断

对这场比赛，答案基本是：**是，H20 更好，而且是明显更合适。**

更准确地说：

- 如果你要在 `RTX 4060 Laptop 8GB` 和 `H20` 之间选一台做主力实验机，选 `H20`
- 如果你的目标是“尽快出第一版完整提交 + 稳定迭代”，选 `H20`
- 如果你的目标是“长上下文、多轮调试、反复试错”，选 `H20`

但也请记住：

- H20 解决的是算力和工程稳定性
- 真正决定榜单表现的仍然是方法设计

## 10. 从零到第一次提交的 10 步清单

下面这 10 步可以当作 H20 机器上的执行 checklist：

### Step 1. 拉代码

```bash
cd /workspace/flagos
git clone https://github.com/FlagAI-Open/OpenSeek.git
git clone https://github.com/FlagOpen/FlagScale.git
```

如果你已经有本地改过的版本，记得把改过的 `src` 目录同步过去。

### Step 2. 建环境

```bash
conda env create -f /workspace/flagos/OpenSeek/openseek/competition/LongContext-ICL-Annotation/environment_h20.yml
conda activate openseek-h20
```

如果你不用 `environment_h20.yml`，就按文档第 `6.2` 节手动安装。

### Step 3. 安装核心运行时依赖

```bash
cd /workspace/flagos

pip install -r ./FlagScale/requirements/common.txt

pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
  torch==2.9.0 \
  torchaudio==2.9.0 \
  torchvision==0.24.0

pip install "vllm @ https://resource.flagos.net/repository/flagos-pypi-hosted/packages/vllm/0.13.0%2Bfl.0.1.cu128.g72506c983/vllm-0.13.0%2Bfl.0.1.cu128.g72506c983-cp312-cp312-linux_x86_64.whl"

pip install transformers==4.57.6 openai==2.29.0
pip install -e ./FlagScale
```

如果你的 H20 机器不是 `cu128` 路线，不要硬套这套 wheel，改用服务器侧已经验证过的 `torch/vllm` 组合。

### Step 4. 校验 GPU 与 Python 依赖

```bash
nvidia-smi
python - <<'PY'
import torch, transformers, requests, openai
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu0:", torch.cuda.get_device_name(0))
print("transformers:", transformers.__version__)
print("requests:", requests.__version__)
print("openai:", openai.__version__)
PY
```

### Step 5. 下载模型

```bash
hf download Qwen/Qwen3-4B --local-dir /workspace/flagos/models/Qwen3-4B
```

### Step 6. 设置环境变量

```bash
export OPENSEEK_TOKENIZER_PATH=/workspace/flagos/models/Qwen3-4B
export OPENSEEK_MODEL_NAME=/workspace/flagos/models/Qwen3-4B
export OPENSEEK_VLLM_HOST=0.0.0.0
export OPENSEEK_VLLM_PORT=2026
export OPENSEEK_VLLM_URL=http://127.0.0.1:2026/v1/completions
export OPENSEEK_CONTEXT_BUDGET=8192
```

### Step 7. 启服务

```bash
cd /workspace/flagos/FlagScale
python run.py \
  --config-path ../OpenSeek/openseek/competition/LongContext-ICL-Annotation/src \
  --config-name llm_config \
  action=run
```

### Step 8. API 烟测

```bash
cd /workspace/flagos/OpenSeek/openseek/competition/LongContext-ICL-Annotation/src
python api_test.py
```

### Step 9. 单任务小样本测试

```bash
python main.py --task_id 1 --sample_limit 5 --tokenizer_path /workspace/flagos/models/Qwen3-4B
```

### Step 10. 跑首个完整提交

```bash
python run_all_tasks.py \
  --tokenizer_path /workspace/flagos/models/Qwen3-4B \
  --output_dir ../outputs/first_submission
```

成功后检查：

- 是否有 `8` 个 `jsonl`
- 是否有 `baseline_submission.zip`
- `prediction` 是否不是大面积 `null`
- `zip` 里是否没有目录嵌套

## 11. 建议的下一步

迁到 H20 后，按这个顺序做：

1. 启动服务
2. 跑 `api_test.py`
3. 跑 `task 1` 的 `sample_limit=5`
4. 跑全量 `run_all_tasks.py`
5. 检查 `zip` 是否符合平台要求
6. 再开始改 `method.py` 做第一轮提升

如果你更想直接走脚本化流程，也可以运行：

```bash
cd /workspace/flagos/OpenSeek/openseek/competition/LongContext-ICL-Annotation
bash bootstrap_h20.sh setup
bash bootstrap_h20.sh start
```

## 12. 参考资料

- 比赛页面：https://flagos.io/RaceDetail?id=296fmsd8&lang=cn
- OpenSeek 仓库：https://github.com/FlagAI-Open/OpenSeek
- FlagScale 仓库：https://github.com/FlagOpen/FlagScale
- NVIDIA H200 官方页面：https://www.nvidia.com/en-in/data-center/h200/
- NVIDIA H100 官方页面：https://www.nvidia.com/en-us/data-center/h100/
- NVIDIA 数据中心驱动发布说明（包含 HGX H20 平台支持条目）：https://docs.nvidia.com/datacenter/tesla/pdf/NVIDIA_Data_Center_GPU_Driver_Release_Notes_580_v4.0.pdf

说明：

- H20 的公开官方规格页不如 H100/H200 那么容易直接检索
- 上面对 H20 更适合本赛题的结论，部分来自其数据中心 Hopper 定位以及 HGX H20 平台支持信息的综合判断
