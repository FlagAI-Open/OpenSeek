# 超长长上下文场景中LLM自动数据标注挑战赛

---

## 消息
<!-- BEGIN NEWS -->
- **[2026-01-20] `发布`：** 赛事信息已在 **Kaggle** 正式上线。详情见：[FlagOS Open Computing Global Challenge](https://www.kaggle.com/competitions/flag-os-open-computing-global-challenge).
- **[2026-01-06] `发布`：** 由 **众智 FlagOS 社区**、**北京智源人工智能研究院（BAAI）** 与 **CCF ODTC** 联合主办的综合性大赛 **FlagOS 开放计算全球挑战赛** 正式发布。详情见：  
  [FlagOS开放计算全球挑战赛- AI赛事通 | 数据算法赛](https://www.competehub.dev/zh/competitions/modelscope180)
<!-- END NEWS -->

---


## 快速开始
### 1. 环境

```bash
openai
torch
flagScale
```

### 2. 下载模型权重
```bash
hf download Qwen/Qwen3-4B --local-dir /root/.cache/modelscope/hub/models/Qwen/Qwen3-4B
# or
modelscope download --model Qwen/Qwen3-4B 
```

### 3. 长文本配置
在`Qwen3-4B/config.json`将原有配置替换为：
```json
"rope_scaling": {
    "rope_type": "yarn",
    "factor": 4.0,
    "original_max_position_embeddings": 32768
}
```

### 4. 模型部署
安装环境和下载模型
```bash
pip install -r requirements.txt
bash create_env_ascend.sh
```

请根据实际需求，配置 `llm_config.yaml` 文件，启动服务
```bash
bash start-qwen3_4_serve.sh
```

在模型服务启动后，可通过以下方式测试本地 API：
```bash
python api_test.py
```

如需停止服务，请执行：
```bash
bash stop-qwen3_4_serve.sh
```

### 5. 运行/改进基线方法（Baseline）
如需开始模型标注，启动如下命令
```bash
bash start-all-tasks_on_4_serve.sh
```

如需结束模型标注，启动如下命令
```bash
bash stop-all-tasks.sh
```

### 6. 安装环境补充说明
```sh
# 系统：比赛官方提供的云服务器 (为了加速实验 - 建议用更多的卡数，推荐用8个)
# GPU：Ascend 910C x 2 卡
# CPU：20核
# 磁盘：200G
# 内存：80G
# NPU：CANN8.3.RC2

# 检查基础环境：
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg
cat /usr/local/Ascend/ascend-toolkit/set_env.sh

# 6.1、安装环境（flagscale所需CANN、vllm、vllm-ascend版本依赖请按实际环境调整）
DIR=your-project-path/OpenSeek/openseek/competition/LongContext-ICL-Annotation/src
cd $DIR
pip install -r $DIR/requirements.txt
bash create_env_ascend.sh

# 6.2、启动服务（启动服务后需要耐心等一小段时间，让模型服务完全就绪，可用npu-smi或nputop查看状态）
DIR=your-project-path/OpenSeek/openseek/competition/LongContext-ICL-Annotation/src 
cd $DIR
# 单机多卡启动 ，若超过2张NPU，请调整llm_config.yaml的参数：tensor_parallel_size数值与ASCEND_RT_VISIBLE_DEVICES数值
VLLM_PLUGINS=fl flagscale server qwen3 --config $DIR/llm_config.yaml
# 或者
bash start-qwen3-server.sh
```

### 7. 执行命令补充说明
#### 7.1) 参数
```bash
--task_id=0，  遍历处理所有任务
--task_id=$id，(one of [1,2,3,4,5,6,7,8])，则只处理对应的一个任务
--task_step=1，则利用ICL样本提取关键操作Operation
--task_step=2，则利用Operation获取预测结果Result
--task_step=0，则首先利用ICL提取Operation，然后利用Operation获取Result
```

#### 7.2) 入口文件
```bash
main.py         总是每次同步提交一个批次，便于前台调试查看过程, 
                例如:
                    DEBUG_PRINT_STREAM=1 python main.py --task_id=5 --task_step=1
                    DEBUG_PRINT_STREAM=2 python main.py --task_id=5 --task_step=1
main_batch.py   可以每次异步并发提交多个批次，用于后台命令完成任务，与服务端张量和数据并行计算协同，加速实验进度
                例如: (不要设置DEBUG_PRINT_STREAM=1，因为多个异步并发请求，实时输出是混乱无序的)
                    REQ_CONCURRENCY_NUM=8 DEBUG_PRINT_STREAM=0 python main_batch.py --task_id=5 --task_step=1
                    DEBUG_PRINT_STREAM=2 python main_batch.py --task_id=5 --task_step=1
```

#### 7.3) 如果你采用了更优的自定义配置，只需设置以下环境变量（可按实际修改默认值）：
```bash
export SERVE_API_KEY=""
export SERVE_BASE_URL="http://localhost:2026/v1"
export SERVE_MODEL_NAME="Qwen/Qwen3-4B"
export SERVE_TOKENIZER_PATH=~/.cache/modelscope/hub/models/Qwen/Qwen3-4B
export REQ_CONCURRENCY_NUM=2
export ICL_REPEAT_N=1
#
# REQ_CONCURRENCY_NUM   默认2，客户端API并发请求数，如果卡很多，可以增加该数值，与服务端张量和数据并行计算协同，加快实验进度
# ICL_REPEAT_N          默认1，在提取Opration时对ICL的重复次数，如果卡很多，可以增加该数值，提高Opration的完整性
#
```

#### 7.4) 执行命令分析解读：需要考虑这8个任务数据kv缓存的相互干扰因素
DIR=your-project-path/OpenSeek/openseek/competition/LongContext-ICL-Annotation/src
##### 7.4.1）最佳执行命令（推荐）
在卡少时可以用a或b方式，本人`单机2卡部署成4个服务(每卡2个服务)`使用c方式 (为了在并发时让不同任务KV缓存隔离，未使用max_num_seqs>1 tensor_parallel_size>1的方案)
```bash
# (a方式) 1个服务，逐个任务串行执行，8个任务的kv缓存相互干扰较小
cd $DIR && bash start-qwen3_1_serve.sh
cd $DIR && bash start-all-tasks_on_1_serve.sh
# 注：也可以在每处理一个后，重启服务清除缓存排除干扰，但是有点麻烦与耗时。若有人感兴趣，可以尝试

# (b方式) 2个服务，两阶段分步执行（干扰比a少，速度比a快）
cd $DIR && bash start-qwen3_2_serve.sh

# 首先在第一个服务：执行第一阶段
cd $DIR && bash start-all-tasks_on_2_serve-step_1.sh
# 然后在第二个服务：启动自动循环监视，在发现第一步中出现第一个文件生成时，就自动开始第二阶段, 在第一阶段完成后，自动接管其服务用于处理第二阶段的其他任务
cd $DIR && nohup bash start-all-tasks_on_2_serve-step_2.sh > s2-wait-s1.log 2>&1 &

# (c方式) 4个服务，8个任务KV缓存在不同服务，多任务并行执行，干扰很小速度较快
cd $DIR && bash start-qwen3_4_serve.sh
cd $DIR && bash start-all-tasks_on_4_serve.sh

# (d方式) 8个服务，多个任务并行执行，因为8个任务的kv缓存在8个不同服务，没有相互干扰（最佳之最推荐）
# 仅作参考
cd $DIR && bash start-all-tasks_on_8_serves.sh
```

##### 7.4.2) 待处理问题：证明KV缓存干扰是否明显影响稳定性与准确率）
在单个服务多个任务并行执行时，其kv缓存相互干扰因素，是否会严重影响数据结果的稳定性和准确率？
若影响越小，则标注方案部署越容易。本人没有时间证明这个问题，若有人感兴趣，可以尝试证明一下。
