# OpenSeek大模型挑战赛决赛提交代码使用指引
**琶洲算法大赛“超越杯Openseek大模型挑战赛”:**
https://deepvision.aicompetition-pz.com/#/homeDetail?id=1933438078467272705

本仓库为决赛提交代码，集成了模型架构、数据集配置、训练与推理脚本，并提供了完整的复现指南及最终训练好的模型权重。
## 训练部分
包含VeRL的PPO训练配置文件：run_openseek_v1_ppo_step1.sh、run_openseek_v1_ppo_step2.sh
### 训练环境配置

**step1.拉取VeRL基础镜像：**  
```
docker pull verlai/verl:base-verl0.5-cu126-cudnn9.8-torch2.7.1-fa2.7.4
```

如dockerhub网络不稳定，可使用国内docker镜像源：
```
docker pull docker.m.daocloud.io/verlai/verl:base-verl0.5-cu126-cudnn9.8-torch2.7.1-fa2.8.0
```

**step2.从Modelscpoe拉取训练模型**
```
pip install modelscope # 安装modelscope库
export MODELSCOPE_CACHE='/root/workspace' # 配置modelscope的下载路径

modelscope download --model BAAI/OpenSeek-Small-v1-SFT # 下载起点模型(Actor模型)

modelscope download --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B # 下载Critic模型

```

**step3.启动镜像：**
```
docker run -it --gpus=all --shm-size="10g" -v /root/workspace/models:/workspace/ckpt docker.m.daocloud.io/verlai/verl:base-verl0.5-cu126-cudnn9.8-torch2.7.1-fa2.8.0 # 启动镜像，并将模型下载路径挂载到容器
```

**step4.进入容器内安装:**
```
git clone https://github.com/volcengine/verl && cd verl
pip3 install -e .[vllm]
```
**step5.准备数据集**

下载gsm8k数据集并进行预处理

```
cd verl/examples/data_preprocess
python gem8k.py
```
如无法访问huggingface，可以设置国内源：
```
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```
下载并预处理gsm8k训练及测试集后，环境准备完毕，开始启动训练。

### 第一阶段的PPO训练
在8卡PCI-E的GPU服务器上训练，需要设置:
```
export NCCL_P2P_DISABLE=1
```
防止训练卡死。

使用step1训练脚本：

```
bash run_openseek_v1_ppo_step1.sh
```
关键参数设置：
```
## run_openseek_v1_ppo_step1.sh

# 配置训练、测试数据集路径
data.train_files=/root/autodl-tmp/Openseek_RL/verl/data/gsm8k/train.parquet \
data.val_files=/root/autodl-tmp/Openseek_RL/verl/data/gsm8k/test.parquet \

# 配置actor、critic模型路径
actor_rollout_ref.model.path=/workspace/model/BAAI/OpenSeek-Small-v1-SFT  # 竞赛起点模型
ritic.model.path=/workspace/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B # deepseek r1 8b模型

# 权重保存为hf格式
save_contents=hf_model

# 一阶段训练15个epoch
trainer.total_epochs=15

# 实验名称
trainer.experiment_name='openseek_ppo_step1'
```

### 第二阶段的PPO训练
重置critic避免模型过拟合

使用step2训练脚本

```
bash run_openseek_v1_ppo_step2.sh
```
关键参数设置：
```
## run_openseek_v1_ppo_step1.sh

# 配置训练、测试数据集路径(与阶段一一致)
data.train_files=/root/autodl-tmp/Openseek_RL/verl/data/gsm8k/train.parquet \
data.val_files=/root/autodl-tmp/Openseek_RL/verl/data/gsm8k/test.parquet \
# 配置actor模型设置为一阶段训练保存的hf权重
actor_rollout_ref.model.path=verl/checkpoints/verl_example/openseek_ppo_step1/actor/huggingface  
# 将critic模型重置为R1-0528-Qwen3-8B初始模型
critic.model.path=/workspace/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B 
# 继续设置权重保存为hf格式
save_contents=hf_model
# 二阶段训练8个epoch
trainer.total_epochs=8
# 实验名称
trainer.experiment_name='openseek_ppo_step2'
```
训练完毕后，在verl/checkpoints/verl_example/openseek_ppo_step2/actor/huggingface的路径下找到最终版本的hf模型。

### 最终提交版本模型
已完成2阶段训练的最终提交版本模型
```
modelscope download --model ccabcca06/Openseek-small-V1-PPO-ccabcca06 
```
## 推理评估部分
评估源码基于 https://github.com/FlagAI-Open/OpenSeek/tree/main/evaluation/qwen_eval 进行修改，根目录：qwen_eval。

### 环境配置
**step1.安装latex2sympy**
```
cd qwen_eval/latex2sympy
pip install -r requirements.txt
pip install -e .
```
**step2.安装评测依赖**
```
cd ..
pip install -r requirements.txt
```
**step3.安装math_verify**
```
pip install math_verify
```

### 配置prompt
在utils.py中配置gsm8k和amc23专用prompt

**GSM8K:**  
在system prompt进行“You are an excellent mathematics professor”的身份设定后，在gsm8k数据集评分有大幅提升
```
    'prompt_gsm8k':(
        "<|im_start|>system\nYou are an excellent mathematics professor<|im_end|>\n"
        "<|im_start|>user\nPlease response it, and just put your final answer within \\boxed{{}}.\nQuestion:\n{input}<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n"
    ),
```
**AMC23:**
测试表明设置system_prompt会大幅降低模型在amc23数据集上的性能表现，故针对amc23数据集不进行角色设定
```
    'prompt_amc':(
        "<|im_start|>system\n<|im_end|>\n"
        "<|im_start|>user\nPlease response it, and just put your final answer within \\boxed{{}}.\nQuestion:\n{input}<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n"
    ),
```

### 运行评估
amc23数据集评估
```
bash run_eval_amc.sh ccabcca06/Openseek-small-V1-PPO-ccabcca06

```
其他数据集评估
```
bash run_eval_amc.sh ccabcca06/Openseek-small-V1-PPO-ccabcca06
```
聚合评估结果
```
python evaluate_final.py --eval_path ./outputs/eval_final
```