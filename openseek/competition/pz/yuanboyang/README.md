# 决赛代码（可完整运行的代码库）

## 文件结构
project_root/
├── README.md # 使用说明（本文件）
├── requirementsverl.txt # verl 训练环境依赖
├── requirementstest.txt # 测试/评测环境依赖
├── download.py # 训练集下载处理
├── verl/ # 修改过的 verl 源码
│ └── verl/utils/reward_score/geo3k.py # reward 函数修改
│ └── verl/examples/data_preprocess/gsm8k.py # 验证集下载处理

## 1. 数据下载与处理
- 训练集下载处理：`download.py`
- 验证集下载处理：`verl/examples/data_preprocess/gsm8k.py`

## 2. 代码修改说明
### 基于 [verl](https://github.com/volcengine/verl) 源码的修改
- 主要修改点：
  - 对于数据源和prompt的修改：
  - examples/data_preprocess/gsm8k.py：
    - 将
    ```python
    import datasets
    ...
    data_source = "openai/gsm8k"
    dataset = datasets.load_dataset(data_source, "main")
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    ```
    - 修改为
    ```python
    from modelscope.msdatasets import MsDataset
    ...
    data_source = "hiyouga/geometry3k"  # 注意：这里的源地址可能是一个笔误，但加载代码本身是针对 modelscope/gsm8k 的
    train_dataset = MsDataset.load('modelscope/gsm8k', subset_name='main', split='train', trust_remote_code=True)
    test_dataset = MsDataset.load('modelscope/gsm8k', subset_name='main', split='test', trust_remote_code=True)
    ```
    - 将
    ```python
    instruction_following = 'Let\'s think step by step and output the final answer after "####".'
    question = question_raw + " " + instruction_following
    ```
    - 修改为
    ```python
    instruction_following = instruction = r'Please reason step by step,and must put your final answer within \boxed{}.Question:'
    question = instruction + " " + question_raw
    ```
  - 对于trust_remote_code=True的修改：
  - verl/model_merger/base_model_merger.py：
    - 将
    ```python
        with init_empty_weights():
            model = auto_model_class.from_config(
                self.model_config, torch_dtype=torch.bfloat16, trust_remote_code=self.config.trust_remote_code
            ) 
    ```    
    - 修改为
    ```python
        with init_empty_weights():
            model = auto_model_class.from_config(
                self.model_config, torch_dtype=torch.bfloat16, trust_remote_code=True
            ) 
    ```
  - verl/trainer/main_ppo.py：
    - 将
    ```python
    trust_remote_code = config.data.get("trust_remote_code", False)
    ```    
    - 修改为
    ```python
    trust_remote_code = True
    ```
  - verl/workers/fsdp_workers.py：
    - 将
    ```python
    trust_remote_code=trust_remote_code
    ```    
    - 修改为
    ```python
    trust_remote_code=True
    ``` 

  - 修改了 `verl/utils/reward_score/geo3k.py` 中的 reward 函数：
    - verl/utils/reward_score/geo3k.py：
    - 将
    ```python
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    ```    
    - 修改为
    ```python
    pattern = re.compile(r".*\\boxed\{.*\}.*", re.DOTALL)
    ```

### 基于 [transformers](https://github.com/huggingface/transformers) 源码的修改
- 修改文件：
  - `/root/miniconda3/envs/verl/lib/python3.10/site-packages/transformers/configuration_utils.py`
- 修改内容：
  - 将第 917 行改为：
    ```python
    json.dumps(config_dict, indent=2, sort_keys=False) + "\n"
    ```

## 3. 环境依赖
```bash
# verl 环境
pip install -r requirementsverl.txt

# 测试环境
pip install -r requirementstest.txt
```
## 4. 运行指令
```bash
nohup env PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=/usr/train3.parquet \   # 需要自己修改位置
  data.train_batch_size=264 \
  data.max_prompt_length=2048 \
  data.max_response_length=512 \
  actor_rollout_ref.model.path=/root/.cache/modelscope/hub/models/BAAI/OpenSeek-Small-v1-SFT \   # 需要自己修改位置
  actor_rollout_ref.actor.optim.lr=1e-5 \
  actor_rollout_ref.actor.ppo_mini_batch_size=72 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.name=vllm \
  +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  trainer.logger=tensorboard \
  trainer.val_before_train=True \
  trainer.n_gpus_per_node=6 \
  trainer.nnodes=1 \
  trainer.save_freq=200 \
  trainer.test_freq=10 \
  trainer.total_epochs=15 \
  data.val_files=$HOME/data/gsm8k/test.parquet \
  actor_rollout_ref.rollout.n=6 \
  > train.log 2>&1 &
```
## 5. 模型融合及评测
### 模型融合
```bash
python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /usr/checkpoints/verl_examples/gsm8k/global_step_8000/actor \
    --target_dir /usr/checkpoints/verl_examples/gsm8k/global_step_8000/actor/huggingface
```
### 评测
- 使用官方代码'/OpenSeek/evaluation/qwen_eval/sh/run_evaluate.sh'
- 以上均需要自行修改模型位置
