# VERL 详细代码差异报告

**生成时间**: 2025-01-09  
**比较版本**:
- 原版本 (verl): `/mnt/cfs/shanhai/qiuchenhao/code/verl/verl`
- 新版本 (verl050): `/mnt/cfs/shanhai/qiuchenhao/code/verl050/verl`

因为改的太多了 真的不知道什么改了什么没改 而且本身我的verl还有很多版本 例如PPO+SFT step加权的 我都没放出来 当然如果需要我可以提供 还有actor和critic共享参数的代码 都需要改很多
很难一次性把自己改动的都说完

## 目录
1. [核心文件详细代码变化](#1-核心文件详细代码变化)
2. [新增功能的具体实现](#2-新增功能的具体实现)
3. [删除代码分析](#3-删除代码分析)
4. [配置和参数变化](#4-配置和参数变化)
5. [Bug修复详情](#5-bug修复详情)

---

## 1. 核心文件详细代码变化

### 1.1 workers/fsdp_workers.py (440行变化)

#### 1.1.1 导入变更

**删除的导入**:
```python
# 原版本
import asyncio
from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType
try:
    from torch.distributed.tensor import DTensor  # torch 2.5+
except ImportError:
    from torch.distributed._tensor import DTensor

# 还删除了以下导入
from verl.utils.device import set_expandable_segments
from verl.utils.fsdp_utils import collect_lora_params, replace_lora_wrapper
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.model import convert_weight_keys
from verl.utils.tensordict_utils import *
from verl.workers.roles.reward_model import *
```

**新增的导入**:
```python
# 新版本
from verl.workers.rollout.rollout_worker import RolloutWorker
from verl.utils.torch_dtypes import PrecisionType
```

#### 1.1.2 新增MoE参数冻结功能

**新增函数** (行108-139):
```python
def _freeze_moe_parameters(module) -> tuple[int, int]:
    """Freeze MoE-related parameters in a HF model module.
    
    Heuristics:
    - Matches parameter names containing 'router' (MoE router) or 'experts' (expert FFNs).
    - Also matches names containing 'moe' or 'shared_expert'.
    - Skips standard dense MLP 'gate_proj' to avoid over-freezing non-MoE GLU gates.
    
    Returns:
        (frozen, total): number of frozen params and total params traversed.
    """
    frozen = 0
    total = 0
    for name, param in module.named_parameters(recurse=True):
        total += 1
        lname = name.lower()
        # Avoid freezing non-MoE GLU gate projection
        if "gate_proj" in lname:
            continue
        if (
            ".router." in lname
            or lname.endswith(".router.weight")
            or ".experts." in lname
            or "experts." in lname
            or "shared_expert" in lname
            or "moe" in lname
        ):
            param.requires_grad = False
            frozen += 1
    return frozen, total
```

#### 1.1.3 ActorRolloutRefWorker类修改

**删除的视觉塔冻结逻辑** (原版本行413-425):
```python
# 原版本 - 已删除
self.use_orig_params = fsdp_config.get("use_orig_params", False)
if self.config.actor.get("freeze_vision_tower", False):
    vision_tower = get_vl_model_vision_tower(actor_module)
    if vision_tower is not None:
        vision_tower.requires_grad_(False)
        self.use_orig_params = True
        if self.rank == 0:
            print("[actor model] Vision tower is set to not trainable.")
```

**新增的MoE冻结逻辑** (新版本行425-438):
```python
# 新版本 - 新增
moe_freeze = False
try:
    moe_freeze = bool(self.config.model.override_config.get("moe_freeze_all", False))
except Exception:
    try:
        moe_freeze = bool(self.config.model.get("moe_freeze_all", False))
    except Exception:
        moe_freeze = False
if role == "actor" and moe_freeze:
    frozen, total = _freeze_moe_parameters(actor_module)
    if self.rank == 0:
        print(f"[MoE Freeze] Frozen MoE params for actor: {frozen}/{total}")
```

#### 1.1.4 RewardModelWorker类重构

**原版本RewardModelWorker** (简化版):
```python
class RewardModelWorker(Worker):
    def __init__(self, config):
        Worker.__init__(self)
        self.config = config
        # 标准初始化
        
    def _build_model(self, config):
        # 只加载标准奖励模型
        from transformers import AutoModelForSequenceClassification
        reward_model = AutoModelForSequenceClassification.from_pretrained(...)
        return reward_model
```

**新版本RewardModelWorker** (支持PRM):
```python
class RewardModelWorker(Worker, DistProfilerExtension):
    def __init__(self, config):
        Worker.__init__(self)
        # ... profiler初始化 ...
        self.prm_manager = None  # 新增：PRM管理器
        self._is_collect_rank = None  # 新增：标记是否为collect rank
        
    def _build_model(self, config):
        # 新增：支持PRM的条件分支
        if config.get("reward_manager") == "qwen_prm":
            # 只在collect rank上初始化PRM（避免重复计算）
            if self._is_collect_rank:
                print(f"[RewardModelWorker] Rank {self.rank}: Initializing QwenPRMRewardManager")
                from verl.workers.reward_manager.qwen_prm import QwenPRMRewardManager
                from verl.utils.fs import copy_to_local
                from verl.utils import hf_tokenizer
                
                # 加载tokenizer
                actor_tokenizer_path = config.model.get("input_tokenizer", config.model.path)
                local_path = copy_to_local(actor_tokenizer_path, use_shm=config.model.get("use_shm", False))
                tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
                
                # 创建PRM管理器
                reward_kwargs = dict(config.get("reward_kwargs", {}))
                self.prm_manager = QwenPRMRewardManager(
                    tokenizer=tokenizer,
                    **reward_kwargs
                )
                
                # 设置标志，跳过标准RM加载
                self._do_switch_chat_template = False
                self.tokenizer = None
                return None
            else:
                # 非collect rank不初始化PRM
                print(f"[RewardModelWorker] Rank {self.rank}: Not collect rank, skipping PRM init")
                self._do_switch_chat_template = False
                self.tokenizer = None
                return None
        
        # 标准奖励模型加载路径（未改变）
        # ...原有代码...
```

#### 1.1.5 compute_rm_score方法修改

**新版本新增PRM支持** (行1755-1780):
```python
def compute_rm_score(self, data: DataProto):
    # 新增：处理PRM计算
    if self.prm_manager is not None:
        # 非collect rank直接返回空
        if not self._is_collect_rank:
            return DataProto.from_dict(tensors={})
        
        print("[RewardModelWorker] Computing PRM rewards on GPU (collect rank)")
        try:
            result = self.prm_manager(data, return_dict=True)
            reward_tensor = result["reward_tensor"]
            
            # 重要：返回CPU tensor保持一致性
            reward_tensor = reward_tensor.to("cpu")
            
            # PRM返回token_level_scores而不是rm_scores
            return DataProto.from_dict(tensors={"token_level_scores": reward_tensor})
        except Exception as e:
            print(f"[RewardModelWorker] Error in PRM computation: {e}")
            import traceback
            traceback.print_exc()
            # 返回零作为fallback
            batch_size = data.batch["input_ids"].shape[0]
            seq_len = data.batch["input_ids"].shape[1]
            zeros = torch.zeros((batch_size, seq_len), dtype=torch.float32)
            return DataProto.from_dict(tensors={"token_level_scores": zeros})
    
    # 原有的标准RM计算逻辑
    # ...
```

### 1.2 trainer/main_ppo.py (90行变化)

#### 1.2.1 删除resource_pool相关逻辑

**原版本** (行171-181):
```python
# 删除的代码
if config.reward_model.enable_resource_pool:
    if config.reward_model.n_gpus_per_node <= 0:
        raise ValueError("config.reward_model.n_gpus_per_node must be greater than 0")
    if config.reward_model.nnodes <= 0:
        raise ValueError("config.reward_model.nnodes must be greater than 0")
    
    reward_pool = [config.reward_model.n_gpus_per_node] * config.reward_model.nnodes
    resource_pool_spec["reward_pool"] = reward_pool
```

**新版本简化**:
```python
# 直接使用global_pool
self.mapping[Role.RewardModel] = "global_pool"
```

#### 1.2.2 移除legacy_worker_impl选项

**原版本** (行196-208):
```python
# 删除的复杂分支
use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
if use_legacy_worker_impl in ["auto", "enable"]:
    if config.reward_model.strategy in {"fsdp", "fsdp2"}:
        from verl.workers.fsdp_workers import RewardModelWorker
    # ...更多分支...
elif use_legacy_worker_impl == "disable":
    from verl.workers.roles import RewardModelWorker
    print("Using new worker implementation")
else:
    raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")
```

**新版本简化**:
```python
# 直接导入，无选项
if config.reward_model.strategy in {"fsdp", "fsdp2"}:
    from verl.workers.fsdp_workers import RewardModelWorker
elif config.reward_model.strategy == "megatron":
    from verl.workers.megatron_workers import RewardModelWorker
else:
    raise NotImplementedError
```

#### 1.2.3 新增PRM特殊处理

**新版本** (行268-292):
```python
# 新增：对qwen_prm的特殊处理
if config.reward_model.get('reward_manager') == 'qwen_prm':
    print("[DEBUG] Skipping reward manager loading for qwen_prm (handled by RewardModelWorker)")
    reward_fn = None
    
    # 创建占位符val_reward_fn以启用验证
    def placeholder_val_reward_fn(data, return_dict=False):
        # 实际PRM验证由RewardModelWorker.compute_rm_score处理
        raise NotImplementedError("PRM validation should use RewardModelWorker")
    val_reward_fn = placeholder_val_reward_fn
    print("[DEBUG] Created placeholder val_reward_fn to enable PRM validation")
else:
    # 标准reward manager加载路径
    print("[DEBUG] Begin loading training reward manager")
    reward_fn = load_reward_manager(
        config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
    )
    val_reward_fn = load_reward_manager(
        config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
    )
```

#### 1.2.4 添加调试日志

**新版本增加了大量调试日志**:
```python
print("[DEBUG] Begin add_actor_rollout_worker")
actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
print("[DEBUG] Done add_actor_rollout_worker")

print("[DEBUG] Begin add_critic_worker")
self.add_critic_worker(config)
print("[DEBUG] Done add_critic_worker")

# ... 更多调试日志 ...
```

### 1.3 trainer/ppo/ray_trainer.py (129行变化)

#### 1.3.1 奖励计算统一化

**原版本**:
```python
# 分别处理rm_scores
if self.use_rm and "rm_scores" not in batch.batch.keys():
    reward_tensor = self.rm_wg.compute_rm_score(batch)
    batch = batch.union(reward_tensor)
    # 然后使用reward_fn计算最终奖励
```

**新版本** (行1031-1065):
```python
# 统一处理，支持两种输出格式
if self.use_rm:
    reward_result = self.rm_wg.compute_rm_score(batch)
    
    # 智能识别输出格式
    if "token_level_scores" in reward_result.batch:
        # PRM直接返回token_level_scores
        reward_tensor = reward_result.batch.get("token_level_scores")
        if self.config.reward_model.get("reward_manager") == "qwen_prm":
            print("[DEBUG] Using PRM token_level_scores from GPU worker")
    elif "rm_scores" in reward_result.batch:
        # 标准RM返回rm_scores，需要后处理
        batch = batch.union(reward_result)
        if self.config.reward_model.launch_reward_fn_async:
            future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
        else:
            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
    else:
        print("[WARNING] No token_level_scores or rm_scores found")
        reward_tensor = None
```

#### 1.3.2 验证阶段PRM支持

**新版本** (行567-585):
```python
# 新增PRM验证支持
if self.config.reward_model.get("reward_manager") == "qwen_prm" and self.use_rm:
    print("[Validation] Using PRM for reward computation")
    # 处理padding
    test_batch_rm_padded, rm_pad_size = pad_dataproto_to_divisor(test_batch, rm_mb_size)
    
    # 调用RewardModelWorker计算
    result_proto_padded = self.rm_wg.compute_rm_score(test_batch_rm_padded)
    result_proto = unpad_dataproto(result_proto_padded, pad_size=rm_pad_size)
    
    # 规范化输出键名
    reward_tensor = result_proto.batch.get("token_level_scores")
    if reward_tensor is None and "rm_scores" in result_proto.batch:
        reward_tensor = result_proto.batch["rm_scores"]
    result = {"reward_tensor": reward_tensor}
elif self.val_reward_fn is not None:
    # 标准验证路径
    result = self.val_reward_fn(test_batch, return_dict=True)
    reward_tensor = result["reward_tensor"]
else:
    raise ValueError("No validation reward function available.")
```

### 1.4 utils/checkpoint/fsdp_checkpoint_manager.py (25行变化)

#### 1.4.1 TRL模型兼容性修复

**原版本**:
```python
# 直接调用，可能失败
custom_object_save(module.module, storage_folder, config=module.module.config)
```

**新版本** (行398-415):
```python
# 添加异常处理
try:
    custom_object_save(module.module, storage_folder, config=module.module.config)
except (AttributeError, KeyError) as e:
    # 处理TRL的AutoModelForCausalLMWithValueHead缺少_auto_class的问题
    print(f"Warning: custom_object_save failed with {e}")
    print("Falling back to standard save_pretrained method")
    
    # 使用标准保存方法
    module.module.save_pretrained(
        storage_folder,
        is_main_process=(torch.distributed.get_rank() == 0),
        state_dict=state_dict,
        save_function=save_file,
        safe_serialization=True
    )
```

## 2. 新增功能的具体实现

### 2.1 QwenPRMRewardManager实现

**文件**: `workers/reward_manager/qwen_prm.py` (新增，41KB)

```python
@register("qwen_prm")
class QwenPRMRewardManager(AbstractRewardManager):
    """使用Qwen2.5-Math-PRM模型的过程奖励管理器"""
    
    def __init__(
        self,
        tokenizer: Any,
        # PRM模型参数
        prm_model_path: str | None = None,
        prm_aggregation: str = "mean_response",  # 或 "pooled_last"
        prm_apply_sigmoid: bool = False,
        # 可选：通过tanh将原始logits映射到有界对称范围
        use_logit_tanh: bool = False,
        tanh_tau: float = 2.0,
        tanh_rmax: float = 1.0,
        prm_micro_batch_size: int = 8,
        # 混合奖励：PRM + ACC
        prm_acc_bonus_weight: float = 0.0,
        # 反作弊参数
        stop_at_answer: bool = True,
        incorrect_answer_penalty: float = 0,
        enable_anti_hack: bool = True,
        # 长度归一化参数
        target_response_length: int = 400,
        length_bonus_weight: float = 0.05,
        # ... 更多参数
    ):
        self.tokenizer = tokenizer
        self.prm_model_path = prm_model_path
        self._init_prm_model()
        
    def _init_prm_model(self):
        """初始化PRM模型"""
        if self.prm_model_path:
            print(f"[QwenPRM] Loading model from {self.prm_model_path}")
            self.prm_model = AutoModel.from_pretrained(
                self.prm_model_path,
                trust_remote_code=True,
                torch_dtype=self._get_dtype(),
                device_map="auto"
            )
            self.prm_tokenizer = AutoTokenizer.from_pretrained(
                self.prm_model_path,
                trust_remote_code=True
            )
            
    def compute_prm_scores(self, input_ids, attention_mask):
        """计算PRM分数"""
        with torch.no_grad():
            outputs = self.prm_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits  # [batch, seq_len]
            
            if self.use_logit_tanh:
                # 应用tanh变换
                scores = self.tanh_rmax * torch.tanh(logits / self.tanh_tau)
            elif self.prm_apply_sigmoid:
                scores = torch.sigmoid(logits)
            else:
                scores = logits
                
        return scores
        
    def __call__(self, data: DataProto, return_dict: bool = False):
        """主要的奖励计算接口"""
        # 提取输入
        input_ids = data.batch["input_ids"]
        attention_mask = data.batch["attention_mask"]
        
        # 分批处理以节省内存
        all_rewards = []
        for i in range(0, len(input_ids), self.prm_micro_batch_size):
            batch_ids = input_ids[i:i+self.prm_micro_batch_size]
            batch_mask = attention_mask[i:i+self.prm_micro_batch_size]
            
            # 计算PRM分数
            prm_scores = self.compute_prm_scores(batch_ids, batch_mask)
            
            # 应用聚合策略
            if self.prm_aggregation == "pooled_last":
                rewards = self._pooled_last_aggregation(prm_scores, batch_mask)
            elif self.prm_aggregation == "mean_response":
                rewards = self._mean_response_aggregation(prm_scores, batch_mask)
            else:
                rewards = prm_scores
                
            all_rewards.append(rewards)
            
        final_rewards = torch.cat(all_rewards, dim=0)
        
        # 应用反作弊和长度归一化
        if self.enable_anti_hack:
            final_rewards = self._apply_anti_hack(final_rewards, data)
        if self.length_bonus_weight > 0:
            final_rewards = self._apply_length_bonus(final_rewards, attention_mask)
            
        if return_dict:
            return {"reward_tensor": final_rewards}
        return final_rewards
```

### 2.2 SkyworkPRMRewardManager实现

**文件**: `workers/reward_manager/skywork_prm.py` (新增，14KB)

```python
@register("skywork_prm")
class SkyworkPRMRewardManager(AbstractRewardManager):
    """Skywork-PRM奖励管理器实现"""
    
    def __init__(self, tokenizer, prm_model_path, **kwargs):
        self.tokenizer = tokenizer
        self.model = self._load_model(prm_model_path)
        
    def _load_model(self, model_path):
        """加载Skywork-PRM模型"""
        from transformers import AutoModelForSequenceClassification
        return AutoModelForSequenceClassification.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        
    def __call__(self, data, return_dict=False):
        """计算奖励分数"""
        # Skywork-PRM特定的处理逻辑
        # ...
```

## 3. 删除代码分析

### 3.1 删除的模块和文件

1. **workers/roles/reward_model.py** - 旧版奖励模型实现
2. **workers/roles/reward_model_engine/** - 整个目录，包含旧引擎
3. **workers/config/reward_model.py** - 旧配置结构
4. **utils/tensordict_utils.py** - TensorDict工具函数
5. **interactions/weather_interaction.py** - 示例交互代码

### 3.2 删除的功能

#### 3.2.1 Resource Pool管理

```python
# 删除的resource_pool相关配置
config.reward_model.enable_resource_pool  # 删除
config.reward_model.n_gpus_per_node      # 删除
config.reward_model.nnodes                # 删除
```

#### 3.2.2 视觉塔冻结功能

```python
# 删除的函数
def get_vl_model_vision_tower(vl_model_instance):
    """提取VL模型的视觉塔"""
    # 整个函数被删除
    
# 删除的配置
config.actor.freeze_vision_tower  # 删除
```

## 4. 配置和参数变化

### 4.1 新增的配置参数

#### PRM相关配置
```yaml
# 基本PRM配置
reward_model:
  reward_manager: qwen_prm  # 新增：指定使用PRM
  reward_kwargs:
    # PRM模型路径
    prm_model_path: /path/to/Qwen2.5-Math-PRM-7B
    
    # 聚合策略
    prm_aggregation: pooled_last  # mean_response, best_of_n
    prm_apply_sigmoid: false
    
    # Tanh变换参数（可选）
    use_logit_tanh: true
    tanh_tau: 2.0
    tanh_rmax: 1.0
    
    # 批处理大小
    prm_micro_batch_size: 16
    
    # 混合奖励权重
    prm_acc_bonus_weight: 0.8
    
    # 反作弊参数
    enable_anti_hack: true
    stop_at_answer: true
    incorrect_answer_penalty: -0.3
    
    # 长度控制
    target_response_length: 400
    length_bonus_weight: 0.05
    max_resp_len: 1024
```

#### MoE冻结配置
```yaml
actor_rollout_ref:
  model:
    override_config:
      moe_freeze_all: true  # 新增：冻结所有MoE参数
```

### 4.2 删除的配置参数

```yaml
# 以下配置已删除
reward_model:
  enable_resource_pool: true  # 删除
  n_gpus_per_node: 4         # 删除
  nnodes: 1                   # 删除

trainer:
  use_legacy_worker_impl: auto  # 删除

actor:
  freeze_vision_tower: false    # 删除
```

## 5. Bug修复详情

### 5.1 TensorDict版本兼容性修复

**文件**: `__init__.py`

**原版本**:
```python
import tensordict
if parse_version(tensordict.__version__) < parse_version("0.10.0"):
    # 只在旧版本应用补丁
    TensorDictBase._sync_all = _sync_all_patch
```

**新版本**:
```python
# 移除版本检查，统一应用补丁
from tensordict.base import TensorDictBase
TensorDictBase._sync_all = _sync_all_patch
```

### 5.2 Checkpoint保存兼容性修复

**问题**: TRL的`AutoModelForCausalLMWithValueHead`缺少`_auto_class`属性

**修复**:
```python
try:
    custom_object_save(module.module, storage_folder, config=module.module.config)
except (AttributeError, KeyError) as e:
    # 回退到标准保存方法
    module.module.save_pretrained(storage_folder, ...)
```

### 5.3 单Rank优化避免重复计算

**问题**: 所有rank都运行PRM导致重复计算

**修复**:
```python
if not self._is_collect_rank:
    # 非collect rank直接返回空，避免重复计算
    return DataProto.from_dict(tensors={})
```

## 6. 性能优化细节

### 6.1 PRM单Rank优化

- 只在collect rank (rank 0)上初始化PRM模型
- 其他rank跳过模型加载和计算
- 减少7倍GPU内存使用（8 GPU配置）
- 避免重复计算，提升性能

### 6.2 内存优化

- PRM使用微批处理 (`prm_micro_batch_size`)
- 支持bf16/fp16精度计算
- 智能的tensor设备管理

## 7. 迁移建议

### 从旧版本迁移到新版本

1. **更新配置文件**:
   - 移除`enable_resource_pool`相关配置
   - 移除`use_legacy_worker_impl`
   - 添加`reward_manager`配置（如使用PRM）

2. **代码更新**:
   - 如果依赖了`workers/roles/reward_model.py`，改用`workers/fsdp_workers.RewardModelWorker`
   - 移除对`freeze_vision_tower`的引用
   - 更新奖励函数接口以兼容新的reward_manager系统

3. **测试验证**:
   - 验证PRM奖励计算的正确性
   - 检查内存使用是否符合预期
   - 确认训练稳定性

---
