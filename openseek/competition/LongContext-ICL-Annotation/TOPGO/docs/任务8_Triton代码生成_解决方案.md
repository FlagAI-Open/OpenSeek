# 任务8：Triton代码生成 - 分阶段解决方案

## 问题分析

任务8要求根据自然语言描述生成完整的Triton GPU编程代码，包括：
- Triton kernel函数（`@triton.jit`装饰）
- Python wrapper函数（调用kernel）

**挑战**：
1. 代码长度长（通常200-800行）
2. 语法严格（Triton语言特性）
3. 模型难以一次性生成完整代码

## 解决方案：分阶段生成策略

### 策略1：结构化提示词分解

将任务分解为5个阶段：

#### 阶段1：理解需求（分析输入描述）
```
### 任务
分析以下Triton kernel的自然语言描述，提取：
1. kernel名称和功能
2. 输入参数列表
3. 输出参数
4. 核心计算逻辑
5. 特殊约束（BLOCK_SIZE、边界处理等）

输入描述：
{自然语言描述}

请以结构化格式输出分析结果。
```

#### 阶段2：生成kernel签名
```
### 任务
基于以下分析结果，生成Triton kernel的函数签名：

{阶段1的分析结果}

要求：
1. 使用 @triton.jit 装饰器
2. 参数类型正确
3. 包含所有必要的 constexpr 参数

输出格式：
```python
@triton.jit
def kernel_name(...):
    pass
```
```

#### 阶段3：生成核心计算逻辑
```
### 任务
为以下Triton kernel生成核心计算逻辑：

{kernel签名}

输入描述：
{自然语言描述}

要求：
1. 正确使用 tl.load 和 tl.store
2. 处理边界条件（mask）
3. 实现核心算法
```

#### 阶段4：生成wrapper函数
```
### 任务
为以下Triton kernel生成Python wrapper函数：

{完整的kernel代码}

要求：
1. 计算grid维度
2. 处理输入张量
3. 分配输出张量
4. 调用kernel

输出格式：
```python
def wrapper_name(...):
    # 实现代码
    return output
```
```

#### 阶段5：代码整合与验证
```
### 任务
整合以下代码片段，生成完整的可执行代码：

Kernel:
{kernel代码}

Wrapper:
{wrapper代码}

要求：
1. 添加必要的import语句
2. 确保代码格式正确
3. 检查语法一致性
```

### 策略2：Few-Shot学习增强

提供完整的示例代码作为参考：

```python
# 示例1：简单kernel（KL散度）
# 示例2：中等kernel（反量化）
# 示例3：复杂kernel（Attention）
```

### 策略3：分步生成与合并

在容器内实现：

```bash
# 步骤1：生成kernel部分
python generate_stage1.py --task 8 --stage kernel

# 步骤2：生成wrapper部分  
python generate_stage1.py --task 8 --stage wrapper

# 步骤3：合并代码
python merge_code.py --task 8
```

## 实施计划

### 方案A：修改提示词（推荐）

修改 `method.py` 中的 `build_prompt` 函数，针对任务8使用特殊的分阶段提示词：

```python
def build_prompt_for_triton(task_description: str, text2annotate: str) -> str:
    """
    为Triton代码生成任务构建分阶段提示词
    """
    prompt = f"""
你是一个专业的Triton GPU编程专家。你的任务是根据自然语言描述生成完整的Triton kernel代码。

### 关键要求
1. 生成完整的、可执行的Python代码
2. 包含所有必要的import语句
3. 使用 @triton.jit 装饰kernel函数
4. 提供完整的Python wrapper函数
5. 处理所有边界条件

### 示例代码结构
```python
import torch
import triton
import triton.language as tl

@triton.jit
def kernel_name(...):
    # 1. 获取程序ID
    pid = tl.program_id(axis=0)
    
    # 2. 计算偏移量
    offsets = ...
    
    # 3. 加载数据
    x = tl.load(..., mask=...)
    
    # 4. 计算
    output = ...
    
    # 5. 存储结果
    tl.store(..., mask=...)

def wrapper_function(...):
    # 1. 准备输出张量
    output = torch.empty_like(...)
    
    # 2. 计算grid
    grid = lambda meta: (...)
    
    # 3. 启动kernel
    kernel_name[grid](...)
    
    return output
```

### 任务描述
{task_description}

### 输入
{text2annotate}

### 输出要求
将完整的代码包裹在 <label> 标签中，格式如下：
<label>
```python
# 完整的代码
```
</label>

请生成完整的Triton代码：
"""
    return prompt
```

### 方案B：使用外部工具（备选）

如果Qwen3-4B无法生成复杂代码，可以考虑：

1. **调用代码生成API**：如Codex、StarCoder
2. **使用模板填充**：基于示例代码模板
3. **人工审核**：生成草稿后人工修正

## 容器内实施步骤

```bash
# 在容器内执行

# 1. 停止当前所有任务
pkill -f "python.*main.py"

# 2. 修改method.py，添加Triton专用提示词
# （需要先在本地修改，然后推送到Gitee）

# 3. 重新运行任务8
cd /home/topgo-openseek
export QWEN_API_BASE="http://localhost:8000/v1/"
export QWEN_API_KEY="EMPTY"

cd src
python main.py --task_id 8 --max_input_length 20000 --log_path_prefix ../outputs/

# 4. 如果仍然失败，使用简化策略：
# 只生成kernel部分，wrapper部分使用模板
```

## 预期效果

通过分阶段生成策略，预期可以：
1. 将成功率从 0% 提升到 30-50%
2. 生成至少部分正确的代码结构
3. 减少语法错误和格式问题

## 后续优化方向

1. **模型升级**：使用更大的模型（如Qwen2.5-7B或14B）
2. **Fine-tuning**：在Triton代码数据集上微调
3. **后处理**：使用语法检查器修正生成的代码
4. **集成开发**：结合代码检索和模板匹配
