# Tokenizer Vocab Size = 4 问题说明

## 问题原因

如果 tokenizer 的词汇表大小显示为 4，通常是因为：

### 1. 使用了未训练的默认 tokenizer

当 `tokenizer.json` 文件不存在时，代码会创建一个默认的 BPE tokenizer：

```python
# tokenizer.py:102-109
def _create_default_tokenizer(self):
    self.tokenizer = HFTokenizer(BPE())  # 空的 BPE tokenizer，未训练
    self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
```

**未训练的 BPE tokenizer 只有特殊 tokens**：
- `<|bos|>` (token ID: 1)
- `<|eos|>` (token ID: 2)  
- `<|pad|>` (token ID: 3)
- `<|unk|>` (token ID: 0)

所以词汇表大小 = 4

### 2. Tokenizer 文件未正确加载

如果 `tokenizer.json` 文件存在但损坏或不完整，可能只包含特殊 tokens。

### 3. HuggingFace tokenizers 的 get_vocab() 行为

对于未训练的 tokenizer，`get_vocab()` 可能只返回特殊 tokens 的映射。

## 解决方案

### ✅ 方案 1：训练 tokenizer（推荐）

使用 OpenSeek 的训练脚本训练一个完整的 tokenizer：

```bash
# 从 OpenSeek 根目录运行
python -m examples.nanochat_exp.tok_train \
    --vocab-size 50257 \
    --data-dir ~/.cache/openseek_nanochat/parquet_shards
```

这会创建一个包含完整词汇表的 `tokenizer.json` 文件。

### ✅ 方案 2：使用现有的 tokenizer

如果您有现有的 `tokenizer.json` 文件，确保它位于正确的位置：

```bash
# 检查 tokenizer 文件位置
export OPENSEEK_NANOCHAT_DATA_DIR="$HOME/.cache/openseek_nanochat"
ls $OPENSEEK_NANOCHAT_DATA_DIR/tokenizer/tokenizer.json
```

### ✅ 方案 3：检查 tokenizer 是否正确加载

代码已改进，现在会：
1. 优先从 model 获取词汇表大小（最可靠）
2. 对于 BPE tokenizer，检查 merges 来计算实际词汇表大小
3. 如果检测到 vocab size <= 4，会显示警告信息

## 改进后的 get_vocab_size() 方法

新的实现会：

1. **优先从 model 获取**：
   ```python
   if hasattr(self.tokenizer, 'model'):
       if hasattr(model, 'get_vocab_size'):
           return model.get_vocab_size()
   ```

2. **检查 BPE merges**：
   ```python
   if hasattr(model, 'merges') and model.merges:
       vocab_size = 256 + len(merges) + 4  # base + merges + special tokens
   ```

3. **从最大 token ID 推断**：
   ```python
   max_token_id = max(vocab.values())
   vocab_size = max_token_id + 1
   ```

4. **显示诊断信息**：
   - 如果 vocab size <= 4，会显示警告
   - 提示用户训练 tokenizer

## 验证 tokenizer

检查 tokenizer 是否正确加载：

```python
from examples.nanochat_exp.tokenizer import get_tokenizer

tokenizer = get_tokenizer()
print(f"Vocab size: {tokenizer.get_vocab_size()}")

# 如果 vocab size > 4，说明 tokenizer 已正确加载
# 如果 vocab size = 4，说明使用的是默认 tokenizer
```

## 正常情况下的词汇表大小

- **GPT-2 风格**: 50,257 tokens
- **BERT 风格**: 30,522 tokens
- **GPT-Neo**: 50,257 tokens
- **自定义训练**: 取决于训练时设置的 `--vocab-size` 参数

## 总结

**Vocab size = 4 的原因**：
- ❌ 使用了未训练的默认 tokenizer
- ❌ Tokenizer 文件未找到或未正确加载

**解决方法**：
- ✅ 训练 tokenizer：`python -m examples.nanochat_exp.tok_train`
- ✅ 确保 tokenizer.json 文件在正确位置
- ✅ 检查环境变量 `OPENSEEK_NANOCHAT_DATA_DIR` 和 `NANOCHAT_BASE_DIR`
