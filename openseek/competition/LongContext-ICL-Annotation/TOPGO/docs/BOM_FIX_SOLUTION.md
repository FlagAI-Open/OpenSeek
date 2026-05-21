# UTF-8 BOM问题解决方案

## 问题描述

在容器中运行评估脚本时出现错误：
```
[2026-04-02 10:12:45] Unexpected UTF-8 BOM (decode using utf-8-sig): line 1 column 1 (char 0)
Traceback (most recent call last):
  File "/root/track3/judge_track3.py", line 273, in evaluate
    em_score = evaluator.exact_match(file, gt_file)
  File "/root/track3/judge_track3.py", line 107, in exact_match
    results = [json.loads(line) for line in f]
  File "/root/track3/judge_track3.py", line 107, in <listcomp>
    results = [json.loads(line) for line in f]
  File "/usr/lib/python3.10/json/__init__.py", line 335, in loads
    raise JSONDecodeError("Unexpected UTF-8 BOM (decode using utf-8-sig)", 
json.decoder.JSONDecodeError: Unexpected UTF-8 BOM (decode using utf-8-sig): line 1 column 1 (char 0)
```

## 根本原因

UTF-8 BOM（字节顺序标记）是文件开头的3个字节：`EF BB BF`。某些编辑器（如Windows记事本）会自动添加BOM。`json.loads()`函数无法正确处理BOM，导致解析失败。

## 解决方案

### 方案一：修复输出文件（推荐）

在运行评估前，先清理所有JSONL文件的BOM：

#### 简单命令（在容器中执行）：
```bash
# 进入项目目录
cd /home/topgo-openseek

# 修复所有JSONL文件的BOM
python3 -c "
import os, json, glob
for f in glob.glob('outputs/*.jsonl'):
    try:
        with open(f, 'r', encoding='utf-8-sig') as infile:
            content = infile.read()
        with open(f, 'w', encoding='utf-8') as outfile:
            outfile.write(content)
        print(f'✅ {os.path.basename(f)}: BOM已移除')
    except Exception as e:
        print(f'❌ {os.path.basename(f)}: {str(e)[:50]}')
"

# 验证修复
python3 -c "
import glob
for f in glob.glob('outputs/*.jsonl'):
    with open(f, 'rb') as fp:
        has_bom = fp.read(3) == b'\\xef\\xbb\\xbf'
    print(f'{\"✅\" if not has_bom else \"❌\"} {os.path.basename(f)}: {\"无BOM\" if not has_bom else \"有BOM\"}')
"
```

#### 完整修复脚本：
```bash
# 在容器中执行以下完整脚本
cd /home/topgo-openseek
cat > fix_bom.py << 'PYEOF'
import os, json, glob
for f in glob.glob('outputs/*.jsonl'):
    with open(f, 'r', encoding='utf-8-sig') as infile:
        lines = [json.dumps(json.loads(l.strip()), ensure_ascii=False) for l in infile if l.strip()]
    with open(f, 'w', encoding='utf-8') as outfile:
        outfile.write('\\n'.join(lines) + '\\n')
    print(f'修复完成: {os.path.basename(f)} ({len(lines)}行)')
PYEOF
python3 fix_bom.py
```

### 方案二：修改评估脚本

如果无法修改输出文件，可以修改评估脚本：

```bash
# 备份原脚本
cp /root/track3/judge_track3.py /root/track3/judge_track3.py.bak

# 修改第107行，添加utf-8-sig编码
sed -i "107s/with open(file, 'r') as f:/with open(file, 'r', encoding='utf-8-sig') as f:/" /root/track3/judge_track3.py

# 或者使用Python修改
python3 -c "
import sys
with open('/root/track3/judge_track3.py', 'r') as f:
    lines = f.readlines()
lines[106] = \"        with open(file, 'r', encoding='utf-8-sig') as f:\\n\"
with open('/root/track3/judge_track3.py', 'w') as f:
    f.writelines(lines)
print('✅ 已修复judge_track3.py')
"
```

### 方案三：创建包装器脚本

创建一个包装器脚本，在调用评估前先处理BOM：

```bash
cat > /tmp/run_judge_fixed.sh << 'EOF'
#!/bin/bash
# 包装器脚本，修复BOM问题后运行评估

PRED_FILE=$1
GT_FILE=$2

# 临时修复文件
TEMP_FILE=$(mktemp)

# 移除BOM并修复JSON格式
python3 -c "
import json
with open('$PRED_FILE', 'r', encoding='utf-8-sig') as f:
    lines = [json.dumps(json.loads(l.strip()), ensure_ascii=False) for l in f if l.strip()]
with open('$TEMP_FILE', 'w', encoding='utf-8') as f:
    f.write('\\n'.join(lines) + '\\n')
"

# 运行评估
python /root/track3/judge_track3.py "$TEMP_FILE" "$GT_FILE"

# 清理临时文件
rm -f "$TEMP_FILE"
EOF

chmod +x /tmp/run_judge_fixed.sh
/tmp/run_judge_fixed.sh <预测文件> <真实文件>
```

## 验证修复

修复后，使用以下命令验证：

```bash
# 检查BOM
head -c 3 outputs/openseek-*.jsonl | xxd

# 或使用Python
python3 -c "
import glob
for f in glob.glob('outputs/*.jsonl'):
    with open(f, 'rb') as fp:
        has_bom = fp.read(3) == b'\\xef\\xbb\\xbf'
    print(f'{\"✅ 无BOM\" if not has_bom else \"❌ 有BOM\"}: {f}')
"
```

## 预防措施

为了避免未来再次出现BOM问题：

1. **在代码中指定编码**：
```python
# 读取时使用utf-8-sig（自动处理BOM）
with open('file.jsonl', 'r', encoding='utf-8-sig') as f:
    data = [json.loads(line) for line in f]

# 写入时使用utf-8（无BOM）
with open('file.jsonl', 'w', encoding='utf-8') as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + '\\n')
```

2. **在生成JSONL文件时避免BOM**：
```python
import codecs

# 错误的方式（可能产生BOM）
with open('output.jsonl', 'w', encoding='utf-8') as f:
    f.write('\\uFEFF')  # 这就是BOM
    # ... 写入内容

# 正确的方式
with open('output.jsonl', 'w', encoding='utf-8') as f:
    # 不写入BOM
    # ... 写入内容
```

3. **使用专门的BOM处理函数**：
```python
def read_jsonl_without_bom(filepath):
    """安全读取JSONL文件，自动处理BOM"""
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def write_jsonl_without_bom(filepath, data):
    """安全写入JSONL文件，确保无BOM"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\\n')
```

## 快速修复命令汇总

### 针对当前问题的快速修复：

```bash
# 方法1：最简单的修复
cd /home/topgo-openseek
python3 -c "
import glob
for f in glob.glob('outputs/*.jsonl'):
    with open(f, 'rb') as fp:
        content = fp.read()
    if content.startswith(b'\\xef\\xbb\\xbf'):
        with open(f, 'wb') as fp:
            fp.write(content[3:])
        print(f'移除BOM: {f}')
"

# 方法2：修复并重新格式化
cd /home/topgo-openseek
python3 -c "
import json, glob
for f in glob.glob('outputs/*.jsonl'):
    with open(f, 'r', encoding='utf-8-sig') as infile:
        lines = [l.strip() for l in infile if l.strip()]
    fixed = []
    for line in lines:
        try:
            fixed.append(json.dumps(json.loads(line), ensure_ascii=False))
        except:
            fixed.append(line)
    with open(f, 'w', encoding='utf-8') as outfile:
        outfile.write('\\n'.join(fixed))
    print(f'修复: {f} ({len(fixed)}行)')
"
```

### 一键修复所有文件：

```bash
# 在容器中执行此命令
cd /home/topgo-openseek && \
python3 -c "
import os, json, glob
for f in glob.glob('outputs/*.jsonl'):
    try:
        # 读取（处理BOM）
        with open(f, 'r', encoding='utf-8-sig') as infile:
            lines = [l.strip() for l in infile if l.strip()]
        # 写入（无BOM）
        with open(f, 'w', encoding='utf-8') as outfile:
            outfile.write('\\n'.join(lines))
        print(f'✅ {os.path.basename(f)}')
    except Exception as e:
        print(f'❌ {os.path.basename(f)}: {e}')
" && \
echo "修复完成！现在可以运行评估脚本。"
```

## 验证评估脚本能否正常工作

修复后，测试评估脚本：

```bash
# 测试一个文件
python /root/track3/judge_track3.py outputs/openseek-1-v1.jsonl /path/to/ground_truth.jsonl

# 如果没有错误，说明修复成功
```

## 常见问题

### Q1：修复后仍然报错？
A：可能是其他JSON格式问题，检查是否有：
- 尾部逗号：`{"key": "value",}`
- 缺少引号：`{prediction: null}`
- 换行符问题

### Q2：如何批量修复多个文件？
A：使用上面的脚本即可批量处理`outputs/*.jsonl`

### Q3：如何防止BOM再次出现？
A：在代码生成文件时使用`encoding='utf-8'`，不要使用`encoding='utf-8-sig'`写入。

### Q4：Windows和Linux环境差异？
A：Windows记事本默认添加BOM，Linux/Mac通常不添加。建议在Linux环境生成文件。

## 联系支持

如果以上方法都无法解决问题：
1. 检查文件编码：`file -i your_file.jsonl`
2. 查看文件开头：`head -c 20 your_file.jsonl | xxd`
3. 使用`iconv`转换：`iconv -f utf-8-sig -t utf-8 input.jsonl > output.jsonl`
4. 手动移除BOM：`sed -i '1s/^\xEF\xBB\xBF//' your_file.jsonl`

---

*解决方案版本：1.0*
*最后更新：2026年4月2日*
*适用问题：JSONDecodeError: Unexpected UTF-8 BOM*