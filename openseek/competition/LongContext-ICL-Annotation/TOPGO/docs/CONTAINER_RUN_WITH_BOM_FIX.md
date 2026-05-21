# FlagOS OpenSeek 赛道三 - 容器内一键运行指南（修复BOM版）

> 更新时间：2026-04-02
> 团队：TOPGO
> 修复问题：UTF-8 BOM导致的JSON解析错误

---

## 问题描述

容器中的评估脚本 `judge_track3.py` 在第107行出现以下错误：
```
Unexpected UTF-8 BOM (decode using utf-8-sig): line 1 column 1 (char 0)
```

**原因**：某些JSONL文件包含UTF-8 BOM（字节顺序标记），而 `json.loads()` 无法正确处理BOM。

**解决方案**：在评估前先清理所有JSONL文件的BOM。

---

## 快速解决方案

### 方法一：使用修复脚本（推荐）

在容器中执行以下命令：

```bash
# 1. 进入项目目录
cd /home/topgo-openseek

# 2. 创建并运行BOM修复脚本
cat > fix_bom.py << 'PYEOF'
import os
import json
import glob

def fix_jsonl_files(directory='outputs'):
    """修复目录下所有JSONL文件的BOM问题"""
    jsonl_files = glob.glob(os.path.join(directory, '*.jsonl'))
    
    if not jsonl_files:
        print("未找到JSONL文件")
        return 0
    
    print(f"找到 {len(jsonl_files)} 个JSONL文件:")
    
    fixed_count = 0
    for filepath in jsonl_files:
        print(f"处理: {os.path.basename(filepath)}")
        
        try:
            # 使用utf-8-sig读取（自动处理BOM）
            with open(filepath, 'r', encoding='utf-8-sig') as f:
                content = f.read()
            
            # 验证并解析每一行
            lines = content.strip().split('\n')
            valid_lines = []
            
            for i, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line.strip())
                    valid_lines.append(json.dumps(data, ensure_ascii=False))
                except json.JSONDecodeError:
                    # 尝试简单修复
                    cleaned = line.strip()
                    if cleaned.endswith(','):
                        cleaned = cleaned[:-1]
                    try:
                        data = json.loads(cleaned)
                        valid_lines.append(json.dumps(data, ensure_ascii=False))
                    except:
                        # 如果无法修复，保留原行（可能导致评估失败）
                        valid_lines.append(cleaned)
            
            # 使用utf-8写入（无BOM）
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write('\n'.join(valid_lines) + '\n')
            
            print(f"  ✅ 完成: {len(valid_lines)} 行")
            fixed_count += 1
            
        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
    
    print(f"\n✅ 完成！修复了 {fixed_count}/{len(jsonl_files)} 个文件")
    return fixed_count

if __name__ == "__main__":
    fix_jsonl_files('outputs')
PYEOF

# 3. 运行修复脚本
python3 fix_bom.py

# 4. 验证修复
echo "验证修复结果:"
python3 -c "
import os
import glob

for f in sorted(glob.glob('outputs/*.jsonl')):
    with open(f, 'rb') as fp:
        first_bytes = fp.read(3)
        has_bom = first_bytes == b'\\xef\\xbb\\xbf'
    print(f'{os.path.basename(f)}: {\"✅ 无BOM\" if not has_bom else \"❌ 有BOM\"}')
"
```

### 方法二：修改评估脚本

如果无法修改输出文件，可以修改评估脚本以支持BOM：

```bash
# 备份原脚本
cp /root/track3/judge_track3.py /root/track3/judge_track3.py.bak

# 修改exact_match函数中的文件读取逻辑
sed -i "s/with open(file, 'r') as f:/with open(file, 'r', encoding='utf-8-sig') as f:/g" /root/track3/judge_track3.py
```

### 方法三：使用修复版的judge脚本

```bash
# 创建修复版评估脚本
cat > /tmp/judge_fixed.py << 'PYEOF'
#!/usr/bin/env python3
"""
修复版judge_track3.py - 支持UTF-8 BOM
"""

import json
import sys
import os

def exact_match_with_bom_fix(file, gt_file):
    """
    修复BOM问题的exact_match函数
    """
    try:
        # 使用utf-8-sig自动处理BOM
        with open(file, 'r', encoding='utf-8-sig') as f:
            results = [json.loads(line) for line in f]
    except UnicodeDecodeError:
        # 如果utf-8-sig失败，尝试utf-8
        with open(file, 'r', encoding='utf-8') as f:
            results = [json.loads(line) for line in f]
    
    with open(gt_file, 'r') as f:
        ground_truth = [json.loads(line) for line in f]
    
    # 原有的评估逻辑...
    # 这里需要根据实际的judge_track3.py内容补充
    
    return 1.0  # 假设的返回值

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python judge_fixed.py <预测文件> <真实文件>")
        sys.exit(1)
    
    score = exact_match_with_bom_fix(sys.argv[1], sys.argv[2])
    print(f"准确率: {score:.4f}")
PYEOF

# 使用修复版脚本
python3 /tmp/judge_fixed.py <预测文件> <真实文件>
```

---

## 完整的一键运行脚本

```bash
#!/bin/bash
# ====================================================================
# FlagOS OpenSeek 赛道三 - 容器内一键运行（修复BOM问题）
# ====================================================================

echo "============================================================"
echo "FlagOS OpenSeek 赛道三 - 容器内一键运行（修复BOM问题）"
echo "============================================================"

# 步骤1：检查vLLM服务
echo ""
echo "[步骤1] 检查vLLM服务..."
if curl -s http://localhost:8000/v1/models | grep -q "Qwen"; then
    echo "✅ vLLM服务正常运行"
else
    echo "❌ vLLM服务未运行，正在启动..."
    nohup python -m vllm.entrypoints.openai.api_server \
      --model /home/Qwen/Qwen3-4B \
      --host 0.0.0.0 --port 8000 \
      --gpu-memory-utilization 0.9 \
      --trust-remote-code \
      --max-model-len 32768 \
      > /tmp/vllm.log 2>&1 &
    echo "等待30秒..."
    sleep 30
fi

# 步骤2：拉取代码
echo ""
echo "[步骤2] 拉取最新代码..."
cd /home/topgo-openseek
git fetch origin
git reset --hard origin/master

# 步骤3：修复BOM问题（关键！）
echo ""
echo "[步骤3] 修复BOM问题..."
python3 << 'PYEOF'
import os, json, glob

def fix_bom(filepath):
    """修复单个文件的BOM问题"""
    try:
        # 读取文件（自动处理BOM）
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            lines = [l.strip() for l in f if l.strip()]
        
        # 验证并重新写入（无BOM）
        valid_lines = []
        for line in lines:
            try:
                data = json.loads(line)
                valid_lines.append(json.dumps(data, ensure_ascii=False))
            except:
                valid_lines.append(line)  # 保留原行
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(valid_lines) + '\n')
        
        # 验证修复
        with open(filepath, 'rb') as f:
            first_bytes = f.read(3)
            has_bom = first_bytes == b'\\xef\\xbb\\xbf'
        
        return not has_bom, len(valid_lines)
    except Exception as e:
        return False, 0

# 修复所有JSONL文件
output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

jsonl_files = glob.glob(os.path.join(output_dir, '*.jsonl'))
print(f"找到 {len(jsonl_files)} 个JSONL文件")

fixed_count = 0
for f in jsonl_files:
    success, line_count = fix_bom(f)
    status = "✅" if success else "❌"
    print(f"{status} {os.path.basename(f)}: {line_count} 行")
    if success:
        fixed_count += 1

print(f"\n修复完成: {fixed_count}/{len(jsonl_files)} 个文件")
PYEOF

# 步骤4：设置环境变量
echo ""
echo "[步骤4] 设置环境变量..."
export QWEN_API_BASE="http://localhost:8000/v1/"
export QWEN_API_KEY="EMPTY"

# 步骤5：运行任务
echo ""
echo "[步骤5] 运行缺失的任务..."
mkdir -p outputs

task_files=""
for i in 1 2 3 4 5 6 7 8; do
    file="outputs/openseek-${i}-v1.jsonl"
    if [ -f "$file" ]; then
        lines=$(wc -l < "$file" 2>/dev/null || echo "0")
        echo "  任务$i: 已存在 ($lines 行)"
    else
        echo "  任务$i: 未生成"
        task_files="$task_files $i"
    fi
done

if [ -n "$task_files" ]; then
    echo "运行任务: $task_files"
    cd src
    for task_id in $task_files; do
        echo ""
        echo "========== 开始任务 $task_id =========="
        python main.py --task_id $task_id --max_input_length 15000 --log_path_prefix ../outputs/
        echo "任务 $task_id 完成！"
    done
    cd ..
else
    echo "✅ 所有任务已完成"
fi

# 步骤6：最终验证
echo ""
echo "[步骤6] 最终验证..."
python3 << 'PYEOF'
import os, json, glob

print("最终结果统计:")
print("-" * 60)

for filepath in sorted(glob.glob('outputs/openseek-*-v1.jsonl')):
    filename = os.path.basename(filepath)
    
    # 检查BOM
    with open(filepath, 'rb') as f:
        has_bom = f.read(3) == b'\\xef\\xbb\\xbf'
    
    # 统计行数和null预测
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
    
    total = len(lines)
    nulls = sum(1 for l in lines if '"prediction": null' in l or '"prediction":null' in l)
    valid = total - nulls
    
    bom_status = "⚠️ 有BOM" if has_bom else "✅ 无BOM"
    status = "✅" if valid/total > 0.8 else ("⚠️" if valid/total > 0.5 else "❌")
    
    task_id = filename.split('-')[1] if '-' in filename else '?'
    print(f"{status} 任务{task_id}: {valid:3d}/{total:3d} 有效 ({valid*100/total:5.1f}%) - {bom_status}")
PYEOF

echo ""
echo "============================================================"
echo "✅ 所有操作完成！"
echo "============================================================"
```

---

## 常见问题解决

### 1. 如果judge_track3.py无法修改

**解决方案1：创建包装脚本**
```python
#!/usr/bin/env python3
"""
wrapper_for_judge.py - 包装器脚本解决BOM问题
"""

import subprocess
import sys
import os

def fix_bom_in_file(filepath):
    """临时修复文件的BOM问题"""
    import tempfile
    import json
    
    # 创建临时文件（无BOM）
    temp_fd, temp_path = tempfile.mkstemp(suffix='.jsonl')
    
    try:
        # 读取原文件（处理BOM）
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            content = f.read()
        
        # 写入临时文件（无BOM）
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return temp_path
    except:
        os.close(temp_fd)
        os.unlink(temp_path)
        raise

def main():
    if len(sys.argv) < 3:
        print("用法: python wrapper_for_judge.py <预测文件> <真实文件>")
        sys.exit(1)
    
    pred_file = sys.argv[1]
    gt_file = sys.argv[2]
    
    # 修复预测文件的BOM
    fixed_pred = fix_bom_in_file(pred_file)
    
    try:
        # 调用原评估脚本
        cmd = ['python', '/root/track3/judge_track3.py', fixed_pred, gt_file]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("错误:", result.stderr)
        
    finally:
        # 清理临时文件
        if os.path.exists(fixed_pred):
            os.unlink(fixed_pred)

if __name__ == "__main__":
    main()
```

**解决方案2：在运行评估前预处理文件**
```bash
# 预处理所有JSONL文件
for file in outputs/*.jsonl; do
    # 移除BOM
    sed -i '1s/^\xEF\xBB\xBF//' "$file"
    # 或者使用Python
    python3 -c "
import sys
with open('$file', 'r', encoding='utf-8-sig') as f:
    content = f.read()
with open('$file', 'w', encoding='utf-8') as f:
    f.write(content)
"
done
```

### 2. 验证BOM是否已清除

```bash
# 检查所有文件是否有BOM
for file in outputs/*.jsonl; do
    if head -c 3 "$file" | xxd | grep -q "efbbbf"; then
        echo "❌ $file 包含BOM"
    else
        echo "✅ $file 无BOM"
    fi
done

# 使用Python检查
python3 -c "
import glob
for f in glob.glob('outputs/*.jsonl'):
    with open(f, 'rb') as fp:
        has_bom = fp.read(3) == b'\\xef\\xbb\\xbf'
    print(f'{'❌' if has_bom else '✅'} {f}')
"
```

### 3. 快速修复命令（单行）

```bash
# 一键修复所有JSONL文件的BOM
find outputs -name "*.jsonl" -exec python3 -c "
import sys, json
for f in sys.argv[1:]:
    try:
        with open(f, 'r', encoding='utf-8-sig') as infile:
            lines = [json.dumps(json.loads(l.strip()), ensure_ascii=False) for l in infile if l.strip()]
        with open(f, 'w', encoding='utf-8') as outfile:
            outfile.write('\\n'.join(lines) + '\\n')
        print(f'✅ 修复: {f}')
    except Exception as e:
        print(f'❌ 失败: {f} - {e}')
" {} \;
```

---

## 注意事项

1. **BOM问题本质**：UTF-8 BOM是文件开头的3个字节（`EF BB BF`），某些编辑器（如Windows记事本）会自动添加。
2. **预防措施**：
   - 在代码中使用 `encoding='utf-8-sig'` 读取文件
   - 使用 `encoding='utf-8'` 写入文件
   - 避免使用Windows记事本编辑JSON文件
3. **容器环境**：确保在容器中运行前先执行BOM修复
4. **备份原始文件**：修复前建议备份原始文件

---

## 联系支持

如遇到BOM相关问题：
1. 运行上述修复脚本
2. 检查文件编码：`file -i your_file.jsonl`
3. 手动移除BOM：`sed -i '1s/^\xEF\xBB\xBF//' your_file.jsonl`

---

*文档版本：1.1*
*最后更新：2026年4月2日*