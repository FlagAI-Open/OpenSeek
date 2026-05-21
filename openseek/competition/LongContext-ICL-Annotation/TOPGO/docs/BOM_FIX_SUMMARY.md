# FlagOS OpenSeek 赛道三 - BOM问题修复总结

## 问题诊断

**错误信息：**
```
[2026-04-02 10:12:45] Unexpected UTF-8 BOM (decode using utf-8-sig): line 1 column 1 (char 0)
```

**错误位置：**
- 文件：`/root/track3/judge_track3.py`
- 行号：第107行
- 原因：`json.loads(line)` 无法解析包含UTF-8 BOM的文件

**影响范围：**
- `openseek-4-v1.jsonl` 文件包含UTF-8 BOM标记（`EF BB BF`）
- 导致评估脚本无法正常解析JSON文件

---

## 解决方案

### 1. 修复文件（已完成）

**执行脚本：** `scripts/remove_bom_fixed.py`

**修复结果：**
```
✅ openseek-1-v1.jsonl: 500 行, BOM=无, 有效JSON: 500/500
✅ openseek-2-v1.jsonl: 500 行, BOM=无, 有效JSON: 500/500
✅ openseek-3-v1.jsonl: 500 行, BOM=无, 有效JSON: 500/500
✅ openseek-4-v1.jsonl: 500 行, BOM=无, 有效JSON: 500/500  ← 已修复BOM
✅ openseek-5-v1.jsonl: 500 行, BOM=无, 有效JSON: 500/500
✅ openseek-6-v1.jsonl: 500 行, BOM=无, 有效JSON: 500/500
✅ openseek-7-v1.jsonl: 500 行, BOM=无, 有效JSON: 500/500
✅ openseek-7-v2-cleaned.jsonl: 500 行, BOM=无, 有效JSON: 500/500
✅ openseek-8-v1.jsonl: 166 行, BOM=无, 有效JSON: 166/166
```

**验证结果：**
所有文件均已：
- 移除UTF-8 BOM标记
- 验证JSON格式正确
- 确保无编码问题

---

## 容器运行命令

### 方式1：快速修复（推荐）

如果容器中再次出现BOM问题，运行：

```bash
cd /home/topgo-openseek

# 创建并运行修复脚本
cat > fix_bom.py << 'PYEOF'
import os, json, glob

def fix_bom(filepath):
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        lines = [l.strip() for l in f if l.strip()]
    
    valid_lines = []
    for line in lines:
        try:
            data = json.loads(line)
            valid_lines.append(json.dumps(data, ensure_ascii=False))
        except:
            valid_lines.append(line)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(valid_lines) + '\n')

for f in glob.glob('outputs/*.jsonl'):
    fix_bom(f)
    print(f'✅ 修复: {os.path.basename(f)}')
PYEOF

python3 fix_bom.py
```

### 方式2：使用完整脚本

```bash
cd /home/topgo-openseek
git pull origin master
python3 scripts/remove_bom_fixed.py
```

### 方式3：使用容器运行脚本

```bash
cd /home/topgo-openseek
bash scripts/container_run_final.sh
```

---

## 当前状态

### 排名信息
- **排名：** 第11位
- **团队：** TOPGO
- **得分：** 63.5分
- **提交时间：** 2026-04-02

### 文件状态
- ✅ 所有输出文件已生成
- ✅ 所有文件无BOM标记
- ✅ 所有JSON格式正确
- ✅ 总有效样本数：4166行

### 任务完成情况
| 任务 | 状态 | 样本数 | 说明 |
|------|------|--------|------|
| 任务1 | ✅ | 500 | 已完成 |
| 任务2 | ✅ | 500 | 已优化（名词/动词计数） |
| 任务3 | ✅ | 500 | 已完成 |
| 任务4 | ✅ | 500 | 已优化（字符串连接）+ BOM修复 |
| 任务5 | ✅ | 500 | 已完成 |
| 任务6 | ✅ | 500 | 已优化（MNLI蕴含分类） |
| 任务7 | ✅ | 500 | 已优化（阅读理解） |
| 任务8 | ✅ | 166 | 已完成（Triton代码） |

---

## 后续优化建议

### 1. 提升分数到78+（追赶第一名）

**当前得分：63.5**
**目标得分：78+**
**需要提升：+14.5分**

**优化方向：**

#### A. 任务特定优化（已实施）
- ✅ 任务2：专用名词/动词计数提示词
- ✅ 任务4：专用字符串连接提示词 + BOM修复
- ✅ 任务6：专用MNLI蕴含分类提示词
- ✅ 任务7：专用阅读理解提示词

#### B. 待优化项目
1. **重新运行优化后的任务**
   ```bash
   cd /home/topgo-openseek/src
   python main.py --task_id 2 --max_input_length 15000 --log_path_prefix ../outputs/
   python main.py --task_id 4 --max_input_length 15000 --log_path_prefix ../outputs/
   python main.py --task_id 6 --max_input_length 15000 --log_path_prefix ../outputs/
   python main.py --task_id 7 --max_input_length 15000 --log_path_prefix ../outputs/
   ```

2. **检查null预测**
   - 任务2有31个null
   - 任务6有8个null
   - 需要分析原因并优化

3. **任务4思考过程污染**
   - 需要进一步优化答案提取逻辑
   - 可能需要调整提示词

4. **任务7思考标记污染**
   - 已优化提示词
   - 需要重新运行验证效果

### 2. 防止BOM问题再次出现

**在代码中添加BOM防护：**

```python
# 在写入文件时明确指定编码
with open(filepath, 'w', encoding='utf-8') as f:  # 注意：不是utf-8-sig
    f.write(content)

# 在读取文件时自动处理BOM
with open(filepath, 'r', encoding='utf-8-sig') as f:  # 自动处理BOM
    content = f.read()
```

**预防措施：**
- ✅ 使用 `encoding='utf-8'` 写入文件（不带BOM）
- ✅ 使用 `encoding='utf-8-sig'` 读取文件（自动处理BOM）
- ✅ 避免使用Windows记事本编辑JSON文件
- ✅ 在容器中运行前执行BOM检查

---

## 提交清单

### 已推送到远程仓库
- ✅ `src/method.py` - 任务特定提示词和答案提取
- ✅ `src/main.py` - 支持任务ID传递
- ✅ `scripts/remove_bom_fixed.py` - BOM修复脚本
- ✅ `scripts/container_run_final.sh` - 容器运行脚本
- ✅ `docs/CONTAINER_RUN_WITH_BOM_FIX.md` - BOM修复文档
- ✅ `docs/ONE_CLICK_RERUN_GUIDE.md` - 一键运行指南

### 本地已修复
- ✅ `outputs/*.jsonl` - 所有输出文件已修复BOM

---

## 快速命令参考

```bash
# 1. 进入容器目录
cd /home/topgo-openseek

# 2. 拉取最新代码
git pull origin master

# 3. 修复BOM（如有需要）
python3 scripts/remove_bom_fixed.py

# 4. 检查vLLM服务
curl http://localhost:8000/v1/models

# 5. 运行优化任务
cd src
python main.py --task_id 2 --max_input_length 15000 --log_path_prefix ../outputs/
python main.py --task_id 4 --max_input_length 15000 --log_path_prefix ../outputs/
python main.py --task_id 6 --max_input_length 15000 --log_path_prefix ../outputs/
python main.py --task_id 7 --max_input_length 15000 --log_path_prefix ../outputs/

# 6. 验证结果
cd ..
python3 -c "
import glob, os
for f in sorted(glob.glob('outputs/openseek-*-v1.jsonl')):
    lines = open(f, 'r', encoding='utf-8').readlines()
    nulls = sum(1 for l in lines if 'null' in l)
    print(f'{os.path.basename(f)}: {len(lines)}行, {nulls}个null')
"

# 7. 检查BOM
python3 -c "
import glob
for f in sorted(glob.glob('outputs/*.jsonl')):
    with open(f, 'rb') as fp:
        has_bom = fp.read(3) == b'\\xef\\xbb\\xbf'
    print(f'{'❌' if has_bom else '✅'} {f}')
"
```

---

## 总结

1. ✅ **BOM问题已完全解决**
   - 所有文件已移除BOM标记
   - 评估脚本可以正常解析

2. ✅ **优化代码已部署**
   - 任务特定提示词已添加
   - 答案提取函数已优化
   - 容器运行脚本已就绪

3. ⏳ **下一步行动**
   - 在容器中重新运行优化后的任务
   - 验证null预测是否减少
   - 提交新结果，冲击更高排名

---

**文档版本：** 1.0
**最后更新：** 2026-04-02
**状态：** BOM问题已修复，准备重新运行优化任务