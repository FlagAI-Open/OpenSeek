# 🚀 全面优化方案 - 容器内一键执行

## ✅ 已推送到Gitee

- **优化提示词**: `src/method_optimized_all_tasks.py`
- **一键应用脚本**: `scripts/apply_all_optimizations.sh`
- **任务分析脚本**: `scripts/comprehensive_task_analysis.py`

---

## 🎯 容器内一键执行

### 方式1：一键自动执行（推荐）

```bash
# 进入容器
docker exec -it <容器名> bash

# 拉取最新代码
cd /home/topgo-openseek
git pull

# 执行一键优化脚本
chmod +x scripts/apply_all_optimizations.sh
bash scripts/apply_all_optimizations.sh

# 选择选项2：运行任务2、4、7、8（推荐）
```

### 方式2：手动分步执行

```bash
# 1. 拉取代码
cd /home/topgo-openseek
git pull

# 2. 应用优化函数到method.py
cat src/method_optimized_all_tasks.py >> src/method.py

# 3. 修改main.py（自动）
bash scripts/apply_all_optimizations.sh
# 选择选项5：仅应用优化，不运行任务

# 4. 手动验证main.py已修改
grep "build_prompt_for_task7_optimized" src/main.py

# 5. 设置环境变量
export QWEN_API_BASE="http://localhost:8000/v1/"
export QWEN_API_KEY="EMPTY"

# 6. 运行优化任务
cd src
python main.py --task_id 4 --max_input_length 20000 --log_path_prefix ../outputs/
python main.py --task_id 7 --max_input_length 15000 --log_path_prefix ../outputs/
python main.py --task_id 2 --max_input_length 15000 --log_path_prefix ../outputs/
python main.py --task_id 8 --max_input_length 20000 --log_path_prefix ../outputs/
```

---

## 📊 优化内容

### 任务2优化 - 词性标注
- **问题**: 31个null预测
- **优化**: 明确JSON格式，减少null
- **预期**: null减少到<10个

### 任务4优化 - 代码生成
- **问题**: 96个null预测（19.2%）
- **优化**: 明确代码格式，提供模板
- **预期**: null减少到<20个

### 任务7优化 - 阅读理解
- **问题**: 特殊字符"৷"
- **优化**: 明确纯英文要求
- **预期**: 消除特殊字符

### 任务8优化 - Triton代码
- **问题**: 代码功能可能不完整
- **优化**: 明确代码结构要求
- **预期**: 提高代码质量

---

## ⏱️ 预估时间

| 任务 | 样本数 | 预估时间 |
|------|--------|----------|
| 任务2 | 500 | ~2小时 |
| 任务4 | 500 | ~3小时 |
| 任务7 | 500 | ~2小时 |
| 任务8 | 166 | ~1小时 |
| **总计** | 1666 | **~8小时** |

---

## 📈 预期效果

| 项目 | 当前 | 目标 |
|------|------|------|
| **任务2 null** | 31个 | <10个 |
| **任务4 null** | 96个 | <20个 |
| **任务7 特殊字符** | 多个 | 0个 |
| **任务8 质量** | 待提升 | 高质量 |
| **总分数** | 55.98 | **62-68** |
| **排名** | 第13名 | **前8名** |

---

## 🔍 验证优化效果

```bash
# 运行后检查结果
cd /home/topgo-openseek

python scripts/comprehensive_task_analysis.py

# 或简单检查
python3 << 'PYEOF'
import json

for task_id in [2, 4, 7, 8]:
    filepath = f'outputs/openseek-{task_id}-v1.jsonl'
    with open(filepath, 'r') as f:
        lines = f.readlines()
    total = len(lines)
    nulls = sum(1 for l in lines if '"prediction": null' in l)

    if task_id == 7:
        special = sum(1 for l in lines if '৷' in l)
        print(f"任务{task_id}: null={nulls}, 特殊字符={special}")
    else:
        print(f"任务{task_id}: null={nulls}/{total}")
PYEOF
```

---

## ⚠️ 注意事项

1. **备份已自动创建** - 文件名: `*.backup_时间戳`
2. **先测试任务7** - 效果最明显
3. **检查模型** - 确保使用Qwen3-4B
4. **清理BOM** - 运行完成后执行 `python scripts/remove_bom.py`

---

## 🚨 故障排除

### 问题1：找不到优化函数
```bash
# 检查是否已添加
grep "build_prompt_for_task7_optimized" src/method.py
# 如果没有，手动添加
cat src/method_optimized_all_tasks.py >> src/method.py
```

### 问题2：main.py没有修改
```bash
# 手动查找并修改
grep -n "input_prompt = build_prompt" src/main.py
# 在找到的行前添加条件判断
```

### 问题3：仍然有特殊字符
```bash
# 确认是否使用了优化函数
grep "build_prompt_for_task7_optimized" src/main.py
# 检查调用是否生效
```

---

## 📞 快速命令

```bash
# 一键拉取并执行
cd /home/topgo-openseek && git pull && bash scripts/apply_all_optimizations.sh

# 只应用优化不运行
bash scripts/apply_all_optimizations.sh
# 然后选择选项5

# 清理BOM
python scripts/remove_bom.py

# 重新打包提交
cd outputs && tar -czf submit.tar.gz openseek-*-v1.jsonl
```

---

## 🎯 目标

**预期最终成绩**: 62-68分
**预期排名**: 前8名
**预期提升**: +6-12分

**现在就在容器内执行一键优化脚本！** 🚀
