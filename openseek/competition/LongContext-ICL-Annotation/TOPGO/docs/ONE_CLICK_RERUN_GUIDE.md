# FlagOS OpenSeek 赛道三 - 一键重新运行指南

## 📊 已完成的优化

### 新增的任务特定提示词

| 任务 | 优化内容 | 预期提升 |
|------|----------|----------|
| 任务2 | 名词/动词计数专用提示词 | 93.8% → 98%+ |
| 任务4 | 字符串连接专用提示词 | 80.8% → 95%+ |
| 任务6 | MNLI蕴含分类专用提示词 | 98.4% → 99%+ |
| 任务7 | 阅读理解专用提示词 | 97.4% → 99%+ |

### 新增的答案提取函数

- `extract_task2_answer()` - 提取数字答案
- `extract_task4_answer()` - 提取连接字符串
- `extract_task6_answer()` - 提取Y/N答案
- `extract_task7_answer()` - 提取简洁答案
- `extract_task8_answer()` - 提取Triton代码

## 🚀 快速部署

### 步骤1: 同步代码到容器

```bash
# 在容器内执行
cd /home/TOPGO

# 拉取最新代码
git pull origin master

# 如果有冲突
git fetch origin
git reset --hard origin/master
```

### 步骤2: 确保vLLM服务运行

```bash
# 检查服务状态
curl http://localhost:8000/v1/models

# 如果没有运行，启动服务
# bash scripts/start_vllm.sh
```

### 步骤3: 重新运行优化任务

```bash
cd /home/TOPGO

# 方式1: 使用脚本运行所有任务
bash scripts/rerun_optimized_tasks.sh

# 方式2: 单独运行
cd src

# 运行任务2
python main.py --task_id 2 --max_input_length 10000 --log_path_prefix ../outputs/

# 运行任务4
python main.py --task_id 4 --max_input_length 10000 --log_path_prefix ../outputs/

# 运行任务6
python main.py --task_id 6 --max_input_length 10000 --log_path_prefix ../outputs/

# 运行任务7
python main.py --task_id 7 --max_input_length 10000 --log_path_prefix ../outputs/
```

### 步骤4: 验证结果

```bash
# 快速统计
python scripts/quick_stats.py

# 详细分析
python scripts/analyze_results.py --outputs_dir outputs/
```

## 📝 修改的文件清单

### src/method.py

**新增函数：**
```python
# 提示词构建
build_prompt_task2()  # 任务2专用
build_prompt_task6()  # 任务6专用

# 答案提取
extract_task2_answer()  # 提取数字
extract_task6_answer()  # 提取Y/N
```

**修改函数：**
```python
build_prompt()         # 支持task_id参数
count_answer()         # 支持task_id参数
annotate_ascend()      # 支持task_id参数
annotate_nvidia()      # 支持task_id参数
```

### src/main.py

**修改内容：**
- 传递task_id到提示词构建函数
- 传递task_id到标注函数
- 更新日志提示

### 新增文件

```
scripts/
├── rerun_optimized_tasks.sh  # 一键运行脚本
├── postprocess_results.py    # 后处理脚本
├── quick_stats.py            # 快速统计
└── analyze_results.py        # 详细分析

docs/
├── OPTIMIZATION_DEPLOYMENT_GUIDE.md
└── ONE_CLICK_RERUN_GUIDE.md  # 本文档
```

## ⚠️ 注意事项

1. **vLLM服务必须运行**: 确保在8000端口有vLLM服务
2. **环境变量**: 设置 `QWEN_API_BASE` 和 `QWEN_API_KEY`
3. **数据路径**: 确保数据文件在 `data/` 目录或容器标准路径
4. **输出覆盖**: 新的运行结果会生成新版本文件，不会覆盖旧文件

## 📈 预期结果

| 任务 | 优化前 | 预期优化后 | 提升 |
|------|--------|-----------|------|
| 任务2 | 93.8% | 98%+ | +4% |
| 任务4 | 80.8% | 95%+ | +14% |
| 任务6 | 98.4% | 99%+ | +1% |
| 任务7 | 97.4% | 99%+ | +2% |
| **整体** | **96.0%** | **98%+** | **+2%** |

---

**准备就绪，可以开始重新运行！**
