环境配置

1. 下载 FlagScale 框架到目录 `LongContext-ICL-Annotation/FlagScale`，并完成对应环境安装与配置。
2. 下载 qwen3_4b 模型到目录 `/root/ascend/log/debug/Qwen3-4B`，并完成模型配置。
3. 安装 Python 依赖：

```bash
pip install -r requirements.txt
```

```bash
挑选 样例需要下载sentence-transformers/all-MiniLM-L6-v2 ， 可以使用国内镜像
```

运行入口

```bash
python assign.py
```

运行后会提示：

```
可用任务: 1 2 3 4 5 6 7 8
请输入要执行的任务编号 (例如 '1 2 3' 或 'all'):
```

- 输入 `all`：运行全部任务脚本。
- 输入任务编号：运行指定任务（可输入多个）。

复核机制（任务 4 / 7 / 8）

当首次生成完成后，再次运行对应脚本（如 `python 4.py`、`python 7.py`、`python 8.py`）会提示进入复核模式，例如：

```
All test samples are already present in openseek-7-v1.jsonl. Enter review mode now? (y/n):
```

- 输入 `y`：进入复核流程，对现有答案进行检查并按规则回写。
- 输入 `n`：跳过复核并退出。

对比实验（统一方案）

```bash
python run.py --task_id 8 --backend nvidia --log_path_prefix ../outputs/result/
```

其他长上下文尝试脚本

```bash
python rule.py
python 8-two_stage.py
```

任务5的最后提交使用了，LongContext-ICL-Annotation_2\outputs\任务5\openseek-5-v1_14.jsonl  该文件结果。会和复现结果有区别。


