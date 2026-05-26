# LongContext-ICL-Annotation 提交说明

本目录为队伍“大力水手”在 `LongContext-ICL-Annotation` 赛道中的可复现开发代码与说明材料。

## 1. 目录内容

- `src/`: 主要方法实现、构包脚本、验证脚本。
- `data/`: 官方任务数据。
- `docs/`: 阶段总结与技术报告源文件。
- `outputs/work_logs/`: 关键实验工件、构包日志、线上 readout。
- `environment_h20.yml`: H20 环境配置。
- `requirements.txt`: 通用 Python 依赖列表。
- `bootstrap_h20.sh`: 一键化环境准备、服务启动、烟测和全量运行脚本。
- `src/README_local.md`: 本地运行流程补充说明。

## 2. 环境准备

推荐优先使用 H20 环境配置：

```bash
cd LongContext-ICL-Annotation
conda env create -f environment_h20.yml
conda activate openseek-h20
pip install -r requirements.txt
```

如果需要完整 GPU 推理环境，使用：

```bash
bash bootstrap_h20.sh setup
```

说明：

- `requirements.txt` 负责安装本项目的通用 Python 依赖。
- `torch`、`vllm` 等 GPU 运行时依赖与 CUDA 强相关，统一由 `bootstrap_h20.sh` 按目标环境安装。
- 运行本项目还需要可用的 `FlagScale` 仓库，脚本默认会从 `flagos` 根目录下查找。

## 3. 模型部署

1. 下载 `Qwen/Qwen3-4B` 到本地目录。
2. 配置好模型路径、tokenizer 路径和 vLLM 服务地址。
3. 启动服务：

```bash
bash bootstrap_h20.sh start
```

4. 做 API 验证：

```bash
bash bootstrap_h20.sh api-test
```

## 4. 推理与生成

单任务烟测：

```bash
cd src
python main.py --task_id 1 --sample_limit 5 --tokenizer_path /path/to/Qwen3-4B
```

全量生成：

```bash
cd src
python run_all_tasks.py --tokenizer_path /path/to/Qwen3-4B --output_dir ../outputs/first_submission
```

Task7 后续候选构建示例：

```bash
python src/build_task7_v48_fact2_candidate.py
python src/check_task7_v48_candidate_smoke.py
python src/write_v48_task7_online_readout.py
```

## 5. 方法与复现口径

- 所有方法改动均保留在 `src/` 的脚本化流程中。
- Task7 的高分版本采用“稳定性门 + changed-row 审计 + 超窄 factual patch”的连续提分方式。
- 少量规则增强或事实修补均通过脚本构建，不直接手工改最终提交文件。
- 所有重要构包、验证和线上结果都在 `outputs/work_logs/` 与 `docs/phase-summaries/` 中留档。

## 6. 关键提交材料

- 技术报告 PDF：见 `flagos/技术报告/技术报告-大力水手.pdf`
- 当前正式最好成绩对应的线上 readout：`docs/phase-summaries/2026-04-11-v48-task7-online-readout.md`
- 版本总账：`outputs/work_logs/official_score_ledger_2026-04-09.json`

## 7. 注意事项

- 比赛要求使用官方原始数据划分，不应修改测试样本集合。
- 若复现完整 GPU 环境，请先确认 CUDA、驱动、FlagScale 与 vLLM 版本兼容。
- 如果只需要检查代码逻辑与构包流程，不必重新生成所有历史 `outputs/` 目录内容。
- 若比赛平台对压缩包大小有限制，可优先提交源码、数据、环境配置、部署脚本和说明文档；历史 `outputs/work_logs/` 留档体积较大，属于复盘证据而非运行必需项。
