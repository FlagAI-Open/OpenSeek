---
project: "LongContext-ICL-Annotation"
created_at: "2026-03-30"
phase: "competition-review"
status: "decision-ready"
---

# Competition Review

## 1. Context

当前不应继续按单个 patch 的节奏快速推进提交，而应先把比赛层面的事实重新对齐。

这次复盘的目标只有三个：

1. 重新确认官方评分/评测口径，区分正式分数与离线代理指标。
2. 按 task 维度整理当前真实状态，避免继续重复分析旧结论。
3. 在最新正式基线之上重排主线优先级，为下一次正式提交提供统一起点。

## 2. Scoring / evaluation re-check

### 2.1 什么是正式分数

从数据说明看，仓库中的 `test_samples` 才是正式提交对象，labels 隐藏；`examples` 主要用于理解任务与离线分析，不直接构成 leaderboard 打分。

依据：
- [data/README.md:27-42](../../data/README.md#L27-L42)

关键事实：
- 所有任务使用固定 test split 做 leaderboard evaluation。
- `examples` 是 demonstration samples，不等于正式评分集。
- 因此本地 `avg_score`、holdout、probe leaderboard 都只能作为代理信号，不能替代官方返回分数。

### 2.2 什么是离线代理指标

仓库里当前常见的离线指标包括：
- `avg_score`
- holdout seed 报告
- probe leaderboard
- delta audit 的 `fix / regression / changed_count`

这些指标的用途是：
- 帮助筛选候选
- 判断方向是否值得继续
- 但**不能**直接当作“已确认可提交”的证据

尤其需要避免两类混线：
1. 用 `examples` 上的涨分替代正式 test evidence
2. 用和当前正式基线不一致的 baseline variant 去解释 probe 增益

这正是 `task2 test-touch` 这轮已经暴露过的问题。

依据：
- [2026-03-30_work_log.md:10-126](../../outputs/work_logs/2026-03-30_work_log.md#L10-L126)

### 2.3 当前正式稳定基线是什么

这里必须更新口径：

- `v23 = 73.00` 是 **2026-03-29 当天** 的稳定正式基线
- 到 `2026-03-30` 的阶段总结里，正式稳定基线更新为：
  - `v25 = 73.03`
- 而在最新正式返回后，当前全局稳定正式基线应继续更新为：
  - **`v27 = 73.05`**

依据：
- [2026-03-29-task2-mainline-summary.md:19-25](2026-03-29-task2-mainline-summary.md#L19-L25)
- [2026-03-30-v26-task7-online-readout.md:10-16](2026-03-30-v26-task7-online-readout.md#L10-L16)
- [task2_mainline_switch_2026-03-30.md](../../outputs/work_logs/task2_mainline_switch_2026-03-30.md)

因此从现在开始，所有“下一条正式主线”的表述都应基于：

- **当前最新稳定正式 base = `v27 = 73.05`**

而不应继续默认写成 `v23` 或 `v25`。

## 3. Official version lineage and what it means

目前仓库内可追溯的关键正式版本谱系是：

- `v10 = 71.08`
- `v15 = 70.18` (`task6` 负向)
- `v17 = 70.98` (`task5` 负向)
- `v18 = 72.55` (`task2` 正向)
- `v19 = 72.70` (`task7` 正向但幅度有限)
- `v20 = 72.53` (`task6` 负向)
- `v21 = 72.60` (`task7` gated secondary 负向)
- `v22 = 72.93` (`task2` 正向)
- `v23 = 73.00` (`task2` 正向)
- `v25 = 73.03` (`task7` line，后续曾视为稳定正式 base)
- `v26 = 72.93` (`task7` typed narrow v3，回撤)
- `v27 = 73.05` (`task2 stuffed_exact`，正式再次转正并刷新稳定 base)

依据：
- [official_score_ledger_2026-03-29.md:8-30](../../outputs/work_logs/official_score_ledger_2026-03-29.md#L8-L30)
- [2026-03-30-v26-task7-online-readout.md:11-16](2026-03-30-v26-task7-online-readout.md#L11-L16)
- [final_submission_v25_task7constraintsecondary_candidate/merge_summary.json:13-23](../../outputs/final_submission_v25_task7constraintsecondary_candidate/merge_summary.json#L13-L23)
- [task2_mainline_switch_2026-03-30.md](../../outputs/work_logs/task2_mainline_switch_2026-03-30.md)

最关键的解释不是“谁最后分数最高”，而是：

- `task2` 是最近**最稳定的正迁移线**
- `task7` 有真实能力，但在线迁移**脆弱且不稳**
- `task5/task6` 已经出现明显 offline/online mismatch

## 4. Current task status table

| Task | Current state | Offline signal | Online evidence | Current label | Decision |
| --- | --- | --- | --- | --- | --- |
| task1 | 本地规则已覆盖 | 不再依赖主要离线搜索 | 无需继续提交驱动 | maintenance | 除非发现 bug，否则不再投入主力 |
| task2 | 当前最可信的低风险增益线 | strongest calibrated | `v18/v22/v23/v27` 全部正向 | mainline | 保留为下一正式提交的首选方向 |
| task3 | 本地规则已覆盖 | 不再依赖主要离线搜索 | 无需继续提交驱动 | maintenance | 除非发现 bug，否则不再投入主力 |
| task4 | 本地规则已覆盖 | 不再依赖主要离线搜索 | 无需继续提交驱动 | maintenance | 除非发现 bug，否则不再投入主力 |
| task5 | 有过离线正信号 | misleading / conditional | `v17` 负向 | exploratory | 暂不做正式主线，除非先修复离线可信度 |
| task6 | 有过离线正信号 | misleading | `v15/v20` 负向 | reject / paused | 暂停正式推进 |
| task7 | 有过一次正迁移，但后续脆弱 | conditional / narrow | `v19` 正向，`v21/v26` 负向或未确认 | diagnosis-first | 保留背景诊断，不作为下一正式主线 |
| task8 | 影响面较小 | 可做保守维护 | 历史上难推动总分跳变 | maintenance | 继续保守维护，不做主突破 |

### 4.1 task1 / task3 / task4

这三类已经被本地规则程序覆盖，不再是当前主要增分空间。

依据：
- [2026-03-28_plateau_diagnosis.md:27-42](../../outputs/work_logs/2026-03-28_plateau_diagnosis.md#L27-L42)

结论：
- 只做维护。
- 除非发现 bug，否则不应继续消耗主分析资源。

### 4.2 task2

`task2` 仍然是当前最可信的正式方向。

依据：
- [offline_online_calibration_2026-03-29.json:5-22](../../outputs/work_logs/offline_online_calibration_2026-03-29.json#L5-L22)
- [2026-03-29-task2-mainline-summary.md:21-25](2026-03-29-task2-mainline-summary.md#L21-L25)
- [task2_mainline_switch_2026-03-30.md:1-145](../../outputs/work_logs/task2_mainline_switch_2026-03-30.md#L1-L145)

关键事实：
- `task2_structured` 是当前仓库里唯一被明确标记为 `trusted / promotable` 的稳定结构化线。
- `v18 / v22 / v23 / v27` 都实现了 offline positive -> online positive。
- 这条线的强项是：
  - 单 task 替换
  - 风险低
  - 改动量小
  - 线上连续转正

当前定位应更新为：
- 在**全局正式 base = `v27`** 的前提下，`task2` 仍是下一次正式提交最可信的低风险增益方向。
- `verb_participle_stuffed_exact` 已不只是候选参考，而是已经完成正式转正的 tiny patch。
- 后续若继续做 task2 residual，应统一相对 `v27` 评估，不再相对 `v25` 或 `v23` 混写。

### 4.3 task5

`task5` 目前不应作为正式主线。

依据：
- [official_score_ledger_2026-03-29.json:24-38](../../outputs/work_logs/official_score_ledger_2026-03-29.json#L24-L38)
- [2026-03-28_plateau_diagnosis.md:117-131](../../outputs/work_logs/2026-03-28_plateau_diagnosis.md#L117-L131)

关键事实：
- `v17` 离线表现强，但线上未转化。
- 当前更像“容易在小样本上做出正信号，但不稳”。

结论：
- 降为 exploratory。
- 只有在离线评估可信度增强后，才值得重新考虑。

### 4.4 task6

`task6` 当前应视为已被线上否证的方向。

依据：
- [official_score_ledger_2026-03-29.json:54-68](../../outputs/work_logs/official_score_ledger_2026-03-29.json#L54-L68)
- [2026-03-28_plateau_diagnosis.md:122-131](../../outputs/work_logs/2026-03-28_plateau_diagnosis.md#L122-L131)

关键事实：
- `v15` 和 `v20` 都有过离线支撑，但线上失败。
- calibration 已把 `task6_structured_hybrid` 归为 `misleading / reject`。

结论：
- 暂停，不作为近期正式主线。

### 4.5 task7

`task7` 不能再作为当前下一条正式主线，但不能彻底丢掉。

依据：
- [official_score_ledger_2026-03-29.json:86-100](../../outputs/work_logs/official_score_ledger_2026-03-29.json#L86-L100)
- [2026-03-30-v26-task7-online-readout.md:70-92](2026-03-30-v26-task7-online-readout.md#L70-L92)

关键事实：
- `v19` 曾经正向，说明这条线不是完全无效。
- 但后续 `v21`、`v26` 以及 `pairwise_triggered_v1/v2` 都证明它非常脆弱。
- 最新结论已经明确：
  - `pairwise` 路线不 promotable
  - task7 应进入 `diagnosis-first`
  - 下一正式主线应切回更低风险方向

结论：
- `task7` 保留为背景诊断线。
- 近期不再消耗下一次正式提交机会。

### 4.6 task8

`task8` 值得继续保守维护，但不应充当主突破方向。

依据：
- [data/README.md:15-24](../../data/README.md#L15-L24)
- [2026-03-28_plateau_diagnosis.md:43-64](../../outputs/work_logs/2026-03-28_plateau_diagnosis.md#L43-L64)

关键事实：
- test 样本量只有 `166`，明显小于其它多数任务的 `500`。
- 因此即使修一些坏样本，也更像保底修复而非总分主杠杆。

结论：
- 继续维护，不做主线。

## 5. Reprioritized mainline

基于当前最新事实，推荐的主线优先级应重排为：

### Priority 1 — 正式主线

**task2 低风险 patching（以 `v25` 为全局正式 base 重新表述）**

理由：
- 这是当前 offline/online 校准最好的线。
- 风险轮廓最清晰。
- 连续三次 task2 单任务替换都成功转正。
- 相比再推 task7，更符合“下一次正式提交应来自更高置信低风险线”的最新结论。

### Priority 2 — 背景诊断线

**task7 diagnosis-first**

理由：
- task7 仍然有潜力，但当前问题是 conversion / selection robustness，不是“再提一个窄变体”就能解决。
- 应继续做诊断，不应当做下一次正式 mainline。

### Priority 3 — 探索线

**task5 / task6（仅在离线可信度增强后重启）**

理由：
- 它们的问题不是完全没有 signal，而是 offline signal 不可靠。
- 在没有更强验证机制前，不值得消耗正式提交机会。

### Priority 4 — 维护线

**task1 / task3 / task4 / task8**

理由：
- 不再是当前主杠杆。
- 保持稳定即可。

## 6. What we should stop repeating

下面这些事当前不应再重复做：

1. **不再把 `task7 pairwise_triggered_v1/v2` 当主线继续推进**
   - 最新 readout 已经明确其不 promotable。

2. **不再把相对错误 baseline 的 `task2` probe 增益当新增证据**
   - `test-touch` 已经证明这是会误导决策的混线。

3. **不再把单个 task 的离线小样本上涨直接解释为可提交**
   - `task5/task6` 已经给出过足够反例。

4. **不再把 `v23` 继续写成当前全局正式稳定 base**
   - 现在应该统一写成 `v25 = 73.03`。

## 7. Submission gate before the next official package

下一次正式提交前，至少应满足：

1. 候选必须明确写清它是相对哪个**当前正式基线**比较。
2. 离线证据必须和正式基线一致，不能再混 `examples baseline` 与 `test compare baseline`。
3. 若候选来自 `task2`：
   - 保持 low-risk tiny patch 范围
   - 必须有相对当前基线的新增离线证据
   - 不接受“只改 test、无新增例证支撑”的 patch
4. 若候选来自 `task7`：
   - 必须先证明不只是离线 narrow uplift
   - 还要额外证明 robustness / conversion signal

## 8. Bottom line

当前最重要的不是继续快推 patch，而是统一事实后再行动。

本轮复盘后的结论可以压成一句话：

- **当前全局正式 base 应统一为 `v25 = 73.03`；下一正式主线应回到低风险且已校准的 `task2`，而 `task7` 保留为背景诊断，不再直接消耗下一次正式提交机会。**
