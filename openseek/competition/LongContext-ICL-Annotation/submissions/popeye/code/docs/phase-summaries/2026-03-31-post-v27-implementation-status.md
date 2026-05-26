---
project: "LongContext-ICL-Annotation"
created_at: "2026-03-31"
phase: "post-v27-implementation-status"
status: "implemented"
---

# Post-v27 执行状态

## 1. 已落地的实现

本轮已经把 post-v27 计划落成三类可复用产物：

1. `task2` 最后一轮 residual 收口脚本与报告
2. `task7` 冻结/资产定版脚本与报告
3. dependency-aware / structure-aware 离线 spike 脚本与报告

对应脚本入口：

- `src/run_task2_v27_residual_round.py`
- `src/freeze_task7_post_v27.py`
- `src/spike_task2_dependency_aware.py`

对应输出产物：

- `outputs/work_logs/task2_v27_residual_round_2026-03-31.json`
- `outputs/work_logs/task2_v27_residual_round_2026-03-31.md`
- `outputs/work_logs/task7_post_v27_freeze_2026-03-31.json`
- `outputs/work_logs/task7_post_v27_freeze_2026-03-31.md`
- `outputs/work_logs/task2_dependency_spike_2026-03-31.json`
- `outputs/work_logs/task2_dependency_spike_2026-03-31.md`
- `outputs/work_logs/task2_dependency_subcluster_audit_2026-03-31.json`
- `outputs/work_logs/task2_dependency_subcluster_audit_2026-03-31.md`
- `outputs/work_logs/task2_dependency_focus_bundle_2026-03-31.json`
- `outputs/work_logs/task2_dependency_focus_bundle_2026-03-31.md`

## 2. task2 最终 residual 结论

`v27 = 73.05` 已被固定为唯一正式基线。

本轮固定检查了已有 `v27` 后 residual 候选：

- `verb_be_vbg_chain`
- `count_rule_helpers`

统一门槛：

- `0 regression`
- full audit 正向
- test diff 非零且极小
- 风险轮廓不高于 `v27`

结果：

- 两个候选都失败
- 两者都存在大量 regression
- 两者 test diff 都明显过大
- 两者都已经滑向 broader helper / chain 扩张，而不是与 `stuffed_exact` 同级别的 exact case

因此本轮明确收口为：

- **`task2 residual closed`**
- **停止继续高频扫 tiny patch**
- **正式基线继续保留 `v27`**
- **主资源转向方法线 spike**

## 3. task7 冻结结论

`task7` 已正式固定为 diagnosis-first。

冻结所依赖的资产入口已经定版为：

- `task7_v26_triggered_diagnostics_2026-03-30.json`
- `task7_v26_online_mismatch_diagnosis_2026-03-30.json`
- `task7_v26_order_bias_probe_2026-03-30.json`
- `task7_v26_pairwise_probe_2026-03-30.json`
- `task7_targeted_topk_diagnosis_2026-03-31.json`

本轮冻结报告确认：

- changed test rows 已被覆盖并可追溯
- retained target 仍只有 `art fleming` / `simon & schuster`
- 两个 retained target 的 gold 都仍然不在当前 candidate set
- order-bias disagreement 仍高
- pairwise 只显示弱正，不足以支持 promotion

因此当前 `task7` 的瓶颈被重新写死为：

- **candidate generation / visibility**

而不是：

- rerank 阈值
- 排列微调
- takeover 规则

## 4. dependency-aware spike 状态

本轮 spike 明确保持：

- 只做离线原型
- 不改 `method.py`
- 不接默认推理路径
- 不引入新的 parser 依赖

当前脚本只针对三个目标簇做结构压缩实验：

- `be_vbg`
- `rel_be_vbg`
- `be_being_vbn`

保护条件：

- 对 `v27` 既有 `stuffed_exact` 保护样本零改动

当前离线结果显示：

- `be_vbg`：从 `0.1455` 提升到 `0.8022`
- `rel_be_vbg`：从 `0.0484` 提升到 `0.8871`
- `be_being_vbn`：从 `0.0` 提升到 `1.0`
- protected `v27` exact cases：`0` changed

因此当前结论是：

- **存在小范围正向且无保护样本副作用**
- **但仍只保留为 offline spike，不进入默认工程化**

### 4.1 子簇审计的进一步收紧

在 spike 之上又补了一层 subcluster audit，目的是把下一轮研究面继续压缩。

当前最值得继续保留的子簇是：

- `be_being_vbn:passive_chain`
- `rel_be_vbg:plain`
- `rel_be_vbg:directional_tail`
- `be_vbg:plain`
- `be_vbg:directional_tail`
- `be_vbg:serial_vbg`
- `be_vbg:secondary_vbg`

当前最应明确排开、不该直接混入下一轮主簇的是：

- `be_vbg:to_inf_tail`

原因很明确：

- 该簇 `improve_count = 12`
- 同时 `worse_count = 12`
- 说明它不是“稳定结构规律”，而是混合了 `be + VBG` 与 `to + VB` 的第二层计数问题

因此下一轮如果继续做 dependency-aware 研究，默认应：

1. 先保留 `plain / directional / passive_chain`
2. 再谨慎看 `serial / secondary_vbg`
3. 暂时把 `to_inf_tail` 作为单独问题，不并入第一轮方法线

### 4.2 focused research bundle 已落盘

为了让下一轮方法线不再重新挑样，本轮又补了一个 focused bundle：

- `src/build_task2_dependency_focus_bundle.py`
- `outputs/work_logs/task2_dependency_focus_bundle_2026-03-31.json`
- `outputs/work_logs/task2_dependency_focus_bundle_2026-03-31.md`

当前 bundle 已经把后续离线研究面固定成：

- focus bundle：`305` rows
- defer bundle：`28` rows

其中：

- focus bundle 保留 clean 子簇
- defer bundle 单独收纳 `to_inf_tail` 与极小 relative mixed buckets

这意味着下一轮如果继续 dependency-aware 研究，默认不需要重新做“保留哪些桶”的决策，直接从 focus bundle 开始即可。

## 5. 当前推荐动作

1. 继续把 `v27` 作为唯一正式 base，不再追加高频 residual tiny patch 扫描
2. `task7` 继续维持 diagnosis-first，只在 `gold-in-candidate` 明显提升时重开
3. dependency-aware spike 下一步只允许继续做更细的 targeted audit / sample review，不允许直接接入正式主链
