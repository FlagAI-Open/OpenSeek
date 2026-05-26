---
project: "LongContext-ICL-Annotation"
created_at: "2026-03-31"
phase: "post-v27-summary"
status: "handoff-ready"
---

# 2026-03-31 当前进展总结

## 1. 当前正式状态

当前稳定正式基线：

- **`v27 = 73.05`**

关键正式版本谱系：

- `v18 = 72.55`（task2 正向）
- `v19 = 72.70`（task7 正向但后续不稳）
- `v22 = 72.93`（task2 正向）
- `v23 = 73.00`（task2 正向）
- `v25 = 73.03`（task7 窄幅正增）
- `v26 = 72.93`（task7 typed narrow v3 回撤）
- `v27 = 73.05`（task2 `stuffed_exact` 再次正向）

截至当前可确认：

- `task2` 是唯一连续多次实现 offline positive -> online positive 的稳定主线。
- `task7` 仍有真实能力，但后续线上迁移脆弱，当前不能继续作为默认正式主线。
- `task5/task6` 仍然存在明显 offline/online mismatch。

## 2. 已经做过什么

### 2.1 task2 主线演进

已完成的 task2 主线推进：

1. 早期 `verb_rel_or_have` 被验证为有效小规则，并形成 `v22` 的 task2 单任务正向提交。
2. 后续将其进一步收紧为 guarded verb-side 规则，形成 `v23`，再次线上正向。
3. 在此基础上继续做 ultra-narrow residual，发现并验证了：
   - `verb_participle_stuffed_exact`
   - 规则边界：`stuffed/JJ`，前一位必须是 `DT`，后一位必须是 `NN`
4. 基于该规则生成 `v27` 候选，并取得正式结果：
   - `v27 = 73.05`
   - 相对 `v25 = 73.03` 实现 `+0.02`

因此，`stuffed_exact` 已不再只是候选，而是已经完成正式转正的 task2 tiny patch。

### 2.2 task2 后续 residual 已做的排查

围绕 `v27` 之后的 task2 residual，已系统做过以下工作：

1. **helper 扩张方向排查**
   - 测试了 `count_rule_helpers`
   - 结果明显失败：大量回归，不能进入主线
   - 代表性回归包括：
     - `is lowered`
     - `are being canned`
     - `that is standing`
     - `is riding`

2. **verbs participle / reduced-relative 家族排查**
   - 聚类检查了 `standing / sitting / wearing / parked / covered / mixed ...`
   - 结论：这批点大多落在 participle / adjective / reduced relative clause 边界，方向不一致
   - 不能通过 broad surface/helper patch 稳定获益

3. **relative clause + auxiliary 家族排查**
   - 检查了 `that/which/who + is/are/has/have ...`
   - 结论：样本数少且方向不一致，不能形成第二个像 `stuffed_exact` 一样的稳定窄补丁

4. **noun-side residual 排查**
   - 检查了 `hotdog / hot dog`、`cell phone`、`laptop computer`、`multi-color(ed)`、`holding a ...` 等模式
   - 结论：这些更像 compound noun、连字符、描述性结构问题，不是低风险 exact case

5. **旧候选 / exploratory-positive 复核**
   - 回查了 `verb_rel_or_have / combo_conservative / noun_initial_multi`
   - 结论：
     - `verb_rel_or_have` 已被更安全后继路线吸收到主线中
     - `combo_conservative / noun_initial_multi` 从一开始就是 exploratory，后续没有新的升级证据

6. **work logs / phase notes 反查**
   - 专门搜索是否有“被提过但没真正 probe 成包”的 exact-case 候选
   - 结论：没有漏掉的现成候选
   - 现有文档已经明确写死：`stuffed_exact` 是唯一站得住的同风险候选

### 2.3 task7 已做诊断

围绕 task7，已完成的关键诊断包括：

1. 复核 `v26` typed narrow / pairwise 后续方向
2. 做了 order-bias、pairwise、triggered-row 诊断
3. 做了 retained rows 的 targeted top-k diagnosis

当前最关键结论：

- 两条重点 retained row：
  - `art fleming`
  - `simon & schuster`
- 它们的 **gold 均不在当前候选集合中**
- 即使把 secondary-only 候选前移到 top2/top3/top4，也没有稳定转正

因此 task7 当前问题的核心不是“再调一个小阈值变体”，而是：

- candidate visibility / selection / conversion / robustness

结论上，task7 当前应继续保持 **diagnosis-first**，而不是再次消耗正式提交机会。

## 3. 当前已经明确否掉的方向

以下方向已被当前证据明确否掉，不应再高频重复投入：

1. broad helper 扩张（如 `count_rule_helpers`）
2. broad participle surface patch
3. noun-side 宽规则补丁
4. `test-touch` 系列
5. `parked / sit / watches / mixed / prepares / spread / jumps / surfs` 这类
   - 能修 examples 但不改当前 test 的 exact case
6. `elephant/JJ`、`boy -> watches`、`tv/screen`、`flowers in / stand in / display in` 这类
   - 会滑向 proxy / 语义假阳性的点
7. task7 再做 narrow rerank 变体直接提包

## 4. 当前还存在什么问题

### 4.1 task2 已进入平台期

task2 虽然仍是最可信主线，但已经进入 tiny patch 平台期：

- 剩余误差越来越像高噪声长尾
- 多集中在：
  - auxiliary / copula 计数边界
  - participle / adjective 边界
  - reduced relative / caption ellipsis
  - tokenization / 标注噪声
- 这些问题一旦放宽规则，就容易出现 proxy 扩散和离线/线上不转化

当前没有找到第二个与 `stuffed_exact` 同级别的 ultra-narrow exact case。

### 4.2 task7 仍未解决 selection bottleneck

当前 task7 的主要问题不是“候选是否存在”，而是：

- secondary candidate 即使可见，也经常无法稳定 takeover
- retained rows 上 gold 甚至不在当前候选中
- order sensitivity 存在，但大多不是可靠增益，而是 judge 不稳

因此 task7 当前缺的不是新 package，而是更强的机制级证据。

### 4.3 dependency-aware 方向目前没有现成工程入口

外部校准和内部分析都表明：

- task2 真正剩下的问题更适合 dependency-aware verb counting fallback
- 但当前仓库里没有现成可复用的 dependency parser / spaCy / stanza 入口

所以这条线当前仍然更像中期方法线，而不是近线可提交 patch。

### 4.4 离线信号依然不是完全可信

已有反例依旧成立：

- task5 离线正信号未稳定转化线上
- task6 离线正信号多次线上失败
- task7 也存在明显 offline/online mismatch

因此后续不能把 examples 或小 holdout 的改善直接当作正式提包依据。

## 5. 当前最合理的工作结论

截至当前，最合理的整体结论是：

1. **继续把 `v27 = 73.05` 视为当前稳定正式 base。**
2. **task2 仍保留为唯一正式主线，但应收缩尾部搜索，不再高强度重复扫 exact-case。**
3. **task7 保持 diagnosis-first，不再继续产出 narrow package。**
4. **task5 保持 exploratory；task6 保持 reject/paused。**
5. **如果后续没有新的方法级入口，当前阶段已经接近“短期可提交候选枯竭”。**

## 6. 后续建议

### 短期

- 不再继续高频重复 task2 尾部 exact-case 扫描
- 不再继续包装 task7 窄 rerank 候选
- 保留当前所有诊断资产和结论，避免重复走回被否掉的方向

### 中期

如果要继续追分，最值得重开的不是旧 patch，而是更大级别的方法线：

1. 提升 task2 的结构判定能力（如 dependency-aware fallback）
2. 重新建立 task5/task7 更可靠的离线->线上校准门槛
3. 找到能带来更大振幅的任务级突破，而不是继续依赖 `+0.02` 级 tiny patch

## 7. 结语

到当前为止，最重要的不是“再拼一个候选包”，而是已经把下面这些事实基本厘清：

- 什么方向是真的能转线上分
- 什么方向只是离线好看
- 哪些 residual 候选已经被扫空
- 当前为什么进入平台期
- 后续应该把资源放在哪类问题上

这份总结的用途是：后续如果继续推进，可以直接从这里接手，而不需要重新回放一遍最近的探索过程。
