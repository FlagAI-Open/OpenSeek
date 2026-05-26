---
project: "LongContext-ICL-Annotation"
created_at: "2026-03-31"
phase: "post-v27-next-phase-plan"
status: "decision-ready"
---

# Post-v27 Next-Phase Plan

## 1. Context

当前正式结果已经推进到 `v27 = 73.05`，而第一名是 `78.33`。

这意味着当前任务已经不再是“先把比赛状态复盘清楚”，而是要回答一个更直接的问题：

- 在 `v27` 已成为当前稳定正式 base 之后，下一阶段怎么继续追分？

这一步的难点不在于“有没有 patch”，而在于：

- `task2` 已经连续实现 `v18 / v22 / v23 / v27` 的正式正迁移，说明它是最可信的主线
- 但 `v27` 只相对 `v25` 提升了 `+0.02`，说明 task2 主线已经进入低振幅区间
- 与第一名 `78.33` 相比，当前仍有明显差距，单靠继续堆同级 tiny patch，大概率不足以追到头部区间

因此，下一阶段的目标必须从“继续找可提交 patch”升级为：

1. 解释为什么当前已经处在平台期；
2. 区分“继续拿微正分的路线”和“寻找更大跃迁线的路线”；
3. 重新安排资源，把正式提交机会留给最可信方向，把高风险方向压回诊断或探索层。

## 2. Current official state and what it means

### 2.1 当前稳定正式 base

当前最新正式稳定基线应统一写为：

- **`v27 = 73.05`**

当前关键正式版本谱系：

- `v18 = 72.55` (`task2` 正向)
- `v19 = 72.70` (`task7` 正向但幅度有限)
- `v22 = 72.93` (`task2` 正向)
- `v23 = 73.00` (`task2` 正向)
- `v25 = 73.03` (`task7` line，窄幅正增)
- `v26 = 72.93` (`task7` typed narrow v3，回撤)
- `v27 = 73.05` (`task2 stuffed_exact`，再次正向并刷新稳定 base)

这条版本谱系的核心含义不是“谁分数最新”，而是：

- `task2` 是当前唯一连续多次实现 offline positive -> online positive 的稳定主线
- `task7` 有真实能力，但后续线上迁移脆弱，不能继续当作默认正式主线
- `task5/task6` 仍然没有摆脱 offline/online mismatch

### 2.2 与第一名差距意味着什么

当前：

- 我们：`73.05`
- 第一名：`78.33`
- 差距：`5.28`

这个差距意味着：

- 当前问题已经不是“再多拿一个 `+0.02` 就行”
- 而是必须区分两类目标：
  1. **短期目标**：继续通过 task2 这类低风险线拿微正分，避免回撤
  2. **中期目标**：重新找到能带来更大级别跃迁的方法级突破口

也就是说：

- `task2 tiny patch` 仍然有价值
- 但它不能再被误写成“足以解决榜首差距的主策略”

## 3. Why we are now in a plateau

当前的“平台期”不是抽象感觉，而是现有证据已经足够支持的结构性状态。

### 3.1 已解决任务不再是主要增分空间

以下任务已经不再是当前主杠杆：

- `task1`
- `task3`
- `task4`

原因：

- 这些任务已基本由本地规则程序覆盖，更多属于维护区而不是突破区。

`task8` 虽然不是“已解决”，但它也不太可能成为总分主杠杆：

- test samples 只有 `166`
- 其它多数任务是 `500`

因此：

- `task8` 继续保守维护是合理的
- 但它很难单独带来显著的 leaderboard 跃迁

### 3.2 高杠杆任务里，只有 task2 形成了稳定正迁移

当前真正决定上限的，仍然主要是：

- `task2`
- `task5`
- `task6`
- `task7`

但它们的状态已经明显分化：

#### task2
- 已连续在 `v18 / v22 / v23 / v27` 上实现正式正迁移
- 是当前最可信的低风险正式主线
- 但当前已进入 tiny patch 区间，振幅越来越小

#### task5
- 有过离线正信号
- 但 `v17` 已经证明线上不转化
- 当前仍应视为 exploratory，而不是正式主线

#### task6
- 有过局部离线正信号
- 但 `v15 / v20` 都已线上失败
- 当前应视为 reject / paused

#### task7
- `v19` 曾经转正，说明这条线不是假的
- 但 `v21 / v26` 和 pairwise 后续已证明它非常脆弱
- 当前只能保留 diagnosis-first，不能直接吃下一次正式提交机会

### 3.3 平台期的本质：剩下的是高噪声长尾误差

为什么 `task2` 虽然稳定，却越来越难做出大跃迁？

因为当前剩下的 residual 越来越像：

- participle / adjective 边界
- auxiliary 计数边界
- caption-style ellipsis
- tokenization / 标注噪声 / 脏点

这类问题的共同特征是：

- 能修，但覆盖面通常很小
- 规则一旦放宽就容易 proxy 扩散
- 很容易出现“离线好看、线上不转化”

因此当前平台期的更准确定义是：

- **大块可转移收益已经基本吃掉，剩下主要是高噪声长尾误差，以及尚未被可靠验证的大改法方向。**

### 3.4 另一个瓶颈：离线信号仍然不够可靠

平台期不只是“方法变难”，也包括“选方向的信号还不够准”。

已有反例已经很明确：

- `task5` 离线好看，但线上失败
- `task6` 离线好看，但线上失败
- `task7` 有一次成功，但后续 follow-up 明显不稳

这意味着：

- 当前不能把小样本 holdout 或 examples 上的涨分直接当作正式提包依据
- 任何想重启 `task5/6/7` 的路线，都必须先解决离线可信度问题

## 4. Updated task roles for the next phase

| Task | Current role | Why |
| --- | --- | --- |
| task1 | maintenance | 本地规则已覆盖，不再是主要杠杆 |
| task2 | mainline | 当前唯一连续稳定正迁移的结构化线 |
| task3 | maintenance | 本地规则已覆盖，不再是主要杠杆 |
| task4 | maintenance | 本地规则已覆盖，不再是主要杠杆 |
| task5 | exploratory | 有 signal，但 offline/online mismatch 未解决 |
| task6 | reject / paused | 已被线上多次否证 |
| task7 | diagnosis-first | 有潜力，但当前 conversion / robustness 明显脆弱 |
| task8 | maintenance | 可保守维护，但影响面较小 |

## 5. Recommended next-phase roadmap

下一阶段只保留一个推荐路线图，不列并行 wish list。

### Priority 1 — 正式主线

**继续保留 task2 为唯一可提交主线，但提高停止阈值。**

这条线仍然最适合承担“继续稳步拿正分”的任务。

但新的要求是：

- 不再默认“只要是 tiny patch 就继续做”
- 只接受同时满足以下条件的 residual 候选：
  - 相对当前正式 base `v27` 有新增离线证据
  - `0 regression`
  - test diff 极小且可解释
  - 风险轮廓与 `v22 / v23 / v27` 同级

#### task2 的停止条件

如果后续 residual 搜索出现下面任一情况，就应停止继续把主资源投入 task2 尾部：

1. 连续一轮没有找到新的 `0 regression` 候选；
2. 新候选虽然修 examples，但不改变 test；
3. 新候选开始明显滑向 broader proxy patch，而不再是 exact / ultra-narrow case；
4. 新候选的离线增益继续缩到几乎不可区分，且无法提供新增可信证据。

一旦满足这些条件，task2 仍然可以保留为“保稳主线”，但不应继续吞噬下一阶段的大部分分析资源。

### Priority 2 — 背景诊断线

**task7 diagnosis-first**

task7 仍然是当前最值得保留的方法级突破候选之一，但前提是：

- 不再直接做新的窄变体提包
- 先回答它到底有没有更大振幅空间
- 问题核心集中在：
  - selection
  - conversion
  - robustness

#### task7 的重启条件

只有当 task7 出现以下新增证据，才值得重新回到正式主线讨论：

1. 不只是离线 narrow uplift，而是出现新的 robustness signal；
2. pairwise / order / triggered-row 分析不再只是局部弱正，而是显示更稳定的 conversion 改善；
3. 能提出一个比“再调阈值、再缩触发器”更像方法升级的机制。

在这之前，task7 应继续停留在 diagnosis-first，而不是 submission-first。

### Priority 3 — 探索线

**task5 / task6 只允许在“先补离线可信度”的前提下重启。**

当前不推荐把它们重新抬回正式主线。

如果要重启，必须先解决：

- 为什么过去离线提升没有映射到正式 test
- 当前 holdout / examples 信号到底缺了什么
- 新验证方式如何比旧验证更接近线上

换句话说：

- 在没有新验证机制前，task5/6 不该再消耗正式提交机会。

### Priority 4 — 维护线

**task1 / task3 / task4 / task8**

策略保持不变：

- 维持稳定
- 只修明显 bug 或高置信坏样本
- 不作为主突破方向

## 6. Practical interpretation: two different goals

为了避免后续继续混线，下一阶段必须把两种目标明确分开：

### 目标 A：继续微升正式分数

如果目标是：

- 尽量稳地从 `73.05` 往上走
- 避免回撤

那么最合理的路线仍然是：

- **task2 residual tiny patching**

### 目标 B：重新寻找更大一级的跃迁线

如果目标是：

- 缩小与 `78.33` 的显著差距
- 不是只追 `+0.01 / +0.02`

那么真正值得投入方法级改造的，只剩：

- **task2 深化版结构突破**
- **task7 diagnosis -> mechanism breakthrough**

而不是继续在：

- task5 小调
- task6 小调
- task8 小修

这些方向上消耗主要资源。

## 8. Immediate work queue

为了让这份计划可以直接执行，下一阶段先按下面的顺序推进。

### 8.1 task2：立即执行的 residual work queue

当前 task2 不应重新发散到 broad rewrite，而应只保留一条更窄的 residual 工作队列：

1. **继续检查与 `stuffed_exact` 同级别的 ultra-narrow exact case**
   - 重点仍放在 participle / adjective-mistag 家族
   - 必须是规则边界清楚、不会向更广 proxy 扩散的 case

2. **优先排除“example-side 修复但不改 test”的候选**
   - 这类候选不应再进入正式主线队列
   - 已经被排除过的 `parked / sit / watches / mixed / prepares / spread / jumps / surfs` 一类点，不应反复回头分析

3. **只保留同风险轮廓的候选**
   - 必须满足：
     - `0 regression`
     - test diff 极小
     - full audit 正增益
     - 与 `v22 / v23 / v27` 同样属于小范围单 task 替换逻辑

4. **如果 narrow residual 队列没有新命中，就停止继续抠 task2 尾部**
   - 这时 task2 仍保留为正式主线
   - 但主资源应转去更高杠杆的 diagnosis / method 线，而不是继续硬抠第二个 tiny patch

#### 当前 residual verdict

基于现有扫描与工作日志，当前已经可以把 task2 residual 的即时判断收紧为：

- `stuffed_exact` 仍然是**唯一**站得住的同风险候选
- `test-touch` 系列已被证实相对 `v23` / `v27` 口径没有新增 example-side 证据，不应回到主线
- 其它被扫到的 exact case 主要分成两类：
  1. `parked / sit / watches / mixed / prepares / spread / jumps / surfs` 这类 **能修 examples 但不改当前 test** 的候选
  2. `elephant/JJ`、`boy -> watches`、`tv/screen`、`flowers in / stand in / display in` 这类 **会滑向 proxy / 语义假阳性** 的候选

因此当前最合理的执行结论是：

- **停止把 task2 尾部继续当作高优先级搜索区反复深挖**
- task2 仍然保留为当前唯一正式主线
- 但 task2 的下一次正式推进，默认应建立在 `v27` 已转正这一事实之上，而不是继续预设“很快还会有第二个同级 tiny patch”

换句话说：

- task2 现在进入的是“保留主线、收缩尾部搜索”的阶段，而不是“继续高强度扫描 residual”的阶段

### 8.2 task7：立即执行的 diagnosis work queue

当前 task7 不该再产出新的 promotable 变体，而应只做 diagnosis-first 队列：

1. **固定保留已完成的 mismatch / triggered-row bundle 作为诊断入口**
   - `v25` vs `v26` changed-row audit
   - triggered-row diagnosis bundle

2. **优先检查 selection bottleneck 是否仍然成立**
   - 重点看：
     - `secondary_visible_not_selected`
     - permutation disagreement
     - secondary-only unique candidate 是否存在稳定 conversion 机会

3. **继续把 pairwise / order probe 只当诊断工具，而不是候选生成工具**
   - `pairwise_triggered_v1/v2` 已经证明不能直接 promotable
   - 后续若继续做 probe，目的应是确认机制瓶颈，而不是再包装一个阈值变体

4. **只有出现 robustness 级新增证据，才允许 task7 重新回到正式主线讨论**
   - 否则 task7 一律停留在背景诊断层

#### 当前 diagnosis verdict

基于当前 `v26` 诊断 bundle，task7 的即时结论已经可以进一步收紧：

- `selection bottleneck` 仍然成立，而且是当前最主要瓶颈
- 现有证据更接近“secondary candidate 看得见，但转不成稳定正确选择”，而不是“secondary candidate 根本不存在”
- 因此 task7 当前的问题核心仍是 `selection / conversion / robustness`，不是继续产出另一个 narrow candidate

支撑证据有三层：

1. **triggered rows 很少，而且大多数是 secondary visible but not selected**
   - `v26` 相对 `v25` 一共改了 `194` 个 task7 test 行
   - 其中真正进入 triggered diagnosis bundle 的只有 `7` 行
   - 这 `7` 行里有 `6/7` 被标成 `secondary_visible_not_selected`
   - 说明瓶颈首先不是触发面太大，而是触发后 secondary 候选即使进入 judge 可见集合，也大多无法完成 takeover

2. **pairwise conversion 很弱，不支持把 secondary 候选直接升格为主路线**
   - 当前 pairwise probe 一共只得到 `16` 个比较
   - 只有 `1` 个 `secondary_strong_win`
   - 只有 `2` 个 `secondary_split_win`
   - 平均 secondary preference rate 只有 `0.2083`
   - 这说明 secondary-only unique 候选并没有展现出稳定、可放大的 conversion 优势

3. **收紧 takeover 规则不会产生更稳的新线，只会把脆弱增益一起消掉**
   - `pairwise_triggered_v1` 只是弱正、且不稳
   - `pairwise_triggered_v2` 在收紧到 `2/2` takeover 后，平均 `judge_accuracy` 和 `oracle_hit_rate` 都掉到 diagnosis baseline 之下
   - 说明 task7 当前并不存在一个“只要阈值再严一点就能稳定转正”的现成机制

因此当前最合理的执行结论是：

- **task7 保留为 diagnosis-first，但暂时不值得继续产出新的 promotable narrow variant**
- 若继续做 task7，只应围绕 `selection / conversion` 机制继续找更强证据
- 在出现新的 robustness 级信号前，task7 不应重新消耗下一次正式提交机会

#### 下一步唯一值得保留的 task7 入口

基于当前 order-bias / pairwise / triggered-row 证据，task7 后续范围还可以继续压缩：

- **唯一值得保留的入口，是检查极少数 `secondary_visible_not_selected` 行里，是否存在“secondary 进入前排后就能稳定转正”的强证据。**

之所以只剩这一个入口，是因为：

1. **order sensitivity 存在，但大多不是 secondary takeover 型改善**
   - 当前 permutation probe 的 `top_choice_flip_count = 4/7`
   - 但 `flips_involving_secondary_only_count = 1/7`
   - 说明顺序敏感确实存在，可它更多表现为 primary 集合内部的不稳，而不是 secondary 候选一旦前移就稳定胜出

2. **目前最像可诊断机制的问题，只剩“secondary 排位过后”**
   - 在 `pairwise_triggered_v1` 中，seed 43 的 `person_entity` 与 `title_or_place` 触发样本里，`judge_secondary_visible_rate = 1.0`
   - 但 `judge_selected_from_secondary_rate = 0.0`
   - 在 `pairwise_triggered_v2` 的 matched rerun 里，这个现象仍然没被扭转
   - 这说明更值得查的是：secondary 候选是否只是**出现得太靠后**，而不是它本身没有任何价值

3. **如果这个入口也拿不出强证据，task7 就应基本冻结在背景层**
   - 因为剩下的 permutation flips 多数只是暴露 judge 不稳
   - 它们不能自然推出一个可提交的新 rerank 机制
   - 若连“secondary 提前后能稳定转正”都证不出来，就说明当前 task7 没有足够明确的可放大机制

因此后续 task7 diagnosis 应只保留一个非常具体的问题：

- 在最小范围的 `secondary_visible_not_selected` 样本上，若把 secondary-only candidate 前移到 top2 / top3，是否会出现**稳定、可重复、且优于 primary baseline** 的 conversion 信号？

若答案仍然是否定的，那么执行结论应进一步收紧为：

- **task7 暂时只保留历史诊断资产，不再作为当前阶段的主动分析主轴。**

#### 最小 diagnosis set（仅保留后续仍值得盯的行）

基于现有 triggered bundle / order probe / pairwise probe，当前真正值得保留为 follow-up target 的行可以压成下面这一小组：

1. **`openseek-7-8ca60b27bdd243a38664220ba472a6fa`**
   - family: `person_entity`
   - gold: `art fleming`
   - 当前是唯一一个在 order probe 里同时满足：
     - `top_choice_flip = true`
     - `flips_involve_secondary_only = true`
   - 同时它还是当前 pairwise probe 里**唯一**出现 `secondary_strong_win_count = 1` 的行
   - 这意味着它是现在最接近“secondary 前移后真的可能改写选择结果”的样本，必须保留

2. **`openseek-7-0734c7f6a8044b228a41955f1db57609`**
   - family: `title_or_place`
   - gold: `simon & schuster`
   - 在 order probe 里有 `top_choice_flip = true`
   - 在 pairwise probe 里有 `secondary_split_win_count = 2`
   - 它还不是强证据，但至少说明 secondary 候选存在局部 preference signal，值得作为第二优先级保留

3. **其余 `secondary_visible_not_selected` 行只保留为对照，不再当重点突破口**
   - `openseek-7-1cc808f724c34098b0d848cce525f298` (`shania twain`)
   - `openseek-7-93e937b2353e490b8a55d8c6bb94b9ce` (`bruce lee`)
   - `openseek-7-60b9966ac99f4b67acbffa2fde0ec17d` (`old mother hubbard`)
   - `openseek-7-dfb4648863774fce9c39c9146f7bee6b` (`sinclair lewis`)
   - 这些行要么没有 secondary takeover 型 flip，要么 pairwise 仍是 `primary_hold`
   - 它们更适合作为失败对照，不值得继续当作主动突破入口

因此 task7 的最小后续范围可以直接固定为：

- **主看 `art fleming` 这一行；次看 `simon & schuster` 这一行；其它行全部降为对照集。**

#### task7 scope-closure note

为了避免 task7 之后再次扩散回“大面积继续调 rerank 变体”的旧节奏，后续范围应直接封口为下面这条执行规则。

##### 唯一允许继续做的事

只允许围绕这两条保留行做最小化验证：

- `openseek-7-8ca60b27bdd243a38664220ba472a6fa` (`art fleming`)
- `openseek-7-0734c7f6a8044b228a41955f1db57609` (`simon & schuster`)

而且验证问题只能是：

- **把 secondary-only candidate 前移到 top2 / top3 后，是否能在这两条上稳定优于 primary baseline？**

这里的“稳定优于”至少要同时满足：

1. 不依赖单一排列顺序；
2. 不是一次性的 pairwise 偶然偏好；
3. 能在这两条保留行上同时出现方向一致的 conversion 信号；
4. 不需要扩展成新的 broad rerank 规则才成立。

##### 立即冻结条件

只要出现下面任一情况，就应把 task7 进一步冻结为“只保留历史诊断资产”：

1. `art fleming` 这条在 secondary 前移后仍然不能稳定胜出；
2. `simon & schuster` 这条仍然只有 split preference，而没有更强 conversion；
3. 任何后续正信号都必须依赖新阈值、新路由或更大范围 rerank 改写才成立；
4. 新结果只是再次暴露 judge 顺序不稳，而没有形成可重复 takeover。

##### 冻结后的口径

一旦触发冻结条件，task7 的项目内角色应进一步固定为：

- 保留 `v19 / v25 / v26` 及其 diagnosis bundle 作为历史参考；
- 不再作为当前阶段主动探索主轴；
- 除非未来出现新的方法级机制证据，否则不再进入正式主线讨论。

## 9. Submission gate before the next official package

下一次正式提交前，至少应满足：

1. 候选必须明确写清它相对的是哪个**当前正式基线**；当前统一为 `v27`。
2. 离线证据必须与 `v27` 口径一致，不能再混旧 baseline。
3. 若候选来自 task2：
   - 必须属于 low-risk residual
   - 必须有新增离线证据
   - 不接受 broad proxy patch
4. 若候选来自 task7：
   - 必须先提供 robustness / conversion 级别的新证据
   - 不接受只靠窄 uplift 的变体直接提包
5. 若候选来自 task5/6：
   - 必须先说明为什么这次离线验证比历史版本更可信

## 10. Bottom line

当前阶段最重要的，不是继续机械地找下一个 tiny patch，而是明确资源该怎么投。

这轮 post-v27 计划可以压缩成一句话：

- **`v27 = 73.05` 之后，task2 仍是唯一可信的正式主线，但它已经进入低振幅区间；下一阶段应一边保留 task2 作为稳步拿分线，一边把真正可能带来更大跃迁的资源投入到 task2 深化版或 task7 诊断突破，而不是继续在 task5/6/8 的低可信小修上消耗主资源。**
