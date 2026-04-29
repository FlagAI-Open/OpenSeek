# RFC: 全能优化系统

## Motivation

当前优化能力散落在多个组合方案中：相似度、多样性、质量过滤、任务感知、动态反馈、对比表示、自洽性等信号虽然各自成立，但缺少统一编排入口，导致 API surface 逐步扩大、策略分支增多，长期 ergonomics 变差。因此需要一个总协调层统一接管排序与调度。

## Guide-level explanation

本方案引入一个统一排序器，将前序方案的主要信号收敛到同一分数空间中；对使用者而言，理解方式很简单：

1. 候选样本先经过基础过滤
2. 所有有效信号在总排序器里融合
3. 最后通过共享比例重平衡控制多任务注入节奏

## Reference-level explanation

关键实体：

- `_unified_optimizer_TASK_CONFIG`
- `rank_examples_for_unified_optimizer()`
- `rebalance_examples_for_unified_optimizer()`

这些组件共同负责：

- 质量过滤
- 相似度与对比表示融合
- 历史奖励注入
- 任务专项权重
- 共享样本重平衡

## Drawbacks

1. 权重数量显著增加。
2. 统一排序器会放大配置耦合。
3. 这不是完整训练得到的系统，因此 design space 仍受启发式规则限制。

## Rationale and Alternatives

被否决方案 A：继续保留多个分支并行启用。
原因：主流程复杂度持续上升，维护成本不可接受。

被否决方案 B：引入新的集中训练器。
原因：超出当前项目约束，且会引入 breaking change 风险。

采用当前方案的理由是：在不扩大 semver 级别风险的前提下，最大化复用已有实现。

## Prior art

- 现有 既有方案：单任务动态自适应
- 现有 既有方案：跨任务共享池
- 现有 既有方案：对比表示 + 强化反馈
- 现有 既有方案：分层组织与质量门槛

## Unresolved Questions

1. 统一排序器的权重是否应进一步任务图谱化？
2. 重平衡逻辑是否会在极端任务上压制高价值共享样本？
3. 后续是否需要单独拆出一个更小的 API surface 供 Task 8 复用？
