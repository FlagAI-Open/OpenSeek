# Local Deep Research (LDR) 集成技术文档

## 概述

本文档描述了如何将 Local Deep Research (LDR) 的核心能力集成到 Track 3 智能标注系统中。LDR 作为一个"研究型智能体"范式，为现有四智能体流水线提供了三大增强：

1. **动态多源检索** - 扩展示例库从静态到动态可扩展
2. **证据验证** - 将质检从"基于规则"升级为"基于规则+证据"
3. **自进化知识库** - 形成数据飞轮，实现系统自我进化

---

## 架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                    四智能体协同流水线                             │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────┐  │
│  │Orchestrator│→│ Retriever │→│Playwright│→│QualityInspector│ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ LDR 增强层
┌─────────────────────────────────────────────────────────────────┐
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ DynamicRetriever │  │EvidenceVerifier │  │SelfEvolvingKB   │  │
│  │  (动态检索增强)   │  │  (证据验证增强)   │  │ (自进化知识库)   │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│           │                   │                   │              │
│           ▼                   ▼                   ▼              │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              LDR 增强层 API                               │    │
│  │  • quick_research()    • verify_with_evidence()          │    │
│  │  • retrieve()          • add_annotation_trajectory()      │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 核心模块

### 1. EvidenceVerifier (证据验证器)

**文件**: `src/evidence_verifier.py`

**功能**: 对存疑实体进行快速证据检索，辅助质检决策

**核心类**:

| 类名 | 说明 |
|------|------|
| `EvidenceVerifier` | LDR风格的快速研究模块 |
| `EnhancedQualityInspector` | 集成证据验证的增强版质检智能体 |
| `Evidence` | 证据记录数据结构 |
| `VerificationDecision` | 验证决策数据结构 |

**使用示例**:

```python
from src.evidence_verifier import EvidenceVerifier, EnhancedQualityInspector

# 初始化证据验证器
evidence_verifier = EvidenceVerifier(
    model_client=None,  # 可选：LLM客户端
    max_duration=120,
    enable_external_search=False
)

# 初始化增强版质检智能体
quality_inspector = EnhancedQualityInspector(
    evidence_verifier=evidence_verifier,
    enable_evidence_verification=True
)

# 对存疑实体进行验证
decision = quality_inspector.verify_entity_with_evidence(
    entity="某复杂实体",
    entity_type="legal_term",
    context="合同上下文...",
    task_domain="legal"
)

print(f"决策: {decision.decision}, 置信度: {decision.confidence}")
```

**API 参考**:

#### EvidenceVerifier.verify_with_evidence()

```python
def verify_with_evidence(
    self,
    entity: str,
    entity_type: str,
    context: str,
    task_domain: str = "general"
) -> VerificationDecision
```

对实体标注进行证据验证，返回包含决策、置信度和使用证据的验证结果。

#### EnhancedQualityInspector.verify_entity_with_evidence()

```python
def verify_entity_with_evidence(
    self,
    entity: str,
    entity_type: str,
    context: str,
    task_domain: str = "general"
) -> VerificationDecision
```

使用证据验证实体标注，是质检智能体的增强方法。

---

### 2. DynamicRetriever (动态检索器)

**文件**: `src/dynamic_retriever.py`

**功能**: 多源检索增强，当本地相似度不足时自动触发外部检索

**核心类**:

| 类名 | 说明 |
|------|------|
| `DynamicRetriever` | LDR风格的多源检索器 |
| `SelfEvolvingKnowledgeBase` | 自进化知识库，用于积累标注轨迹 |
| `DynamicRetrievalResult` | 动态检索结果 |
| `RetrievalStrategy` | 检索策略枚举 |

**使用示例**:

```python
from src.dynamic_retriever import DynamicRetriever, SelfEvolvingKnowledgeBase
from src.retriever import ExampleRetriever
from src.models.embedding import EmbeddingModel

# 初始化
embedding_model = EmbeddingModel(...)
base_retriever = ExampleRetriever(embedding_model)

# 创建动态检索器
dynamic_retriever = DynamicRetriever(
    embedding_model=embedding_model,
    base_retriever=base_retriever,
    top_k=3,
    similarity_threshold=0.5,
    fallback_threshold=0.3,
    enable_external_fallback=True
)

# 执行动态检索
result = dynamic_retriever.retrieve(
    query_text="查询文本",
    context="可选上下文",
    force_fallback=False
)

print(f"检索结果: {result.total_results}, 触发fallback: {result.fallback_triggered}")

# 使用自进化知识库
kb = SelfEvolvingKnowledgeBase(embedding_model)

# 添加高质量标注轨迹
kb.add_annotation_trajectory(
    text_chunk="文本块",
    annotation={"entities": [...], "relations": [...]},
    context={"domain": "legal", "task": "responsibility"},
    quality_score=0.95
)

# 检索相似案例
similar_cases = kb.retrieve_similar_cases(
    query="新查询文本",
    top_k=3,
    min_quality_score=0.7
)
```

**API 参考**:

#### DynamicRetriever.retrieve()

```python
def retrieve(
    self,
    query_text: str,
    context: Optional[str] = None,
    force_fallback: bool = False
) -> DynamicRetrievalResult
```

执行动态检索，自动选择最佳策略组合。

#### SelfEvolvingKnowledgeBase.add_annotation_trajectory()

```python
def add_annotation_trajectory(
    self,
    text_chunk: str,
    annotation: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
    quality_score: float = 1.0
)
```

添加高质量标注轨迹到知识库。

#### SelfEvolvingKnowledgeBase.retrieve_similar_cases()

```python
def retrieve_similar_cases(
    self,
    query: str,
    top_k: int = 3,
    min_quality_score: float = 0.7
) -> List[Dict[str, Any]]
```

从知识库检索相似案例。

---

## 集成到现有流水线

### 方式一：直接替换

```python
from src import (
    FourAgentPipeline,
    DynamicRetriever,  # 替换 ExampleRetriever
    EnhancedQualityInspector  # 替换 QualityInspectorAgent
)
from src.evidence_verifier import EvidenceVerifier

# 使用LDR增强的组件
evidence_verifier = EvidenceVerifier()
enhanced_qc = EnhancedQualityInspector(evidence_verifier=evidence_verifier)

pipeline = FourAgentPipeline(
    model_client=model_client,
    retriever=DynamicRetriever(...),  # 使用动态检索器
    # 其他参数...
)
```

### 方式二：渐进式集成

```python
# 阶段1：仅启用证据验证
from src.evidence_verifier import EnhancedQualityInspector

# 在现有质检流程中添加证据验证钩子
def quality_check_with_evidence(entity, context):
    verifier = EvidenceVerifier()
    enhanced_qc = EnhancedQualityInspector(evidence_verifier=verifier)
    return enhanced_qc.verify_entity_with_evidence(entity, context)

# 阶段2：启用动态检索
from src.dynamic_retriever import DynamicRetriever

dynamic_retriever = DynamicRetriever(
    embedding_model=embedding_model,
    base_retriever=existing_retriever,
    enable_external_fallback=True
)

# 阶段3：启用自进化知识库
from src.dynamic_retriever import SelfEvolvingKnowledgeBase

kb = SelfEvolvingKnowledgeBase(embedding_model)

# 在每个高质量标注后添加到知识库
kb.add_annotation_trajectory(text, annotation, quality_score=score)
```

---

## 数据飞轮机制

```
┌─────────────┐     标注越多      ┌─────────────┐
│   标注任务   │ ──────────────→  │  知识库积累  │
└─────────────┘                  └─────────────┘
      ↑                                │
      │                                ▼
      │                         ┌─────────────┐
      └─────── 更高质量标注 ─────│ 知识库检索  │
                                └─────────────┘
```

1. **初始阶段**: 使用固定示例库进行标注
2. **积累阶段**: 高质量标注轨迹自动添加到知识库
3. **增强阶段**: 检索时从知识库获取相似案例
4. **进化阶段**: 知识库越丰富，后续标注质量越高

---

## 预期收益

| 指标 | 当前 | LDR增强后 | 提升 |
|------|------|----------|------|
| 复杂案例准确率 | 基准 | +5-10% | 证据验证 |
| 未知领域泛化 | 有限 | 显著提升 | 动态检索 |
| 系统自适应性 | 静态 | 自进化 | 知识库积累 |

---

## 注意事项

1. **不是替代**: LDR模块不是替代Qwen3-4B做标注，而是增强智能体系统的感知、决策和记忆能力

2. **隐私合规**: 所有模块支持完全本地部署，标注数据无需出库

3. **性能考虑**: 
   - 证据验证默认超时120秒
   - 外部搜索可按需启用
   - 知识库定期清理低质量轨迹

4. **扩展性**: 
   - 支持自定义检索来源
   - 支持自定义证据源
   - 支持知识库持久化

---

## 版本历史

- **v1.1.0**: 添加LDR集成模块
  - EvidenceVerifier: 证据验证模块
  - DynamicRetriever: 动态检索模块
  - SelfEvolvingKnowledgeBase: 自进化知识库