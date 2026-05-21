# LDR集成 - 容器使用指南

## 快速开始

### 1. 基础导入

```python
from src import (
    FourAgentPipeline,
    DynamicRetriever,
    EnhancedQualityInspector
)
from src.evidence_verifier import EvidenceVerifier
from src.dynamic_retriever import SelfEvolvingKnowledgeBase
```

### 2. 初始化LDR增强组件

```python
# 初始化嵌入模型
from src.models.embedding import EmbeddingModel
embedding_model = EmbeddingModel()

# 初始化证据验证器
evidence_verifier = EvidenceVerifier(
    model_client=None,  # 可选：LLM客户端
    max_duration=120,
    enable_external_search=False
)

# 初始化增强版质检智能体
enhanced_qc = EnhancedQualityInspector(
    evidence_verifier=evidence_verifier,
    enable_evidence_verification=True
)

# 初始化动态检索器
from src.retriever import ExampleRetriever
base_retriever = ExampleRetriever(embedding_model)

dynamic_retriever = DynamicRetriever(
    embedding_model=embedding_model,
    base_retriever=base_retriever,
    top_k=3,
    similarity_threshold=0.5,
    fallback_threshold=0.3,
    enable_external_fallback=True
)

# 初始化自进化知识库
kb = SelfEvolvingKnowledgeBase(embedding_model)
```

### 3. 在流水线中使用

```python
# 使用动态检索器
result = dynamic_retriever.retrieve(
    query_text="查询文本",
    context="上下文（可选）",
    force_fallback=False
)
print(f"检索结果: {result.total_results}, 触发fallback: {result.fallback_triggered}")

# 使用证据验证
decision = enhanced_qc.verify_entity_with_evidence(
    entity="某实体",
    entity_type="legal_term",
    context="合同上下文...",
    task_domain="legal"
)
print(f"决策: {decision.decision}, 置信度: {decision.confidence}")

# 使用自进化知识库
kb.add_annotation_trajectory(
    text_chunk="文本块",
    annotation={"entities": [], "relations": []},
    context={"domain": "legal"},
    quality_score=0.95
)

similar_cases = kb.retrieve_similar_cases(
    query="新查询",
    top_k=3,
    min_quality_score=0.7
)
```

## 容器命令

### 查看检索统计

```python
# 获取动态检索器统计
stats = dynamic_retriever.get_stats()
print(f"总检索次数: {stats['total_retrievals']}")
print(f"Fallback次数: {stats['fallback_count']}")
print(f"Fallback率: {stats['fallback_rate']:.2%}")

# 获取知识库统计
kb_stats = kb.get_stats()
print(f"知识库条目: {kb_stats['total_entries']}")
print(f"平均质量分: {kb_stats['avg_quality_score']:.2f}")
```

### 重置统计

```python
dynamic_retriever.reset_stats()
```

### 导出知识库（可选持久化）

```python
import json

# 导出知识库
kb_data = {
    "entries": kb.entries,
    "stats": kb.get_stats()
}
with open("knowledge_base.json", "w", encoding="utf-8") as f:
    json.dump(kb_data, f, ensure_ascii=False, indent=2)
```

## 性能调优建议

| 参数 | 默认值 | 调优建议 |
|------|--------|----------|
| `top_k` | 3 | 复杂任务可增至5 |
| `similarity_threshold` | 0.5 | 高精度需求可提高至0.7 |
| `fallback_threshold` | 0.3 | 低于此值触发外部检索 |
| `max_duration` | 120s | 证据验证超时时间 |

## 注意事项

1. **外部搜索**: `enable_external_search=True` 需要网络访问
2. **知识库清理**: 定期清理低质量轨迹（quality_score < 0.7）
3. **内存管理**: 知识库增大会影响内存，定期持久化