# 基于上下文感知与自洽性验证的长文档智能标注系统
# FlagOS开放计算全球挑战赛 - 赛道三：自动数据标注

__version__ = "1.1.0"
__author__ = "FlagOS Challenge Team"

# 四智能体流水线
from .orchestrator import OrchestratorAgent, TaskPlan, TextChunk, WorkMemory
from .playwright import PlaywrightAgent, PromptScript, ChainOfThoughtStep
from .quality_inspector import QualityInspectorAgent, KnowledgeGraph, ValidationResult
from .four_agent_pipeline import FourAgentPipeline, PipelineConfig, PipelineResult

# Agent最佳实践工具
from .agent_utils import (
    # 重试机制
    retry_with_exponential_backoff,
    retry_with_fallback,
    # 缓存
    SearchCache,
    CacheEntry,
    # 性能监控
    PerformanceMonitor,
    global_monitor,
    # 答案提取与归一化
    AnswerExtractor,
    AnswerNormalizer,
    AnswerValidator,
    # 错误处理
    ErrorHandler,
    # 日志记录
    ExecutionLogger,
    # 便捷函数
    safe_execute
)

# LDR集成模块 - Local Deep Research增强
from .evidence_verifier import (
    EvidenceVerifier,
    EvidenceSource,
    Evidence,
    ResearchResult,
    VerificationDecision,
    EnhancedQualityInspector
)

from .dynamic_retriever import (
    DynamicRetriever,
    DynamicRetrievalResult,
    RetrievalStrategy,
    RetrievalSource,
    ExternalSearchResult,
    SelfEvolvingKnowledgeBase
)
