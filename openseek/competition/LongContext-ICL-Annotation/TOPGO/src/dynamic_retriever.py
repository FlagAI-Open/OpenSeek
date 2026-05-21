#!/usr/bin/env python3
"""
动态检索模块 (Dynamic Retriever)

LDR风格的多源检索增强模块，为示例检索添加动态外部知识源 fallback。

功能：
1. 当本地示例相似度不足时，自动触发外部检索
2. 支持多源检索策略（关键词 + 语义 + 知识）
3. 动态扩展示例库

参考 Local Deep Research 的多源检索架构
"""

import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger

from .models.embedding import EmbeddingModel
from .data.loader import Example


class RetrievalStrategy(Enum):
    """检索策略"""
    LOCAL_SIMILARITY = "local_similarity"
    KEYWORD_MATCH = "keyword_match"
    SEMANTIC_SEARCH = "semantic_search"
    KNOWLEDGE_BASE = "knowledge_base"
    EXTERNAL_SEARCH = "external_search"


@dataclass
class RetrievalSource:
    """检索来源"""
    name: str
    strategy: RetrievalStrategy
    priority: int
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "strategy": self.strategy.value,
            "priority": self.priority,
            "enabled": self.enabled
        }


@dataclass
class DynamicRetrievalResult:
    """动态检索结果"""
    examples: List[Example]
    sources_used: List[RetrievalSource]
    strategies_attempted: List[RetrievalStrategy]
    fallback_triggered: bool
    total_results: int
    retrieval_time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "examples": [
                {"text_chunk": e.text_chunk, "annotation": e.annotation}
                for e in self.examples
            ],
            "sources_used": [s.to_dict() for s in self.sources_used],
            "strategies_attempted": [s.value for s in self.strategies_attempted],
            "fallback_triggered": self.fallback_triggered,
            "total_results": self.total_results,
            "retrieval_time_ms": self.retrieval_time_ms
        }


@dataclass
class ExternalSearchResult:
    """外部搜索结果"""
    source: str
    content: str
    url: Optional[str] = None
    relevance_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_example(self) -> Example:
        """转换为Example格式"""
        return Example(
            text_chunk=self.content,
            annotation={},
            embedding=None
        )


class DynamicRetriever:
    """
    动态检索器
    
    LDR风格的多源检索增强：
    1. 首先尝试本地相似度检索
    2. 当相似度不足时，自动触发外部检索
    3. 支持多种检索策略的组合
    """
    
    # 默认检索来源配置
    DEFAULT_SOURCES = [
        RetrievalSource("local_examples", RetrievalStrategy.LOCAL_SIMILARITY, 1),
        RetrievalSource("internal_kb", RetrievalStrategy.KNOWLEDGE_BASE, 2),
        RetrievalSource("keyword_index", RetrievalStrategy.KEYWORD_MATCH, 3),
        RetrievalSource("semantic_search", RetrievalStrategy.SEMANTIC_SEARCH, 4),
    ]
    
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        base_retriever: Any = None,
        top_k: int = 3,
        similarity_threshold: float = 0.5,
        fallback_threshold: float = 0.3,
        enable_external_fallback: bool = True,
        external_sources: Optional[List[str]] = None
    ):
        """
        初始化动态检索器
        
        Args:
            embedding_model: 嵌入模型
            base_retriever: 基础检索器实例
            top_k: 返回Top-K个示例
            similarity_threshold: 相似度阈值
            fallback_threshold: 触发fallback的阈值
            enable_external_fallback: 是否启用外部检索fallback
            external_sources: 外部源列表
        """
        self.embedding_model = embedding_model
        self.base_retriever = base_retriever
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.fallback_threshold = fallback_threshold
        self.enable_external_fallback = enable_external_fallback
        self.external_sources = external_sources or []
        
        # 检索来源管理
        self.sources: List[RetrievalSource] = self.DEFAULT_SOURCES.copy()
        
        # 内部知识库（用于存储高质量标注轨迹）
        self._annotation_kb: List[Example] = []
        self._annotation_embeddings: Optional[Any] = None
        
        # 检索统计
        self._stats = {
            "total_retrievals": 0,
            "fallback_count": 0,
            "external_search_count": 0
        }
        
        logger.info(
            f"动态检索器初始化, top_k={top_k}, threshold={similarity_threshold}, "
            f"fallback_threshold={fallback_threshold}, external={enable_external_fallback}"
        )
    
    def set_base_retriever(self, retriever: Any):
        """设置基础检索器"""
        self.base_retriever = retriever
        logger.debug("基础检索器已设置")
    
    def add_to_knowledge_base(self, example: Example):
        """
        添加示例到内部知识库
        
        用于积累高质量标注轨迹，形成数据飞轮
        """
        self._annotation_kb.append(example)
        
        # 更新嵌入
        if self._annotation_embeddings is None:
            self._annotation_embeddings = []
        
        if example.embedding:
            self._annotation_embeddings.append(example.embedding)
        else:
            # 动态计算嵌入
            emb = self.embedding_model.encode([example.text_chunk])[0]
            self._annotation_embeddings.append(emb)
            example.embedding = emb.tolist() if hasattr(emb, 'tolist') else emb
        
        logger.debug(f"添加示例到知识库，当前共 {len(self._annotation_kb)} 个示例")
    
    def retrieve(
        self,
        query_text: str,
        context: Optional[str] = None,
        force_fallback: bool = False
    ) -> DynamicRetrievalResult:
        """
        动态检索示例
        
        策略：
        1. 首先尝试本地相似度检索
        2. 如果结果不足或相似度低，触发fallback
        3. 从内部知识库检索
        4. 如果仍不足，触发外部搜索
        
        Args:
            query_text: 查询文本
            context: 可选的上下文信息
            force_fallback: 强制触发fallback
            
        Returns:
            DynamicRetrievalResult: 检索结果
        """
        start_time = time.time()
        self._stats["total_retrievals"] += 1
        
        examples: List[Example] = []
        sources_used: List[RetrievalSource] = []
        strategies_attempted: List[RetrievalStrategy] = []
        fallback_triggered = False
        
        # 策略1：本地相似度检索
        local_results = self._retrieve_local(query_text)
        strategies_attempted.append(RetrievalStrategy.LOCAL_SIMILARITY)
        
        if local_results:
            examples.extend(local_results)
            sources_used.append(self._find_source("local_examples"))
            
            # 检查是否需要fallback
            max_similarity = local_results[0].similarity if local_results else 0
            
            if force_fallback or max_similarity < self.fallback_threshold:
                fallback_triggered = True
                self._stats["fallback_count"] += 1
                
                # 策略2：内部知识库检索
                kb_results = self._retrieve_from_kb(query_text)
                if kb_results:
                    examples.extend(kb_results)
                    sources_used.append(self._find_source("internal_kb"))
                    strategies_attempted.append(RetrievalStrategy.KNOWLEDGE_BASE)
                
                # 策略3：关键词匹配
                keyword_results = self._retrieve_by_keywords(query_text, context)
                if keyword_results:
                    examples.extend(keyword_results)
                    sources_used.append(self._find_source("keyword_index"))
                    strategies_attempted.append(RetrievalStrategy.KEYWORD_MATCH)
                
                # 策略4：外部搜索（如果启用）
                if self.enable_external_fallback and len(examples) < self.top_k:
                    external_results = self._search_external(query_text, context)
                    if external_results:
                        examples.extend(external_results)
                        sources_used.append(RetrievalSource(
                            "external_search",
                            RetrievalStrategy.EXTERNAL_SEARCH,
                            5
                        ))
                        strategies_attempted.append(RetrievalStrategy.EXTERNAL_SEARCH)
                        self._stats["external_search_count"] += 1
        
        # 去重并限制数量
        examples = self._deduplicate_and_limit(examples)
        
        retrieval_time = (time.time() - start_time) * 1000
        
        result = DynamicRetrievalResult(
            examples=examples,
            sources_used=sources_used,
            strategies_attempted=strategies_attempted,
            fallback_triggered=fallback_triggered,
            total_results=len(examples),
            retrieval_time_ms=retrieval_time
        )
        
        logger.debug(
            f"动态检索完成: query='{query_text[:50]}...', "
            f"results={len(examples)}, fallback={fallback_triggered}, "
            f"time={retrieval_time:.2f}ms"
        )
        
        return result
    
    def _retrieve_local(self, query_text: str) -> List[Any]:
        """从基础检索器获取本地结果"""
        if self.base_retriever and hasattr(self.base_retriever, 'retrieve'):
            try:
                results = self.base_retriever.retrieve(query_text, top_k=self.top_k)
                return results
            except Exception as e:
                logger.warning(f"本地检索失败: {e}")
        return []
    
    def _retrieve_from_kb(self, query_text: str) -> List[Example]:
        """从内部知识库检索"""
        if not self._annotation_kb:
            return []
        
        # 计算查询嵌入
        query_emb = self.embedding_model.encode([query_text])[0]
        
        # 计算相似度
        kb_embeddings = self.embedding_model.compute_similarity_matrix(
            query_emb.reshape(1, -1),
            self._annotation_embeddings
        )[0]
        
        # 排序并选取Top-K
        sorted_indices = kb_embeddings.argsort()[::-1]
        
        results = []
        for idx in sorted_indices[:self.top_k]:
            if kb_embeddings[idx] >= self.similarity_threshold:
                results.append(self._annotation_kb[idx])
        
        return results
    
    def _retrieve_by_keywords(
        self,
        query_text: str,
        context: Optional[str]
    ) -> List[Example]:
        """基于关键词检索"""
        # 提取关键词
        keywords = self._extract_keywords(query_text)
        if not keywords:
            return []
        
        # 在基础检索器的示例库中搜索
        if self.base_retriever and hasattr(self.base_retriever, 'example_library'):
            matches = []
            for example in self.base_retriever.example_library:
                text_lower = example.text_chunk.lower()
                if any(kw.lower() in text_lower for kw in keywords):
                    matches.append(example)
                    if len(matches) >= self.top_k:
                        break
            return matches
        
        return []
    
    def _search_external(
        self,
        query_text: str,
        context: Optional[str]
    ) -> List[Example]:
        """
        外部搜索（模拟LDR的多源检索）
        
        在实际部署时，可集成：
        - Wikipedia API
        - arXiv API
        - 搜索引擎API
        - 私有文档库
        """
        results = []
        
        # 模拟外部搜索结果
        # 实际部署时替换为真实API调用
        if self.external_sources:
            for source in self.external_sources:
                # 模拟从外部源获取结果
                mock_result = ExternalSearchResult(
                    source=source,
                    content=f"从{source}检索到与'{query_text}'相关的示例",
                    relevance_score=0.6
                )
                results.append(mock_result.to_example())
        
        logger.debug(f"外部搜索返回 {len(results)} 个结果")
        return results
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词（简单实现）"""
        # 简单实现：提取长度大于3的词
        import re
        words = re.findall(r'\b\w{4,}\b', text)
        return list(set(words))[:10]  # 最多10个关键词
    
    def _deduplicate_and_limit(
        self,
        examples: List[Example]
    ) -> List[Example]:
        """去重并限制数量"""
        seen_texts = set()
        unique_examples = []
        
        for ex in examples:
            if ex.text_chunk not in seen_texts:
                seen_texts.add(ex.text_chunk)
                unique_examples.append(ex)
                
                if len(unique_examples) >= self.top_k:
                    break
        
        return unique_examples
    
    def _find_source(self, name: str) -> RetrievalSource:
        """查找检索来源"""
        for source in self.sources:
            if source.name == name:
                return source
        return RetrievalSource(name, RetrievalStrategy.LOCAL_SIMILARITY, 99)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取检索统计"""
        return {
            **self._stats,
            "kb_size": len(self._annotation_kb),
            "fallback_rate": (
                self._stats["fallback_count"] / self._stats["total_retrievals"]
                if self._stats["total_retrievals"] > 0 else 0
            )
        }
    
    def reset_stats(self):
        """重置统计"""
        self._stats = {
            "total_retrievals": 0,
            "fallback_count": 0,
            "external_search_count": 0
        }


class SelfEvolvingKnowledgeBase:
    """
    自进化知识库
    
    用于积累高质量标注轨迹，形成数据飞轮：
    - 标注越多，知识库越丰富
    - 知识库越丰富，后续标注的参考质量越高
    
    对应 Hermes "自我进化" 的核心思想
    """
    
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        storage_path: Optional[str] = None
    ):
        """
        初始化自进化知识库
        
        Args:
            embedding_model: 嵌入模型
            storage_path: 存储路径（可选）
        """
        self.embedding_model = embedding_model
        self.storage_path = storage_path
        
        # 知识库内容
        self.entries: List[Dict[str, Any]] = []
        self.embeddings: Optional[Any] = None
        
        logger.info("自进化知识库初始化")
    
    def add_annotation_trajectory(
        self,
        text_chunk: str,
        annotation: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        quality_score: float = 1.0
    ):
        """
        添加标注轨迹
        
        Args:
            text_chunk: 文本块
            annotation: 标注结果
            context: 上下文信息（任务描述、领域等）
            quality_score: 质量分数（用于过滤低质量轨迹）
        """
        if quality_score < 0.7:
            logger.debug(f"跳过低质量轨迹: score={quality_score}")
            return
        
        # 计算嵌入
        embedding = self.embedding_model.encode([text_chunk])[0]
        
        entry = {
            "text_chunk": text_chunk,
            "annotation": annotation,
            "context": context or {},
            "quality_score": quality_score,
            "embedding": embedding.tolist() if hasattr(embedding, 'tolist') else embedding
        }
        
        self.entries.append(entry)
        
        # 更新嵌入矩阵
        if self.embeddings is None:
            self.embeddings = embedding.reshape(1, -1)
        else:
            self.embeddings = np.vstack([self.embeddings, embedding.reshape(1, -1)])
        
        logger.debug(f"添加标注轨迹，当前共 {len(self.entries)} 个")
    
    def retrieve_similar_cases(
        self,
        query: str,
        top_k: int = 3,
        min_quality_score: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        检索相似案例
        
        Args:
            query: 查询文本
            top_k: 返回数量
            min_quality_score: 最低质量分数
            
        Returns:
            相似案例列表
        """
        if not self.entries:
            return []
        
        # 计算查询嵌入
        query_emb = self.embedding_model.encode([query])[0]
        
        # 计算相似度
        similarities = self.embedding_model.compute_similarity_matrix(
            query_emb.reshape(1, -1),
            self.embeddings
        )[0]
        
        # 排序
        sorted_indices = similarities.argsort()[::-1]
        
        results = []
        for idx in sorted_indices:
            entry = self.entries[idx]
            if entry["quality_score"] >= min_quality_score:
                results.append({
                    **entry,
                    "similarity": float(similarities[idx])
                })
                
                if len(results) >= top_k:
                    break
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """获取知识库统计"""
        return {
            "total_entries": len(self.entries),
            "avg_quality_score": (
                sum(e["quality_score"] for e in self.entries) / len(self.entries)
                if self.entries else 0
            ),
            "domains": list(set(
                e.get("context", {}).get("domain", "unknown")
                for e in self.entries
            ))
        }


# 导入numpy（用于嵌入矩阵操作）
import numpy as np