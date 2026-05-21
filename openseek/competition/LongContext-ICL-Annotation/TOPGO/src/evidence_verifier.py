#!/usr/bin/env python3
"""
证据验证模块 (Evidence Verifier)

LDR风格的证据检索与验证模块，用于质检智能体的复杂实体标注验证。
当遇到模糊实体时，可从外部知识源获取权威证据辅助决策。

功能：
1. 快速研究查询 (quick_research)
2. 证据提取与解析
3. 多源知识检索
4. 决策辅助

参考 Local Deep Research 架构设计
"""

import json
import re
import time
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger


class EvidenceSource(Enum):
    """证据来源类型"""
    WIKIPEDIA = "wikipedia"
    ACADEMIC = "academic"
    LEGAL = "legal"
    TECHNICAL = "technical"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    INTERNAL_KB = "internal_kb"


@dataclass
class Evidence:
    """证据记录"""
    source: str
    source_type: EvidenceSource
    content: str
    url: Optional[str] = None
    relevance_score: float = 0.0
    citation: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "source_type": self.source_type.value,
            "content": self.content,
            "url": self.url,
            "relevance_score": self.relevance_score,
            "citation": self.citation
        }


@dataclass
class ResearchResult:
    """研究结果"""
    query: str
    evidence_list: List[Evidence]
    summary: str
    duration_seconds: float
    sources_used: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "evidence_list": [e.to_dict() for e in self.evidence_list],
            "summary": self.summary,
            "duration_seconds": self.duration_seconds,
            "sources_used": self.sources_used
        }


@dataclass
class VerificationDecision:
    """验证决策"""
    entity: str
    entity_type: str
    decision: str  # "confirm", "revise", "uncertain"
    confidence: float
    evidence_used: List[Evidence]
    reasoning: str
    alternative_interpretations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity": self.entity,
            "entity_type": self.entity_type,
            "decision": self.decision,
            "confidence": self.confidence,
            "evidence_used": [e.to_dict() for e in self.evidence_used],
            "reasoning": self.reasoning,
            "alternative_interpretations": self.alternative_interpretations
        }


class EvidenceVerifier:
    """
    证据验证器
    
    LDR风格的快速研究模块，用于：
    1. 对存疑实体进行快速证据检索
    2. 从多个知识源收集权威信息
    3. 辅助质检决策
    """
    
    # 预定义的实体类型查询模板
    ENTITY_QUERY_TEMPLATES = {
        "person": "在${context}中，实体'${entity}'作为人物承担什么角色或责任？",
        "organization": "在${context}中，实体'${entity}'作为组织有什么职能和责任？",
        "technology": "在${context}中，技术术语'${entity}'的定义和用途是什么？",
        "legal_term": "在法律合同中，实体'${entity}'通常承担什么责任或角色？提供权威来源。",
        "default": "在以下上下文中，实体'${entity}'的准确含义和类型是什么？${context}"
    }
    
    # 可信知识源配置
    TRUSTED_SOURCES = {
        EvidenceSource.WIKIPEDIA: True,
        EvidenceSource.ACADEMIC: True,
        EvidenceSource.LEGAL: True,
        EvidenceSource.TECHNICAL: True,
        EvidenceSource.KNOWLEDGE_GRAPH: True,
        EvidenceSource.INTERNAL_KB: True
    }
    
    def __init__(
        self,
        model_client: Any = None,
        max_duration: int = 120,
        enable_external_search: bool = False
    ):
        """
        初始化证据验证器
        
        Args:
            model_client: LLM客户端（用于生成研究查询和总结）
            max_duration: 最大研究时长（秒）
            enable_external_search: 是否启用外部搜索（需要网络）
        """
        self.model_client = model_client
        self.max_duration = max_duration
        self.enable_external_search = enable_external_search
        
        # 内部知识库（模拟）
        self._internal_kb: Dict[str, List[Evidence]] = {}
        
        logger.info(f"证据验证器初始化, max_duration={max_duration}s, external_search={enable_external_search}")
    
    def verify_with_evidence(
        self,
        entity: str,
        entity_type: str,
        context: str,
        task_domain: str = "general"
    ) -> VerificationDecision:
        """
        使用证据验证实体标注
        
        Args:
            entity: 实体文本
            entity_type: 实体类型
            context: 上下文文本
            task_domain: 任务领域
            
        Returns:
            VerificationDecision: 验证决策
        """
        logger.info(f"开始验证实体: {entity} (类型: {entity_type})")
        start_time = time.time()
        
        # 1. 构建研究查询
        query = self._build_query(entity, entity_type, context, task_domain)
        
        # 2. 执行快速研究
        research_result = self._quick_research(query, context)
        
        # 3. 提取证据
        evidence_list = research_result.evidence_list
        
        # 4. 生成决策
        decision = self._make_decision(
            entity, entity_type, context, evidence_list
        )
        
        duration = time.time() - start_time
        logger.info(f"实体验证完成: {entity}, 决策: {decision.decision}, 耗时: {duration:.2f}s")
        
        return decision
    
    def _build_query(
        self,
        entity: str,
        entity_type: str,
        context: str,
        task_domain: str
    ) -> str:
        """构建研究查询"""
        template = self.ENTITY_QUERY_TEMPLATES.get(
            entity_type,
            self.ENTITY_QUERY_TEMPLATES["default"]
        )
        
        # 截取上下文（避免过长）
        context_snippet = context[:500] if len(context) > 500 else context
        
        query = template.replace("${entity}", entity).replace("${context}", context_snippet)
        return query
    
    def _quick_research(
        self,
        query: str,
        context: str
    ) -> ResearchResult:
        """
        快速研究（模拟LDR的quick_research功能）
        
        在实际部署时，可集成：
        - Wikipedia API
        - arXiv API
        - 私有知识库
        - 搜索引擎API
        
        Args:
            query: 研究查询
            context: 上下文文本
            
        Returns:
            ResearchResult: 研究结果
        """
        start_time = time.time()
        evidence_list: List[Evidence] = []
        sources_used: List[str] = []
        
        # 1. 从内部知识图谱检索
        kg_evidence = self._search_knowledge_graph(query, context)
        if kg_evidence:
            evidence_list.extend(kg_evidence)
            sources_used.append("knowledge_graph")
        
        # 2. 从内部知识库检索
        kb_evidence = self._search_internal_kb(query)
        if kb_evidence:
            evidence_list.extend(kb_evidence)
            sources_used.append("internal_kb")
        
        # 3. 模拟外部源检索（实际部署时替换为真实API调用）
        if self.enable_external_search:
            ext_evidence = self._search_external_sources(query)
            if ext_evidence:
                evidence_list.extend(ext_evidence)
                sources_used.append("external")
        
        # 4. 如果没有找到证据，生成基于上下文的推理证据
        if not evidence_list:
            inferred_evidence = self._generate_inferred_evidence(query, context)
            evidence_list.append(inferred_evidence)
            sources_used.append("inferred")
        
        # 生成总结
        summary = self._generate_summary(query, evidence_list)
        
        duration = time.time() - start_time
        
        return ResearchResult(
            query=query,
            evidence_list=evidence_list,
            summary=summary,
            duration_seconds=duration,
            sources_used=sources_used
        )
    
    def _search_knowledge_graph(
        self,
        query: str,
        context: str
    ) -> List[Evidence]:
        """从知识图谱检索证据"""
        # 提取查询中的关键实体
        entities = self._extract_entities(query)
        
        evidence_list = []
        for ent in entities[:3]:  # 限制数量
            # 模拟知识图谱检索
            evidence_list.append(Evidence(
                source=f"知识图谱: {ent}",
                source_type=EvidenceSource.KNOWLEDGE_GRAPH,
                content=f"在知识图谱中找到与'{ent}'相关的实体记录",
                relevance_score=0.8
            ))
        
        return evidence_list
    
    def _search_internal_kb(self, query: str) -> List[Evidence]:
        """从内部知识库检索"""
        evidence_list = []
        
        # 检查内部KB中是否有相关记录
        for key, evidence_items in self._internal_kb.items():
            if key.lower() in query.lower():
                evidence_list.extend(evidence_items[:2])
        
        return evidence_list
    
    def _search_external_sources(self, query: str) -> List[Evidence]:
        """搜索外部源（模拟）"""
        # 在实际部署时，这里会调用：
        # - Wikipedia API
        # - 学术数据库API
        # - 搜索引擎API
        
        evidence_list = []
        
        # 模拟Wikipedia结果
        evidence_list.append(Evidence(
            source="Wikipedia",
            source_type=EvidenceSource.WIKIPEDIA,
            content=f"从Wikipedia检索到关于'{query}'的权威定义",
            url="https://wikipedia.org",
            relevance_score=0.75
        ))
        
        return evidence_list
    
    def _generate_inferred_evidence(
        self,
        query: str,
        context: str
    ) -> Evidence:
        """生成基于上下文的推理证据"""
        return Evidence(
            source="上下文推理",
            source_type=EvidenceSource.INTERNAL_KB,
            content=f"基于提供上下文的推理分析：{query}",
            relevance_score=0.5
        )
    
    def _generate_summary(
        self,
        query: str,
        evidence_list: List[Evidence]
    ) -> str:
        """生成研究总结"""
        if not evidence_list:
            return "未找到相关证据"
        
        sources = [e.source for e in evidence_list]
        return f"从以下来源获取了{len(evidence_list)}条证据: {', '.join(sources)}"
    
    def _extract_entities(self, text: str) -> List[str]:
        """提取文本中的实体（简单实现）"""
        # 简单实现：提取引号内的内容
        matches = re.findall(r"'([^']+)'", text)
        if matches:
            return matches
        
        # 提取被引号包围的实体
        matches = re.findall(r'"([^"]+)"', text)
        return matches
    
    def _make_decision(
        self,
        entity: str,
        entity_type: str,
        context: str,
        evidence_list: List[Evidence]
    ) -> VerificationDecision:
        """基于证据做出验证决策"""
        
        if not evidence_list:
            return VerificationDecision(
                entity=entity,
                entity_type=entity_type,
                decision="uncertain",
                confidence=0.0,
                evidence_used=[],
                reasoning="未找到相关证据，无法验证"
            )
        
        # 计算平均相关性分数
        avg_relevance = sum(e.relevance_score for e in evidence_list) / len(evidence_list)
        
        # 基于证据做出决策
        if avg_relevance >= 0.7:
            decision = "confirm"
            confidence = avg_relevance
            reasoning = f"找到{len(evidence_list)}条高相关性证据，支持当前标注"
        elif avg_relevance >= 0.4:
            decision = "revise"
            confidence = avg_relevance
            reasoning = f"证据支持修订当前标注，存在更准确的解释"
        else:
            decision = "uncertain"
            confidence = avg_relevance
            reasoning = "证据不足以确认或修订当前标注"
        
        return VerificationDecision(
            entity=entity,
            entity_type=entity_type,
            decision=decision,
            confidence=confidence,
            evidence_used=evidence_list,
            reasoning=reasoning
        )
    
    def add_to_knowledge_base(
        self,
        entity: str,
        entity_type: str,
        evidence: Evidence
    ):
        """添加实体证据到内部知识库"""
        key = f"{entity_type}:{entity}"
        if key not in self._internal_kb:
            self._internal_kb[key] = []
        self._internal_kb[key].append(evidence)
        logger.debug(f"添加证据到知识库: {key}")
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        return {
            "total_entities": len(self._internal_kb),
            "total_evidence": sum(len(v) for v in self._internal_kb.values()),
            "entities": list(self._internal_kb.keys())
        }


class EnhancedQualityInspector:
    """
    增强版质检智能体
    
    集成LDR风格的证据验证能力，
    将质检从"基于规则"升级为"基于规则+证据"
    """
    
    def __init__(
        self,
        evidence_verifier: Optional[EvidenceVerifier] = None,
        enable_evidence_verification: bool = True
    ):
        """
        初始化增强版质检智能体
        
        Args:
            evidence_verifier: 证据验证器实例
            enable_evidence_verification: 是否启用证据验证
        """
        self.evidence_verifier = evidence_verifier or EvidenceVerifier()
        self.enable_evidence_verification = enable_evidence_verification
        
        logger.info(f"增强版质检智能体初始化, evidence_verification={enable_evidence_verification}")
    
    def verify_entity_with_evidence(
        self,
        entity: str,
        entity_type: str,
        context: str,
        task_domain: str = "general"
    ) -> VerificationDecision:
        """
        使用证据验证实体标注
        
        Args:
            entity: 实体文本
            entity_type: 实体类型
            context: 上下文
            task_domain: 任务领域
            
        Returns:
            VerificationDecision: 验证决策
        """
        if not self.enable_evidence_verification:
            # 如果禁用证据验证，返回默认决策
            return VerificationDecision(
                entity=entity,
                entity_type=entity_type,
                decision="confirm",
                confidence=1.0,
                evidence_used=[],
                reasoning="证据验证已禁用"
            )
        
        return self.evidence_verifier.verify_with_evidence(
            entity=entity,
            entity_type=entity_type,
            context=context,
            task_domain=task_domain
        )
    
    def batch_verify_entities(
        self,
        entities: List[Dict[str, str]],
        context: str,
        task_domain: str = "general"
    ) -> List[VerificationDecision]:
        """
        批量验证实体
        
        Args:
            entities: 实体列表 [{"entity": "...", "type": "..."}]
            context: 上下文
            task_domain: 任务领域
            
        Returns:
            验证决策列表
        """
        results = []
        for ent in entities:
            decision = self.verify_entity_with_evidence(
                entity=ent.get("entity", ""),
                entity_type=ent.get("type", "unknown"),
                context=context,
                task_domain=task_domain
            )
            results.append(decision)
        
        return results