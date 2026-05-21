#!/usr/bin/env python3
"""
质检与一致性智能体 (Quality Inspector Agent)
对编剧智能体产出的初步标注结果进行交叉验证，
利用预设规则和模型自检，自动修正矛盾，
并将关键实体和关系同步至全局知识图谱
"""

import json
import re
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from loguru import logger

from .orchestrator import TaskPlan
from .playwright import PromptScript


@dataclass
class EntityRecord:
    """实体记录"""
    text: str
    normalized_text: str  # 标准化后的文本（用于消歧）
    entity_type: str
    chunk_id: str
    start_pos: int
    end_pos: int
    aliases: Set[str] = field(default_factory=set)  # 别名集合
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "normalized_text": self.normalized_text,
            "entity_type": self.entity_type,
            "chunk_id": self.chunk_id,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "aliases": list(self.aliases)
        }


@dataclass
class RelationRecord:
    """关系记录"""
    head: str  # 头实体文本
    tail: str  # 尾实体文本
    relation_type: str
    chunk_id: str
    evidence: str = ""  # 文本证据
    reversed: bool = False  # 是否反向
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "head": self.head,
            "tail": self.tail,
            "relation_type": self.relation_type,
            "chunk_id": self.chunk_id,
            "evidence": self.evidence,
            "reversed": self.reversed
        }


@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    original_annotation: Dict[str, Any]
    corrected_annotation: Optional[Dict[str, Any]]
    issues: List[str]
    corrections: List[Dict[str, str]]  # 修正记录
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "original_annotation": self.original_annotation,
            "corrected_annotation": self.corrected_annotation,
            "issues": self.issues,
            "corrections": self.corrections
        }


class KnowledgeGraph:
    """
    全局知识图谱
    
    维护整个文档处理过程中的实体和关系一致性
    """
    
    def __init__(self):
        """初始化知识图谱"""
        self.entities: Dict[str, List[EntityRecord]] = defaultdict(list)  # normalized_text -> records
        self.relations: List[RelationRecord] = []
        self.entity_texts: Dict[str, str] = {}  # 别名 -> 规范名称
        
        logger.info("知识图谱初始化")
    
    def add_entity(self, entity: EntityRecord) -> str:
        """
        添加实体到知识图谱
        
        Returns:
            规范化的实体文本
        """
        normalized = self._normalize(entity.text)
        entity.normalized_text = normalized
        
        # 更新别名映射
        if normalized not in self.entity_texts:
            self.entity_texts[normalized] = entity.text
        
        # 添加到实体列表
        self.entities[normalized].append(entity)
        
        logger.debug(f"添加实体: {entity.text} -> {normalized}")
        
        return normalized
    
    def add_relation(self, relation: RelationRecord):
        """添加关系到知识图谱"""
        self.relations.append(relation)
        logger.debug(f"添加关系: {relation.head} --{relation.relation_type}--> {relation.tail}")
    
    def get_canonical_name(self, text: str) -> Optional[str]:
        """
        获取实体的规范名称
        
        Args:
            text: 实体文本
            
        Returns:
            规范名称，如果不存在则返回None
        """
        normalized = self._normalize(text)
        return self.entity_texts.get(normalized)
    
    def resolve_entity(self, text: str) -> str:
        """
        解析实体文本，返回规范名称
        
        Args:
            text: 实体文本
            
        Returns:
            规范名称（如果存在），否则返回原文本
        """
        canonical = self.get_canonical_name(text)
        return canonical if canonical else text
    
    def get_entity_type(self, text: str) -> Optional[str]:
        """
        获取实体的类型
        
        Args:
            text: 实体文本
            
        Returns:
            实体类型，如果不存在则返回None
        """
        normalized = self._normalize(text)
        records = self.entities.get(normalized, [])
        if records:
            return records[0].entity_type
        return None
    
    def check_consistency(self) -> List[str]:
        """
        检查知识图谱内部一致性
        
        Returns:
            一致性问题列表
        """
        issues = []
        
        # 检查同一实体是否有不同类型
        for normalized, records in self.entities.items():
            if len(records) > 1:
                types = set(r.entity_type for r in records)
                if len(types) > 1:
                    issues.append(
                        f"实体'{normalized}'在不同位置被标注为不同类型: {types}"
                    )
        
        return issues
    
    def _normalize(self, text: str) -> str:
        """标准化实体文本"""
        # 移除空格、换行等
        normalized = re.sub(r'\s+', '', text)
        # 转小写
        normalized = normalized.lower()
        return normalized
    
    def get_context_for_consistency(self) -> str:
        """
        生成用于实体一致性的上下文
        
        Returns:
            上下文字符串
        """
        if not self.entities:
            return ""
        
        lines = ["# 已标注实体（保持命名一致）"]
        
        for normalized, records in self.entities.items():
            canonical = self.entity_texts.get(normalized, normalized)
            entity_type = records[0].entity_type if records else "未知"
            lines.append(f"- {canonical} (类型: {entity_type})")
        
        return "\n".join(lines)
    
    def merge_duplicates(self) -> int:
        """
        合并重复实体
        
        Returns:
            合并的实体数量
        """
        merged = 0
        
        for normalized, records in list(self.entities.items()):
            if len(records) > 1:
                # 保留最完整的记录
                best = max(records, key=lambda r: r.end_pos - r.start_pos)
                
                # 收集所有别名
                for record in records:
                    if record.text != best.text:
                        best.aliases.add(record.text)
                
                # 更新实体列表
                self.entities[normalized] = [best]
                merged += 1
        
        return merged


class QualityInspectorAgent:
    """
    质检与一致性智能体
    
    核心职责：
    1. 交叉验证标注结果
    2. 自动修正矛盾
    3. 维护全局知识图谱
    4. 确保实体命名一致性
    """
    
    def __init__(self, max_validation_rounds: int = 2):
        """
        初始化质检智能体
        
        Args:
            max_validation_rounds: 最大验证轮数
        """
        self.max_validation_rounds = max_validation_rounds
        self.knowledge_graph = KnowledgeGraph()
        
        logger.info(f"质检智能体初始化, 最大验证轮数: {max_validation_rounds}")
    
    def validate(
        self,
        annotation: Dict[str, Any],
        task_plan: TaskPlan,
        chunk_id: str
    ) -> ValidationResult:
        """
        验证标注结果
        
        Args:
            annotation: 标注结果
            task_plan: 任务规划
            chunk_id: 块ID
            
        Returns:
            验证结果
        """
        logger.info(f"开始验证标注: {chunk_id}")
        
        issues = []
        corrections = []
        current_annotation = annotation.copy()
        
        # 1. 基础验证
        basic_issues = self._basic_validation(current_annotation, task_plan)
        issues.extend(basic_issues)
        
        # 2. 实体一致性验证
        entity_issues, entity_corrections = self._validate_entity_consistency(
            current_annotation, chunk_id
        )
        issues.extend(entity_issues)
        corrections.extend(entity_corrections)
        
        # 3. 关系一致性验证
        relation_issues, relation_corrections = self._validate_relation_consistency(
            current_annotation, chunk_id
        )
        issues.extend(relation_issues)
        corrections.extend(relation_corrections)
        
        # 4. 类型验证
        type_issues, type_corrections = self._validate_types(
            current_annotation, task_plan
        )
        issues.extend(type_issues)
        corrections.extend(type_corrections)
        
        # 5. 应用修正
        if corrections:
            current_annotation = self._apply_corrections(current_annotation, corrections)
        
        # 6. 更新知识图谱
        self._update_knowledge_graph(current_annotation, chunk_id)
        
        is_valid = len([i for i in issues if "错误" in i or "缺失" in i]) == 0
        
        result = ValidationResult(
            is_valid=is_valid,
            original_annotation=annotation,
            corrected_annotation=current_annotation if corrections else None,
            issues=issues,
            corrections=corrections
        )
        
        logger.info(
            f"验证完成: {chunk_id}, "
            f"有效={is_valid}, 问题数={len(issues)}, 修正数={len(corrections)}"
        )
        
        return result
    
    def cross_validate(
        self,
        annotations: List[Dict[str, Any]],
        chunk_ids: List[str]
    ) -> List[ValidationResult]:
        """
        跨块交叉验证
        
        Args:
            annotations: 多个块的标注结果
            chunk_ids: 块ID列表
            
        Returns:
            验证结果列表
        """
        logger.info(f"开始跨块交叉验证: {len(annotations)} 个块")
        
        results = []
        
        # 首先单独验证每个块
        for annotation, chunk_id in zip(annotations, chunk_ids):
            result = self.validate(annotation, None, chunk_id)
            results.append(result)
        
        # 然后进行跨块一致性检查
        cross_issues = self._cross_chunk_consistency(annotations, chunk_ids)
        
        # 合并跨块问题到结果中
        for i, result in enumerate(results):
            result.issues.extend(cross_issues.get(chunk_ids[i], []))
        
        # 检查知识图谱一致性
        kg_issues = self.knowledge_graph.check_consistency()
        if kg_issues:
            logger.warning(f"知识图谱一致性问题: {kg_issues}")
        
        return results
    
    def get_knowledge_graph_context(self) -> str:
        """
        获取知识图谱上下文（用于后续块的处理）
        
        Returns:
            知识图谱上下文字符串
        """
        return self.knowledge_graph.get_context_for_consistency()
    
    def reset_knowledge_graph(self):
        """重置知识图谱"""
        self.knowledge_graph = KnowledgeGraph()
        logger.info("知识图谱已重置")
    
    def _basic_validation(
        self,
        annotation: Dict[str, Any],
        task_plan: TaskPlan
    ) -> List[str]:
        """基础验证"""
        issues = []
        
        # 检查是否为空
        if not annotation:
            issues.append("标注结果为空")
            return issues
        
        # 检查entities字段
        entities = annotation.get("entities", [])
        if not entities:
            issues.append("警告：未提取到任何实体")
        
        # 检查relations字段
        relations = annotation.get("relations", [])
        
        # 检查实体格式
        for i, entity in enumerate(entities):
            if "text" not in entity:
                issues.append(f"实体{i}缺少text字段")
            if "type" not in entity:
                issues.append(f"实体{i}缺少type字段")
        
        # 检查关系格式
        for i, relation in enumerate(relations):
            if "head" not in relation:
                issues.append(f"关系{i}缺少head字段")
            if "tail" not in relation:
                issues.append(f"关系{i}缺少tail字段")
            if "relation" not in relation:
                issues.append(f"关系{i}缺少relation字段")
        
        return issues
    
    def _validate_entity_consistency(
        self,
        annotation: Dict[str, Any],
        chunk_id: str
    ) -> Tuple[List[str], List[Dict[str, str]]]:
        """实体一致性验证"""
        issues = []
        corrections = []
        
        entities = annotation.get("entities", [])
        seen_texts: Dict[str, str] = {}  # text -> normalized
        
        for entity in entities:
            text = entity.get("text", "")
            if not text:
                continue
            
            normalized = self.knowledge_graph._normalize(text)
            
            # 检查是否已存在
            if normalized in self.knowledge_graph.entity_texts:
                canonical = self.knowledge_graph.entity_texts[normalized]
                
                if canonical != text:
                    # 发现不一致，需要修正
                    corrections.append({
                        "type": "entity_name",
                        "from": text,
                        "to": canonical,
                        "reason": "实体命名不一致"
                    })
                    entity["text"] = canonical
            else:
                # 新实体，添加到知识图谱
                entity_record = EntityRecord(
                    text=text,
                    normalized_text=normalized,
                    entity_type=entity.get("type", "未知"),
                    chunk_id=chunk_id,
                    start_pos=entity.get("start", 0),
                    end_pos=entity.get("end", 0)
                )
                self.knowledge_graph.add_entity(entity_record)
        
        return issues, corrections
    
    def _validate_relation_consistency(
        self,
        annotation: Dict[str, Any],
        chunk_id: str
    ) -> Tuple[List[str], List[Dict[str, str]]]:
        """关系一致性验证"""
        issues = []
        corrections = []
        
        relations = annotation.get("relations", [])
        entities = {e.get("text"): e.get("type") for e in annotation.get("entities", [])}
        
        for i, relation in enumerate(relations):
            head = relation.get("head", "")
            tail = relation.get("tail", "")
            rel_type = relation.get("relation", "")
            
            # 检查头尾实体是否存在
            head_normalized = self.knowledge_graph._normalize(head)
            tail_normalized = self.knowledge_graph._normalize(tail)
            
            # 解析头尾实体
            resolved_head = self.knowledge_graph.resolve_entity(head)
            resolved_tail = self.knowledge_graph.resolve_entity(tail)
            
            if resolved_head != head:
                corrections.append({
                    "type": "relation_head",
                    "from": head,
                    "to": resolved_head,
                    "reason": "头实体命名不一致"
                })
                relation["head"] = resolved_head
            
            if resolved_tail != tail:
                corrections.append({
                    "type": "relation_tail",
                    "from": tail,
                    "to": resolved_tail,
                    "reason": "尾实体命名不一致"
                })
                relation["tail"] = resolved_tail
            
            # 添加关系到知识图谱
            relation_record = RelationRecord(
                head=resolved_head,
                tail=resolved_tail,
                relation_type=rel_type,
                chunk_id=chunk_id
            )
            self.knowledge_graph.add_relation(relation_record)
        
        return issues, corrections
    
    def _validate_types(
        self,
        annotation: Dict[str, Any],
        task_plan: TaskPlan
    ) -> Tuple[List[str], List[Dict[str, str]]]:
        """类型验证"""
        issues = []
        corrections = []
        
        if not task_plan:
            return issues, corrections
        
        valid_entity_types = set(task_plan.target_entity_types)
        valid_relation_types = set(task_plan.target_relation_types)
        
        # 验证实体类型
        for entity in annotation.get("entities", []):
            entity_type = entity.get("type", "")
            if entity_type and entity_type not in valid_entity_types:
                issues.append(f"实体类型'{entity_type}'不在允许范围内")
        
        # 验证关系类型
        for relation in annotation.get("relations", []):
            rel_type = relation.get("relation", "")
            if rel_type and rel_type not in valid_relation_types:
                issues.append(f"关系类型'{rel_type}'不在允许范围内")
        
        return issues, corrections
    
    def _cross_chunk_consistency(
        self,
        annotations: List[Dict[str, Any]],
        chunk_ids: List[str]
    ) -> Dict[str, List[str]]:
        """跨块一致性检查"""
        issues_map: Dict[str, List[str]] = {cid: [] for cid in chunk_ids}
        
        # 收集所有实体
        all_entities: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
        # normalized_text -> [(chunk_id, text, type)]
        
        for annotation, chunk_id in zip(annotations, chunk_ids):
            for entity in annotation.get("entities", []):
                text = entity.get("text", "")
                entity_type = entity.get("type", "")
                normalized = self.knowledge_graph._normalize(text)
                
                all_entities[normalized].append((chunk_id, text, entity_type))
        
        # 检查同一实体在不同块中的类型是否一致
        for normalized, occurrences in all_entities.items():
            if len(occurrences) > 1:
                types = set(t for _, _, t in occurrences)
                if len(types) > 1:
                    # 类型不一致
                    for chunk_id, text, _ in occurrences:
                        issues_map[chunk_id].append(
                            f"实体'{text}'在不同位置被标注为不同类型: {types}"
                        )
        
        return issues_map
    
    def _apply_corrections(
        self,
        annotation: Dict[str, Any],
        corrections: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """应用修正"""
        annotation = json.loads(json.dumps(annotation))  # 深拷贝
        
        for correction in corrections:
            corr_type = correction.get("type", "")
            from_text = correction.get("from", "")
            to_text = correction.get("to", "")
            
            if corr_type == "entity_name":
                for entity in annotation.get("entities", []):
                    if entity.get("text") == from_text:
                        entity["text"] = to_text
            
            elif corr_type == "relation_head":
                for relation in annotation.get("relations", []):
                    if relation.get("head") == from_text:
                        relation["head"] = to_text
            
            elif corr_type == "relation_tail":
                for relation in annotation.get("relations", []):
                    if relation.get("tail") == from_text:
                        relation["tail"] = to_text
        
        return annotation
    
    def _update_knowledge_graph(
        self,
        annotation: Dict[str, Any],
        chunk_id: str
    ):
        """更新知识图谱"""
        # 添加实体
        for entity in annotation.get("entities", []):
            entity_record = EntityRecord(
                text=entity.get("text", ""),
                normalized_text=self.knowledge_graph._normalize(entity.get("text", "")),
                entity_type=entity.get("type", "未知"),
                chunk_id=chunk_id,
                start_pos=entity.get("start", 0),
                end_pos=entity.get("end", 0)
            )
            self.knowledge_graph.add_entity(entity_record)
        
        # 添加关系
        for relation in annotation.get("relations", []):
            relation_record = RelationRecord(
                head=relation.get("head", ""),
                tail=relation.get("tail", ""),
                relation_type=relation.get("relation", ""),
                chunk_id=chunk_id
            )
            self.knowledge_graph.add_relation(relation_record)