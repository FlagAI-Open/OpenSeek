#!/usr/bin/env python3
"""
管理员智能体 (Orchestrator Agent)
负责任务规划与上下文门控 - 动态分析文档长度与结构，
将其切割为语义连贯的段落块，并为后续智能体分配合适的工作记忆
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger

from .data.loader import Document


@dataclass
class TextChunk:
    """文本块"""
    chunk_id: str
    text: str
    start_pos: int
    end_pos: int
    section_path: List[str]
    chunk_type: str  # "heading", "paragraph", "list", "table"
    semantic_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "section_path": self.section_path,
            "chunk_type": self.chunk_type,
            "semantic_score": self.semantic_score,
            "metadata": self.metadata
        }


@dataclass
class TaskPlan:
    """任务规划"""
    task_id: str
    task_description: str
    target_entity_types: List[str]
    target_relation_types: List[str]
    estimated_chunks: int
    priority: str  # "high", "medium", "low"
    constraints: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_description": self.task_description,
            "target_entity_types": self.target_entity_types,
            "target_relation_types": self.target_relation_types,
            "estimated_chunks": self.estimated_chunks,
            "priority": self.priority,
            "constraints": self.constraints
        }


@dataclass 
class WorkMemory:
    """工作记忆 - 分配给每个智能体的上下文窗口"""
    max_tokens: int
    used_tokens: int
    remaining_tokens: int
    chunks: List[TextChunk]
    context_summary: str
    
    def can_fit(self, additional_tokens: int) -> bool:
        return (self.remaining_tokens - additional_tokens) >= 0
    
    def add_chunk(self, chunk: TextChunk, estimated_tokens: int) -> bool:
        if self.can_fit(estimated_tokens):
            self.chunks.append(chunk)
            self.used_tokens += estimated_tokens
            self.remaining_tokens -= estimated_tokens
            return True
        return False


class OrchestratorAgent:
    """
    管理员智能体
    
    核心职责：
    1. 任务规划 - 分析标注任务，生成执行计划
    2. 上下文门控 - 动态分配工作记忆，管理上下文窗口
    3. 文档切分 - 将长文档切割为语义连贯的段落块
    """
    
    # 任务类型到实体/关系类型的映射
    TASK_TYPE_MAPPINGS = {
        "responsibility": {
            "entity_types": ["责任主体", "责任内容", "责任对象", "责任范围"],
            "relation_types": ["承担", "履行", "转移", "免除"],
            "priority": "high"
        },
        "entity_extraction": {
            "entity_types": ["方法", "技术", "数据集", "指标", "机构", "人员"],
            "relation_types": ["使用", "属于", "包含", "依赖"],
            "priority": "medium"
        },
        "relation_extraction": {
            "entity_types": ["主体", "客体"],
            "relation_types": ["关联", "因果", "时序", "包含"],
            "priority": "medium"
        },
        "general": {
            "entity_types": ["实体", "概念"],
            "relation_types": ["关系"],
            "priority": "low"
        }
    }
    
    def __init__(
        self,
        max_context_tokens: int = 8000,
        min_chunk_size: int = 100,
        max_chunk_size: int = 2000,
        overlap_tokens: int = 200
    ):
        """
        初始化管理员智能体
        
        Args:
            max_context_tokens: 最大上下文token数
            min_chunk_size: 最小块大小（字符）
            max_chunk_size: 最大块大小（字符）
            overlap_tokens: 重叠token数
        """
        self.max_context_tokens = max_context_tokens
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_tokens = overlap_tokens
        
        # 估算：1个token约等于4个中文字符或0.75个英文单词
        self.chars_per_token = 4
        
        logger.info(
            f"管理员智能体初始化: "
            f"max_context={max_context_tokens} tokens, "
            f"chunk_size=[{min_chunk_size}, {max_chunk_size}] chars"
        )
    
    def plan_task(
        self,
        task_description: str,
        document: Document
    ) -> TaskPlan:
        """
        任务规划 - 分析任务描述和文档，生成执行计划
        
        Args:
            task_description: 任务描述（如"找出所有责任条款及其主体"）
            document: 文档对象
            
        Returns:
            任务规划对象
        """
        logger.info(f"开始任务规划: {task_description[:50]}...")
        
        # 1. 分析任务类型
        task_type = self._classify_task(task_description)
        type_info = self.TASK_TYPE_MAPPINGS.get(task_type, self.TASK_TYPE_MAPPINGS["general"])
        
        # 2. 估算文档需要切分的块数
        estimated_chunks = self._estimate_chunks_needed(document, task_type)
        
        # 3. 确定优先级
        priority = type_info["priority"]
        
        # 4. 生成任务ID
        task_id = f"task_{document.doc_id}_{hash(task_description) % 10000}"
        
        task_plan = TaskPlan(
            task_id=task_id,
            task_description=task_description,
            target_entity_types=type_info["entity_types"],
            target_relation_types=type_info["relation_types"],
            estimated_chunks=estimated_chunks,
            priority=priority,
            constraints={
                "task_type": task_type,
                "max_chunk_size": self.max_chunk_size,
                "min_chunk_size": self.min_chunk_size
            }
        )
        
        logger.info(
            f"任务规划完成: {task_id}, "
            f"类型={task_type}, 预估块数={estimated_chunks}, 优先级={priority}"
        )
        
        return task_plan
    
    def chunk_document(
        self,
        document: Document,
        task_plan: TaskPlan
    ) -> List[TextChunk]:
        """
        文档切分 - 将长文档切割为语义连贯的段落块
        
        Args:
            document: 文档对象
            task_plan: 任务规划
            
        Returns:
            文本块列表
        """
        logger.info(f"开始文档切分: {document.doc_id}")
        
        text = document.text
        chunks = []
        
        # 1. 解析文档结构
        sections = self._parse_document_structure(text)
        
        # 2. 基于结构切分
        if sections:
            chunks = self._split_by_structure(text, document.doc_id, sections)
        else:
            chunks = self._split_by_size(text, document.doc_id)
        
        # 3. 为每个块分配section_path
        chunks = self._assign_section_paths(chunks, sections)
        
        # 4. 评估每个块的语义相关性
        chunks = self._score_chunks(chunks, task_plan)
        
        logger.info(f"文档切分完成: {document.doc_id}, 共 {len(chunks)} 个块")
        
        return chunks
    
    def allocate_work_memory(
        self,
        chunks: List[TextChunk],
        task_plan: TaskPlan
    ) -> List[WorkMemory]:
        """
        分配工作记忆 - 为后续智能体分配合适的上下文窗口
        
        Args:
            chunks: 文本块列表
            task_plan: 任务规划
            
        Returns:
            工作记忆列表
        """
        logger.info(f"开始分配工作记忆: {len(chunks)} 个块")
        
        # 按语义得分排序
        sorted_chunks = sorted(chunks, key=lambda x: x.semantic_score, reverse=True)
        
        work_memories = []
        current_memory = WorkMemory(
            max_tokens=self.max_context_tokens,
            used_tokens=0,
            remaining_tokens=self.max_context_tokens,
            chunks=[],
            context_summary=""
        )
        
        for chunk in sorted_chunks:
            estimated_tokens = len(chunk.text) // self.chars_per_token
            
            if not current_memory.can_fit(estimated_tokens):
                # 当前工作记忆已满，创建新的
                current_memory.context_summary = self._summarize_memory(current_memory.chunks)
                work_memories.append(current_memory)
                current_memory = WorkMemory(
                    max_tokens=self.max_context_tokens,
                    used_tokens=0,
                    remaining_tokens=self.max_context_tokens,
                    chunks=[],
                    context_summary=""
                )
            
            current_memory.add_chunk(chunk, estimated_tokens)
        
        # 添加最后一个工作记忆
        if current_memory.chunks:
            current_memory.context_summary = self._summarize_memory(current_memory.chunks)
            work_memories.append(current_memory)
        
        logger.info(f"工作记忆分配完成: {len(work_memories)} 个上下文窗口")
        
        return work_memories
    
    def _classify_task(self, task_description: str) -> str:
        """分类任务类型"""
        task_lower = task_description.lower()
        
        if any(kw in task_lower for kw in ["责任", "义务", "承担", "履行"]):
            return "responsibility"
        elif any(kw in task_lower for kw in ["抽取", "提取", "识别", "找出"]):
            return "entity_extraction"
        elif any(kw in task_lower for kw in ["关系", "关联", "联系"]):
            return "relation_extraction"
        else:
            return "general"
    
    def _estimate_chunks_needed(self, document: Document, task_type: str) -> int:
        """估算需要的块数"""
        text_length = len(document.text)
        
        # 根据任务类型调整块大小
        if task_type == "responsibility":
            avg_chunk_size = self.max_chunk_size * 0.8  # 责任条款需要更细粒度
        else:
            avg_chunk_size = self.max_chunk_size * 0.9
        
        estimated = max(1, text_length // int(avg_chunk_size))
        
        return min(estimated, 50)  # 最多50个块
    
    def _parse_document_structure(self, text: str) -> List[Dict[str, Any]]:
        """解析文档结构"""
        sections = []
        
        # 匹配标题（# 开头的Markdown格式或数字标题）
        heading_patterns = [
            r'^#{1,6}\s+(.+)$',  # Markdown # 标题
            r'^(\d+\.)+\s+(.+)$',  # 数字标题 1. 2. 3.
            r'^【(.+?)】$',  # 【标题】格式
        ]
        
        lines = text.split('\n')
        current_pos = 0
        
        for line in lines:
            for pattern in heading_patterns:
                match = re.match(pattern, line.strip())
                if match:
                    title = match.group(1) if match.lastindex else match.group(0)
                    sections.append({
                        "title": title,
                        "position": current_pos,
                        "level": len(match.group(0)) - len(match.group(0).lstrip('#'))
                    })
                    break
            
            current_pos += len(line) + 1  # +1 for newline
        
        return sections
    
    def _split_by_structure(
        self,
        text: str,
        doc_id: str,
        sections: List[Dict[str, Any]]
    ) -> List[TextChunk]:
        """基于文档结构切分"""
        chunks = []
        
        if not sections:
            return self._split_by_size(text, doc_id)
        
        # 按位置排序
        sections = sorted(sections, key=lambda x: x["position"])
        
        for i, section in enumerate(sections):
            start = section["position"]
            
            # 确定结束位置
            if i + 1 < len(sections):
                end = sections[i + 1]["position"]
            else:
                end = len(text)
            
            section_text = text[start:end].strip()
            
            if len(section_text) < self.min_chunk_size:
                continue
            
            # 如果段落太长，进一步切分
            if len(section_text) > self.max_chunk_size:
                sub_chunks = self._split_large_chunk(section_text, doc_id, i)
                chunks.extend(sub_chunks)
            else:
                chunk = TextChunk(
                    chunk_id=f"{doc_id}_chunk_{i}",
                    text=section_text,
                    start_pos=start,
                    end_pos=end,
                    section_path=[section["title"]],
                    chunk_type="heading"
                )
                chunks.append(chunk)
        
        return chunks
    
    def _split_by_size(self, text: str, doc_id: str) -> List[TextChunk]:
        """基于大小切分"""
        chunks = []
        start = 0
        chunk_idx = 0
        
        while start < len(text):
            end = min(start + self.max_chunk_size, len(text))
            
            # 尝试在句子边界切分
            if end < len(text):
                # 向前查找最后一个句号或换行
                for sep in ['。', '！', '？', '\n']:
                    last_sep = text.rfind(sep, start + self.max_chunk_size // 2, end)
                    if last_sep > start + self.max_chunk_size // 2:
                        end = last_sep + 1
                        break
            
            chunk_text = text[start:end].strip()
            
            if len(chunk_text) >= self.min_chunk_size:
                chunk = TextChunk(
                    chunk_id=f"{doc_id}_chunk_{chunk_idx}",
                    text=chunk_text,
                    start_pos=start,
                    end_pos=end,
                    section_path=[],
                    chunk_type="paragraph"
                )
                chunks.append(chunk)
                chunk_idx += 1
            
            # 移动窗口（考虑重叠）
            start = end - self.overlap_tokens // self.chars_per_token
            start = max(start, end - self.min_chunk_size)
        
        return chunks
    
    def _split_large_chunk(
        self,
        text: str,
        doc_id: str,
        section_idx: int
    ) -> List[TextChunk]:
        """切分大块文本"""
        chunks = []
        start = 0
        sub_idx = 0
        
        while start < len(text):
            end = min(start + self.max_chunk_size, len(text))
            
            if end < len(text):
                for sep in ['。', '！', '？', '\n']:
                    last_sep = text.rfind(sep, start + self.max_chunk_size // 2, end)
                    if last_sep > start + self.max_chunk_size // 2:
                        end = last_sep + 1
                        break
            
            chunk_text = text[start:end].strip()
            
            if len(chunk_text) >= self.min_chunk_size:
                chunk = TextChunk(
                    chunk_id=f"{doc_id}_chunk_{section_idx}_{sub_idx}",
                    text=chunk_text,
                    start_pos=start,
                    end_pos=end,
                    section_path=[],
                    chunk_type="paragraph"
                )
                chunks.append(chunk)
                sub_idx += 1
            
            start = end - self.overlap_tokens // self.chars_per_token
            start = max(start, end - self.min_chunk_size)
        
        return chunks
    
    def _assign_section_paths(
        self,
        chunks: List[TextChunk],
        sections: List[Dict[str, Any]]
    ) -> List[TextChunk]:
        """为每个块分配章节路径"""
        if not sections:
            return chunks
        
        for chunk in chunks:
            # 找到chunk所在位置之前的最后一个section
            path = []
            for section in sections:
                if section["position"] <= chunk.start_pos:
                    path.append(section["title"])
                else:
                    break
            
            chunk.section_path = path
        
        return chunks
    
    def _score_chunks(
        self,
        chunks: List[TextChunk],
        task_plan: TaskPlan
    ) -> List[TextChunk]:
        """评估每个块的语义相关性"""
        # 提取关键词
        keywords = self._extract_keywords(task_plan)
        
        for chunk in chunks:
            score = 0.0
            text_lower = chunk.text.lower()
            
            # 标题匹配
            for section in chunk.section_path:
                if any(kw in section.lower() for kw in keywords):
                    score += 2.0
            
            # 内容匹配
            for kw in keywords:
                if kw in text_lower:
                    score += 1.0
            
            # 长度惩罚（太长或太短都降低分数）
            length = len(chunk.text)
            if length < 200:
                score *= 0.8
            elif length > 1500:
                score *= 0.9
            
            chunk.semantic_score = score
        
        return chunks
    
    def _extract_keywords(self, task_plan: TaskPlan) -> List[str]:
        """提取任务关键词"""
        keywords = []
        
        # 从任务描述提取
        desc_keywords = re.findall(r'[\w]{2,}', task_plan.task_description.lower())
        keywords.extend(desc_keywords[:5])  # 取前5个
        
        # 从实体类型提取
        keywords.extend([et.lower() for et in task_plan.target_entity_types])
        
        return list(set(keywords))
    
    def _summarize_memory(self, chunks: List[TextChunk]) -> str:
        """生成工作记忆摘要"""
        if not chunks:
            return ""
        
        summaries = []
        for chunk in chunks[:3]:  # 只取前3个块
            section = chunk.section_path[-1] if chunk.section_path else "未分类"
            summaries.append(f"[{section}] {chunk.text[:100]}...")
        
        return " | ".join(summaries)