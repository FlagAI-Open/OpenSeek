#!/usr/bin/env python3
"""
四智能体协同流水线 (Four-Agent Pipeline)

协调管理员智能体(Orchestrator)、检索智能体(Retriever)、
编剧智能体(Playwright)、质检智能体(Quality Inspector)
完成长文档标注任务

类似于接力队协同工作：
1. Orchestrator: 规划任务，切分文档，分配工作记忆
2. Retriever: 从示例库检索相关示例
3. Playwright: 生成带Chain-of-Thought的提示剧本
4. Quality Inspector: 验证结果，维护知识图谱一致性

集成最佳实践：
- 重试机制 (Exponential backoff)
- 检索缓存 (Search caching)
- 性能监控 (Performance monitoring)
- 答案提取与归一化 (Answer extraction & normalization)
"""

import time
import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from loguru import logger

from .orchestrator import OrchestratorAgent, TaskPlan, TextChunk, WorkMemory
from .retriever import ExampleRetriever, RetrievedExample
from .playwright import PlaywrightAgent, PromptScript
from .quality_inspector import QualityInspectorAgent, ValidationResult
from .data.loader import Document
from .models.qwen_client import QwenClient
from .agent_utils import (
    retry_with_exponential_backoff,
    SearchCache,
    PerformanceMonitor,
    AnswerExtractor,
    AnswerNormalizer,
    AnswerValidator,
    ExecutionLogger,
    global_monitor
)


@dataclass
class PipelineConfig:
    """流水线配置"""
    max_context_tokens: int = 8000
    top_k_examples: int = 3
    similarity_threshold: float = 0.5
    max_validation_rounds: int = 2
    enable_cross_validation: bool = True
    enable_knowledge_graph: bool = True


@dataclass
class PipelineResult:
    """流水线结果"""
    doc_id: str
    task_description: str
    entities: List[Dict[str, Any]]
    relations: List[Dict[str, Any]]
    chunks_processed: int
    processing_time: float
    validation_results: List[ValidationResult]
    issues: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "task_description": self.task_description,
            "entities": self.entities,
            "relations": self.relations,
            "chunks_processed": self.chunks_processed,
            "processing_time": self.processing_time,
            "validation_results": [v.to_dict() for v in self.validation_results],
            "issues": self.issues
        }


class FourAgentPipeline:
    """
    四智能体协同流水线
    
    工作流程：
    1. Orchestrator接收任务和文档，进行任务规划和文档切分
    2. Retriever为每个文本块检索相关示例
    3. Playwright生成带Chain-of-Thought的提示剧本
    4. Model执行标注
    5. QualityInspector验证结果并维护知识图谱
    6. 汇总所有块的标注结果
    """
    
    def __init__(
        self,
        model_client: QwenClient,
        retriever: ExampleRetriever,
        config: Optional[PipelineConfig] = None
    ):
        """
        初始化四智能体流水线
        
        Args:
            model_client: 模型客户端
            retriever: 示例检索器
            config: 流水线配置
        """
        self.config = config or PipelineConfig()
        
        # 初始化各智能体
        self.orchestrator = OrchestratorAgent(
            max_context_tokens=self.config.max_context_tokens
        )
        self.retriever = retriever
        self.playwright = PlaywrightAgent()
        self.quality_inspector = QualityInspectorAgent(
            max_validation_rounds=self.config.max_validation_rounds
        )
        
        self.model_client = model_client
        
        # 初始化最佳实践组件
        self.retrieval_cache = SearchCache(max_size=500, default_ttl=3600)
        self.monitor = PerformanceMonitor()
        self.logger = ExecutionLogger(verbose=True)
        self.answer_validator = AnswerValidator()
        
        # 统计信息
        self.stats = {
            "total_documents": 0,
            "processed_documents": 0,
            "failed_documents": 0,
            "total_chunks": 0,
            "total_entities": 0,
            "total_relations": 0
        }
        
        logger.info("四智能体流水线初始化完成")
        logger.info(f"配置: max_tokens={self.config.max_context_tokens}, "
                   f"top_k={self.config.top_k_examples}, "
                   f"cross_val={self.config.enable_cross_validation}")
    
    def run(
        self,
        document: Document,
        task_description: str,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> PipelineResult:
        """
        运行完整流水线
        
        Args:
            document: 文档对象
            task_description: 任务描述
            progress_callback: 进度回调函数 (stage, current, total)
            
        Returns:
            流水线结果
        """
        start_time = time.time()
        self.stats["total_documents"] += 1
        
        logger.info(f"=" * 60)
        logger.info(f"开始处理文档: {document.doc_id}")
        logger.info(f"任务: {task_description}")
        logger.info(f"=" * 60)
        
        issues = []
        all_entities = []
        all_relations = []
        validation_results = []
        
        try:
            # ========== 阶段1: Orchestrator - 任务规划与文档切分 ==========
            if progress_callback:
                progress_callback("规划", 0, 10)
            
            logger.info("[阶段1] Orchestrator: 任务规划与文档切分")
            
            task_plan = self.orchestrator.plan_task(task_description, document)
            chunks = self.orchestrator.chunk_document(document, task_plan)
            work_memories = self.orchestrator.allocate_work_memory(chunks, task_plan)
            
            logger.info(f"文档切分为 {len(chunks)} 个块，分配为 {len(work_memories)} 个工作记忆")
            
            if progress_callback:
                progress_callback("切分", 3, 10)
            
            # ========== 阶段2: Retriever - 检索相关示例 ==========
            if progress_callback:
                progress_callback("检索", 3, 10)
            
            logger.info("[阶段2] Retriever: 检索相关示例")
            
            # 为每个块检索示例（带缓存）
            chunk_examples: Dict[str, List[RetrievedExample]] = {}
            for chunk in chunks:
                # 尝试从缓存获取
                cache_key = chunk.text[:100]  # 使用前100字符作为缓存键
                cached_examples = self.retrieval_cache.get(cache_key)
                
                if cached_examples is not None:
                    logger.debug(f"缓存命中: {chunk.chunk_id}")
                    chunk_examples[chunk.chunk_id] = cached_examples
                else:
                    # 执行检索
                    self.monitor.start_timer("retrieval_time")
                    examples = self.retriever.retrieve(
                        query_text=chunk.text,
                        top_k=self.config.top_k_examples
                    )
                    self.monitor.stop_timer("retrieval_time")
                    
                    # 存入缓存
                    self.retrieval_cache.set(cache_key, examples)
                    chunk_examples[chunk.chunk_id] = examples
            
            logger.info(f"示例检索完成，共为 {len(chunks)} 个块检索到示例")
            logger.debug(f"缓存统计: {self.retrieval_cache.get_stats()}")
            
            if progress_callback:
                progress_callback("示例", 5, 10)
            
            # ========== 阶段3: Playwright + Model - 生成提示并执行 ==========
            if progress_callback:
                progress_callback("标注", 5, 10)
            
            logger.info("[阶段3] Playwright + Model: 生成提示并执行标注")
            
            chunk_annotations: Dict[str, Dict[str, Any]] = {}
            
            for i, chunk in enumerate(chunks):
                logger.debug(f"处理块 {i+1}/{len(chunks)}: {chunk.chunk_id}")
                
                # 获取该块的示例
                examples = chunk_examples.get(chunk.chunk_id, [])
                
                # Playwright生成提示剧本
                script = self.playwright.generate_script(
                    task_plan=task_plan,
                    context_chunks=[chunk],
                    examples=examples
                )
                
                # 获取知识图谱上下文（用于实体一致性）
                kg_context = ""
                if self.config.enable_knowledge_graph:
                    kg_context = self.quality_inspector.get_knowledge_graph_context()
                
                # 构建完整提示
                full_prompt = self.playwright.build_full_prompt(
                    script=script,
                    context_text=chunk.text,
                    knowledge_graph_context=kg_context
                )
                
                # 调用模型（带重试机制）
                self.monitor.start_timer("model_call_time")
                response = self._call_model_with_retry(full_prompt)
                self.monitor.stop_timer("model_call_time")
                
                if response:
                    # 解析模型输出
                    annotation = self._parse_model_output(response)
                    chunk_annotations[chunk.chunk_id] = annotation
                else:
                    logger.error(f"模型调用失败: {chunk.chunk_id}")
                    issues.append(f"块{chunk.chunk_id}模型调用失败")
                    chunk_annotations[chunk.chunk_id] = {"entities": [], "relations": []}
                
                if progress_callback:
                    progress_callback("标注", 5 + int(4 * (i + 1) / len(chunks)), 10)
            
            # ========== 阶段4: QualityInspector - 验证与修正 ==========
            if progress_callback:
                progress_callback("验证", 9, 10)
            
            logger.info("[阶段4] QualityInspector: 验证与修正")
            
            for chunk_id, annotation in chunk_annotations.items():
                validation_result = self.quality_inspector.validate(
                    annotation=annotation,
                    task_plan=task_plan,
                    chunk_id=chunk_id
                )
                validation_results.append(validation_result)
                
                # 收集修正后的标注
                final_annotation = (
                    validation_result.corrected_annotation 
                    if validation_result.corrected_annotation 
                    else validation_result.original_annotation
                )
                
                all_entities.extend(final_annotation.get("entities", []))
                all_relations.extend(final_annotation.get("relations", []))
                
                # 记录问题
                issues.extend(validation_result.issues)
            
            # 跨块验证
            if self.config.enable_cross_validation and len(chunks) > 1:
                logger.info("执行跨块交叉验证")
                cross_results = self.quality_inspector.cross_validate(
                    annotations=[chunk_annotations[c.chunk_id] for c in chunks],
                    chunk_ids=[c.chunk_id for c in chunks]
                )
                validation_results.extend(cross_results)
            
            if progress_callback:
                progress_callback("完成", 10, 10)
            
            # 更新统计
            self.stats["processed_documents"] += 1
            self.stats["total_chunks"] += len(chunks)
            self.stats["total_entities"] += len(all_entities)
            self.stats["total_relations"] += len(all_relations)
            
            processing_time = time.time() - start_time
            
            logger.info(f"=" * 60)
            logger.info(f"文档处理完成: {document.doc_id}")
            logger.info(f"处理时间: {processing_time:.2f}秒")
            logger.info(f"提取实体: {len(all_entities)}个")
            logger.info(f"提取关系: {len(all_relations)}个")
            logger.info(f"验证问题: {len(issues)}个")
            logger.info(f"=" * 60)
            
            return PipelineResult(
                doc_id=document.doc_id,
                task_description=task_description,
                entities=all_entities,
                relations=all_relations,
                chunks_processed=len(chunks),
                processing_time=processing_time,
                validation_results=validation_results,
                issues=issues
            )
            
        except Exception as e:
            logger.error(f"流水线执行失败: {e}")
            self.stats["failed_documents"] += 1
            issues.append(f"流水线执行失败: {str(e)}")
            
            return PipelineResult(
                doc_id=document.doc_id,
                task_description=task_description,
                entities=[],
                relations=[],
                chunks_processed=0,
                processing_time=time.time() - start_time,
                validation_results=[],
                issues=issues
            )
    
    def run_batch(
        self,
        documents: List[Document],
        task_description: str,
        progress_callback: Optional[Callable[[str, int, int, int], None]] = None
    ) -> List[PipelineResult]:
        """
        批量运行流水线
        
        Args:
            documents: 文档列表
            task_description: 任务描述
            progress_callback: 进度回调 (stage, current, total, doc_id)
            
        Returns:
            结果列表
        """
        results = []
        
        for i, doc in enumerate(documents):
            logger.info(f"处理文档 {i+1}/{len(documents)}: {doc.doc_id}")
            
            if progress_callback:
                progress_callback("文档", i, len(documents), doc.doc_id)
            
            result = self.run(doc, task_description)
            results.append(result)
        
        return results
    
    @retry_with_exponential_backoff(max_attempts=3, base_delay=1.0, max_delay=10.0)
    def _call_model_with_retry(self, prompt: str) -> Optional[str]:
        """
        带重试的模型调用
        
        Args:
            prompt: 提示词
            
        Returns:
            模型响应或None
        """
        try:
            response = self.model_client.generate(
                prompt=prompt,
                temperature=0.1
            )
            return response
        except Exception as e:
            logger.warning(f"模型调用异常: {e}, 重试中...")
            raise  # 让装饰器处理重试
    
    def _parse_model_output(self, output: str) -> Dict[str, Any]:
        """
        解析模型输出（使用AnswerExtractor）
        
        Args:
            output: 模型原始输出
            
        Returns:
            解析后的标注结果
        """
        # 使用AnswerExtractor提取答案
        extracted = AnswerExtractor.extract(output)
        
        # 尝试解析为JSON
        try:
            # 如果提取的内容是JSON格式
            if extracted.startswith('{'):
                result = json.loads(extracted)
            else:
                # 尝试从原始输出中查找JSON
                json_start = output.find('{')
                if json_start >= 0:
                    depth = 0
                    json_end = json_start
                    for i, char in enumerate(output[json_start:]):
                        if char == '{':
                            depth += 1
                        elif char == '}':
                            depth -= 1
                            if depth == 0:
                                json_end = json_start + i + 1
                                break
                    
                    json_str = output[json_start:json_end]
                    result = json.loads(json_str)
                else:
                    result = {"entities": [], "relations": []}
            
            # 确保有entities和relations字段
            if "entities" not in result:
                result["entities"] = []
            if "relations" not in result:
                result["relations"] = []
            
            return result
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON解析失败: {e}")
            return {"entities": [], "relations": []}
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.stats.copy()
        stats["performance"] = self.monitor.report()
        stats["cache"] = self.retrieval_cache.get_stats()
        return stats
    
    def reset(self):
        """重置流水线状态"""
        self.quality_inspector.reset_knowledge_graph()
        self.retrieval_cache.clear()
        self.monitor.reset()
        self.stats = {
            "total_documents": 0,
            "processed_documents": 0,
            "failed_documents": 0,
            "total_chunks": 0,
            "total_entities": 0,
            "total_relations": 0
        }
        logger.info("流水线状态已重置")