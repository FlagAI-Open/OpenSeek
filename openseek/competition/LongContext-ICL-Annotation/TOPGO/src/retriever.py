"""
动态Few-shot示例检索模块
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

from .models.embedding import EmbeddingModel
from .data.loader import Example


@dataclass
class RetrievedExample:
    """检索到的示例"""
    example: Example
    similarity: float
    rank: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text_chunk": self.example.text_chunk,
            "annotation": self.example.annotation,
            "similarity": self.similarity,
            "rank": self.rank
        }


class ExampleRetriever:
    """动态示例检索器"""
    
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        top_k: int = 3,
        similarity_threshold: float = 0.5
    ):
        """
        初始化检索器
        
        Args:
            embedding_model: 嵌入模型
            top_k: 返回Top-K个示例
            similarity_threshold: 相似度阈值
        """
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        
        self.example_library: List[Example] = []
        self.example_embeddings: Optional[np.ndarray] = None
        
        logger.info(f"示例检索器初始化, top_k={top_k}, threshold={similarity_threshold}")
    
    def build_index(self, examples: List[Example]):
        """
        构建示例库索引
        
        Args:
            examples: 示例列表
        """
        self.example_library = examples
        
        # 检查是否有预计算的嵌入
        texts_to_encode = []
        indices_to_encode = []
        
        for i, example in enumerate(examples):
            if example.embedding is not None:
                # 使用预计算的嵌入
                if self.example_embeddings is None:
                    # 获取嵌入维度
                    dim = len(example.embedding)
                    self.example_embeddings = np.zeros((len(examples), dim))
                
                self.example_embeddings[i] = np.array(example.embedding)
            else:
                texts_to_encode.append(example.text_chunk)
                indices_to_encode.append(i)
        
        # 编码没有预计算嵌入的文本
        if texts_to_encode:
            logger.info(f"正在编码 {len(texts_to_encode)} 个示例...")
            new_embeddings = self.embedding_model.encode(
                texts_to_encode,
                show_progress=True
            )
            
            if self.example_embeddings is None:
                self.example_embeddings = new_embeddings
            else:
                for idx, emb in zip(indices_to_encode, new_embeddings):
                    self.example_embeddings[idx] = emb
        
        logger.info(f"示例库索引构建完成, 共 {len(self.example_library)} 个示例")
    
    def retrieve(
        self,
        query_text: str,
        top_k: Optional[int] = None
    ) -> List[RetrievedExample]:
        """
        检索相关示例
        
        Args:
            query_text: 查询文本
            top_k: 返回数量，默认使用初始化时的值
            
        Returns:
            检索到的示例列表
        """
        if not self.example_library:
            logger.warning("示例库为空，无法检索")
            return []
        
        top_k = top_k or self.top_k
        
        # 编码查询
        query_embedding = self.embedding_model.encode([query_text])[0]
        
        # 计算相似度
        similarities = self.embedding_model.compute_similarity_matrix(
            query_embedding.reshape(1, -1),
            self.example_embeddings
        )[0]
        
        # 排序并选取Top-K
        sorted_indices = np.argsort(similarities)[::-1]
        
        results = []
        for rank, idx in enumerate(sorted_indices[:top_k]):
            similarity = similarities[idx]
            
            if similarity < self.similarity_threshold:
                continue
            
            results.append(RetrievedExample(
                example=self.example_library[idx],
                similarity=float(similarity),
                rank=rank + 1
            ))
        
        logger.debug(f"检索完成，返回 {len(results)} 个示例")
        return results
    
    def retrieve_batch(
        self,
        query_texts: List[str],
        top_k: Optional[int] = None
    ) -> List[List[RetrievedExample]]:
        """
        批量检索
        
        Args:
            query_texts: 查询文本列表
            top_k: 返回数量
            
        Returns:
            每个查询的检索结果列表
        """
        top_k = top_k or self.top_k
        
        # 批量编码
        query_embeddings = self.embedding_model.encode(query_texts)
        
        # 计算相似度矩阵
        similarity_matrix = self.embedding_model.compute_similarity_matrix(
            query_embeddings,
            self.example_embeddings
        )
        
        # 为每个查询选取Top-K
        all_results = []
        for i, similarities in enumerate(similarity_matrix):
            sorted_indices = np.argsort(similarities)[::-1]
            
            results = []
            for rank, idx in enumerate(sorted_indices[:top_k]):
                similarity = similarities[idx]
                
                if similarity < self.similarity_threshold:
                    continue
                
                results.append(RetrievedExample(
                    example=self.example_library[idx],
                    similarity=float(similarity),
                    rank=rank + 1
                ))
            
            all_results.append(results)
        
        logger.debug(f"批量检索完成，处理 {len(query_texts)} 个查询")
        return all_results
    
    def add_example(self, example: Example, compute_embedding: bool = True):
        """
        添加新示例
        
        Args:
            example: 新示例
            compute_embedding: 是否计算嵌入
        """
        self.example_library.append(example)
        
        if compute_embedding:
            embedding = self.embedding_model.encode([example.text_chunk])[0]
            if self.example_embeddings is None:
                self.example_embeddings = embedding.reshape(1, -1)
            else:
                self.example_embeddings = np.vstack([
                    self.example_embeddings,
                    embedding
                ])
        
        logger.debug(f"添加示例，当前库大小: {len(self.example_library)}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取检索器统计信息
        
        Returns:
            统计信息字典
        """
        return {
            "total_examples": len(self.example_library),
            "embedding_dim": self.example_embeddings.shape[1] if self.example_embeddings is not None else 0,
            "top_k": self.top_k,
            "similarity_threshold": self.similarity_threshold
        }
    
    def format_examples_for_prompt(
        self,
        examples: List[RetrievedExample],
        include_annotation: bool = True
    ) -> str:
        """
        格式化示例为提示词
        
        Args:
            examples: 示例列表
            include_annotation: 是否包含标注
            
        Returns:
            格式化后的字符串
        """
        if not examples:
            return ""
        
        parts = []
        for i, retrieved in enumerate(examples):
            example = retrieved.example
            parts.append(f"示例 {i + 1}:")
            parts.append(f"文本: {example.text_chunk[:200]}...")
            
            if include_annotation and example.annotation:
                parts.append(f"标注: {example.annotation}")
            
            parts.append("")  # 空行分隔
        
        return '\n'.join(parts)
