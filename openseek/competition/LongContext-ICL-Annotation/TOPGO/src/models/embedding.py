"""
嵌入模型模块
"""
import numpy as np
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
from loguru import logger


class EmbeddingModel:
    """嵌入模型封装"""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        cache_dir: Optional[str] = None
    ):
        """
        初始化嵌入模型
        
        Args:
            model_name: 模型名称
            device: 计算设备
            cache_dir: 缓存目录
        """
        self.model_name = model_name
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._model = None
        self._cache: Dict[str, List[float]] = {}
        
        logger.info(f"嵌入模型初始化: {model_name}, 设备: {device}")
    
    def _load_model(self):
        """延迟加载模型"""
        if self._model is not None:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"加载嵌入模型: {self.model_name}")
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=str(self.cache_dir) if self.cache_dir else None
            )
            logger.info("嵌入模型加载完成")
        except ImportError:
            logger.warning("sentence_transformers未安装，使用简化嵌入")
            self._model = None
    
    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        编码文本为向量
        
        Args:
            texts: 文本列表
            batch_size: 批大小
            show_progress: 是否显示进度
            
        Returns:
            向量数组 [N, D]
        """
        self._load_model()
        
        if self._model is not None:
            # 使用真实模型
            embeddings = self._model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=True
            )
            return np.array(embeddings)
        else:
            # 使用简化嵌入（用于测试）
            return self._simple_encode(texts)
    
    def encode_single(self, text: str) -> List[float]:
        """
        编码单个文本
        
        Args:
            text: 文本
            
        Returns:
            向量列表
        """
        # 检查缓存
        if text in self._cache:
            return self._cache[text]
        
        embedding = self.encode([text])[0].tolist()
        self._cache[text] = embedding
        return embedding
    
    def _simple_encode(self, texts: List[str]) -> np.ndarray:
        """简化的编码方法（用于测试环境）"""
        # 使用简单的字符频率作为特征
        dim = 256
        embeddings = []
        
        for text in texts:
            # 基于字符hash的简化嵌入
            vec = np.zeros(dim)
            for i, char in enumerate(text):
                vec[hash(char) % dim] += 1
            
            # 归一化
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            
            embeddings.append(vec)
        
        return np.array(embeddings)
    
    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        计算余弦相似度
        
        Args:
            embedding1: 向量1
            embedding2: 向量2
            
        Returns:
            相似度
        """
        if embedding1.ndim == 1:
            embedding1 = embedding1.reshape(1, -1)
        if embedding2.ndim == 1:
            embedding2 = embedding2.reshape(1, -1)
        
        # 归一化
        norm1 = np.linalg.norm(embedding1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(embedding2, axis=1, keepdims=True)
        
        if norm1[0, 0] > 0:
            embedding1 = embedding1 / norm1
        if norm2[0, 0] > 0:
            embedding2 = embedding2 / norm2
        
        return float(np.dot(embedding1, embedding2.T)[0, 0])
    
    def compute_similarity_matrix(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray
    ) -> np.ndarray:
        """
        计算相似度矩阵
        
        Args:
            embeddings1: 向量矩阵1 [N, D]
            embeddings2: 向量矩阵2 [M, D]
            
        Returns:
            相似度矩阵 [N, M]
        """
        # 归一化
        norm1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)
        
        embeddings1_norm = embeddings1 / (norm1 + 1e-8)
        embeddings2_norm = embeddings2 / (norm2 + 1e-8)
        
        return np.dot(embeddings1_norm, embeddings2_norm.T)
    
    def save_cache(self, path: str):
        """保存嵌入缓存"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self._cache, f)
        
        logger.info(f"嵌入缓存已保存: {path}")
    
    def load_cache(self, path: str):
        """加载嵌入缓存"""
        path = Path(path)
        if not path.exists():
            logger.warning(f"缓存文件不存在: {path}")
            return
        
        with open(path, 'r', encoding='utf-8') as f:
            self._cache = json.load(f)
        
        logger.info(f"嵌入缓存已加载: {len(self._cache)} 条记录")
    
    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
        logger.info("嵌入缓存已清空")
