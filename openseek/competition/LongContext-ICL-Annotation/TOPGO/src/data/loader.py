"""
数据加载器
"""
import json
import jsonlines
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterator
from dataclasses import dataclass, field, asdict
from loguru import logger


@dataclass
class Document:
    """文档数据类"""
    doc_id: str
    text: str
    title: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    annotation: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        return cls(
            doc_id=data.get("doc_id", data.get("id", "")),
            text=data.get("text", data.get("content", "")),
            title=data.get("title"),
            metadata=data.get("metadata", {}),
            annotation=data.get("annotation", data.get("label", None))
        )


@dataclass
class Example:
    """示例数据类"""
    doc_id: str
    text_chunk: str
    annotation: Dict[str, Any]
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Example":
        return cls(
            doc_id=data.get("doc_id", ""),
            text_chunk=data.get("text_chunk", data.get("text", "")),
            annotation=data.get("annotation", {}),
            embedding=data.get("embedding", None)
        )


class DataLoader:
    """数据加载器"""
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        初始化数据加载器
        
        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent.parent.parent / "data"
        logger.info(f"数据加载器初始化，数据目录: {self.data_dir}")
    
    def load_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """
        加载JSONL文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            数据列表
        """
        path = self._resolve_path(file_path)
        data = []
        
        with jsonlines.open(path, 'r') as reader:
            for item in reader:
                data.append(item)
        
        logger.info(f"加载JSONL文件: {path}, 共 {len(data)} 条记录")
        return data
    
    def load_json(self, file_path: str) -> Dict[str, Any]:
        """
        加载JSON文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            数据字典
        """
        path = self._resolve_path(file_path)
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"加载JSON文件: {path}")
        return data
    
    def load_text(self, file_path: str) -> str:
        """
        加载文本文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            文本内容
        """
        path = self._resolve_path(file_path)
        
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        logger.info(f"加载文本文件: {path}")
        return text
    
    def load_documents(self, file_path: str) -> List[Document]:
        """
        加载文档列表
        
        Args:
            file_path: 文件路径
            
        Returns:
            文档列表
        """
        data = self.load_jsonl(file_path)
        documents = [Document.from_dict(item) for item in data]
        logger.info(f"加载 {len(documents)} 个文档")
        return documents
    
    def load_example_library(self, file_path: str) -> List[Example]:
        """
        加载示例库
        
        Args:
            file_path: 文件路径
            
        Returns:
            示例列表
        """
        data = self.load_jsonl(file_path)
        examples = [Example.from_dict(item) for item in data]
        logger.info(f"加载 {len(examples)} 个示例")
        return examples
    
    def save_jsonl(self, data: List[Dict[str, Any]], file_path: str):
        """
        保存为JSONL文件
        
        Args:
            data: 数据列表
            file_path: 文件路径
        """
        path = self._resolve_path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with jsonlines.open(path, 'w') as writer:
            for item in data:
                writer.write(item)
        
        logger.info(f"保存JSONL文件: {path}, 共 {len(data)} 条记录")
    
    def save_json(self, data: Dict[str, Any], file_path: str):
        """
        保存为JSON文件
        
        Args:
            data: 数据字典
            file_path: 文件路径
        """
        path = self._resolve_path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"保存JSON文件: {path}")
    
    def _resolve_path(self, file_path: str) -> Path:
        """解析路径"""
        path = Path(file_path)
        if path.is_absolute():
            return path
        return self.data_dir / file_path
    
    def iter_documents(self, file_path: str) -> Iterator[Document]:
        """
        迭代器方式加载文档（节省内存）
        
        Args:
            file_path: 文件路径
            
        Yields:
            文档对象
        """
        path = self._resolve_path(file_path)
        
        with jsonlines.open(path, 'r') as reader:
            for item in reader:
                yield Document.from_dict(item)
