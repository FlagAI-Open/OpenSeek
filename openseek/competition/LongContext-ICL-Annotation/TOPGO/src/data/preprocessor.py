"""
数据预处理器
"""
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
from pathlib import Path

from .loader import Document, Example


@dataclass
class TextChunk:
    """文本分块"""
    chunk_id: str
    text: str
    start_idx: int
    end_idx: int
    doc_id: str
    section_title: Optional[str] = None
    parent_section: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "doc_id": self.doc_id,
            "section_title": self.section_title,
            "parent_section": self.parent_section
        }


class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100
    ):
        """
        初始化预处理器
        
        Args:
            chunk_size: 分块大小（字符数）
            chunk_overlap: 分块重叠
            min_chunk_size: 最小分块大小
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        logger.info(f"数据预处理器初始化, chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    def clean_text(self, text: str) -> str:
        """
        清洗文本
        
        Args:
            text: 原始文本
            
        Returns:
            清洗后的文本
        """
        # 去除多余换行符
        text = re.sub(r'\n{3,}', '\n\n', text)
        # 规范化空格
        text = re.sub(r' {2,}', ' ', text)
        # 去除行首行尾空格
        text = '\n'.join(line.strip() for line in text.split('\n'))
        # 去除首尾空白
        text = text.strip()
        
        return text
    
    def extract_sections(self, text: str) -> List[Dict[str, Any]]:
        """
        提取章节结构（基于Markdown格式）
        
        Args:
            text: 文档文本
            
        Returns:
            章节列表
        """
        sections = []
        lines = text.split('\n')
        current_section = {
            "level": 0,
            "title": "root",
            "start": 0,
            "end": len(lines),
            "content": []
        }
        
        for i, line in enumerate(lines):
            # 匹配Markdown标题
            match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if match:
                # 保存上一个章节
                if current_section["content"]:
                    current_section["end"] = i
                    sections.append(current_section.copy())
                
                # 开始新章节
                level = len(match.group(1))
                title = match.group(2).strip()
                current_section = {
                    "level": level,
                    "title": title,
                    "start": i,
                    "end": len(lines),
                    "content": []
                }
            else:
                current_section["content"].append(line)
        
        # 添加最后一个章节
        if current_section["content"]:
            sections.append(current_section)
        
        logger.debug(f"提取到 {len(sections)} 个章节")
        return sections
    
    def chunk_document(
        self,
        document: Document,
        by_section: bool = True
    ) -> List[TextChunk]:
        """
        文档分块
        
        Args:
            document: 文档对象
            by_section: 是否按章节分块
            
        Returns:
            分块列表
        """
        chunks = []
        text = self.clean_text(document.text)
        
        if by_section:
            sections = self.extract_sections(text)
            
            for sec_idx, section in enumerate(sections):
                section_text = '\n'.join(section["content"])
                
                if len(section_text) < self.min_chunk_size:
                    # 小章节直接作为一个块
                    if section_text.strip():
                        chunks.append(TextChunk(
                            chunk_id=f"{document.doc_id}_sec{sec_idx}",
                            text=section_text,
                            start_idx=section["start"],
                            end_idx=section["end"],
                            doc_id=document.doc_id,
                            section_title=section["title"],
                            parent_section=None
                        ))
                else:
                    # 大章节按固定大小分块
                    section_chunks = self._chunk_text(
                        section_text,
                        f"{document.doc_id}_sec{sec_idx}"
                    )
                    for chunk in section_chunks:
                        chunk.section_title = section["title"]
                        chunks.append(chunk)
        else:
            chunks = self._chunk_text(text, document.doc_id)
        
        logger.debug(f"文档 {document.doc_id} 分块完成, 共 {len(chunks)} 个块")
        return chunks
    
    def _chunk_text(
        self,
        text: str,
        base_id: str
    ) -> List[TextChunk]:
        """
        按固定大小分块文本
        
        Args:
            text: 文本
            base_id: 基础ID
            
        Returns:
            分块列表
        """
        chunks = []
        start = 0
        chunk_idx = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # 尝试在句子边界切分
            if end < len(text):
                # 向后找句子边界
                boundary = self._find_sentence_boundary(text, end)
                if boundary > start + self.min_chunk_size:
                    end = boundary
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append(TextChunk(
                    chunk_id=f"{base_id}_chunk{chunk_idx}",
                    text=chunk_text,
                    start_idx=start,
                    end_idx=end,
                    doc_id=base_id.split('_')[0] if '_' in base_id else base_id
                ))
                chunk_idx += 1
            
            # 移动到下一个位置（考虑重叠）
            start = end - self.chunk_overlap if end < len(text) else end
        
        return chunks
    
    def _find_sentence_boundary(self, text: str, position: int) -> int:
        """
        查找句子边界
        
        Args:
            text: 文本
            position: 起始位置
            
        Returns:
            边界位置
        """
        # 向后查找句子结束标记
        for i in range(position, min(position + 100, len(text))):
            if text[i] in '。！？.!?\n':
                return i + 1
        
        # 向前查找
        for i in range(position, max(position - 100, 0), -1):
            if text[i] in '。！？.!?\n':
                return i + 1
        
        return position
    
    def build_example_library(
        self,
        documents: List[Document],
        save_path: Optional[str] = None
    ) -> List[Example]:
        """
        从标注文档构建示例库
        
        Args:
            documents: 已标注的文档列表
            save_path: 保存路径
            
        Returns:
            示例列表
        """
        examples = []
        
        for doc in documents:
            if doc.annotation is None:
                logger.warning(f"文档 {doc.doc_id} 没有标注，跳过")
                continue
            
            chunks = self.chunk_document(doc)
            
            for chunk in chunks:
                # 为每个块创建示例
                example = Example(
                    doc_id=doc.doc_id,
                    text_chunk=chunk.text,
                    annotation=self._extract_chunk_annotation(
                        chunk.text,
                        doc.annotation
                    )
                )
                examples.append(example)
        
        logger.info(f"构建示例库完成，共 {len(examples)} 个示例")
        
        if save_path:
            self._save_examples(examples, save_path)
        
        return examples
    
    def _extract_chunk_annotation(
        self,
        chunk_text: str,
        full_annotation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        提取分块相关的标注
        
        Args:
            chunk_text: 分块文本
            full_annotation: 完整标注
            
        Returns:
            分块标注
        """
        chunk_annotation = {
            "entities": [],
            "relations": []
        }
        
        # 提取在当前分块中的实体
        entities = full_annotation.get("entities", [])
        for entity in entities:
            entity_text = entity.get("text", "")
            if entity_text in chunk_text:
                chunk_annotation["entities"].append(entity)
        
        # 提取相关关系
        relations = full_annotation.get("relations", [])
        entity_texts = {e.get("text") for e in chunk_annotation["entities"]}
        for relation in relations:
            head = relation.get("head", "")
            tail = relation.get("tail", "")
            if head in entity_texts or tail in entity_texts:
                chunk_annotation["relations"].append(relation)
        
        return chunk_annotation
    
    def _save_examples(self, examples: List[Example], save_path: str):
        """保存示例库"""
        import jsonlines
        
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with jsonlines.open(path, 'w') as writer:
            for example in examples:
                writer.write(example.to_dict())
        
        logger.info(f"示例库已保存到: {save_path}")
    
    def process_dataset(
        self,
        train_docs: List[Document],
        test_docs: List[Document]
    ) -> Tuple[List[Example], List[Document]]:
        """
        处理数据集
        
        Args:
            train_docs: 训练文档
            test_docs: 测试文档
            
        Returns:
            (示例库, 测试文档列表)
        """
        # 构建示例库
        example_library = self.build_example_library(train_docs)
        
        # 处理测试文档
        processed_test_docs = []
        for doc in test_docs:
            doc.text = self.clean_text(doc.text)
            processed_test_docs.append(doc)
        
        logger.info(f"数据集处理完成，示例库: {len(example_library)}，测试文档: {len(processed_test_docs)}")
        
        return example_library, processed_test_docs
