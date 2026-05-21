"""
分层级上下文动态构建模块
"""
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

from .data.loader import Document


@dataclass
class ContextWindow:
    """上下文窗口"""
    summary: str  # 文档摘要
    section_path: List[str]  # 章节路径（从根到当前章节）
    target_text: str  # 目标文本
    preceding_context: str  # 前文上下文
    following_context: str  # 后文上下文
    full_context: str  # 完整上下文
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "section_path": self.section_path,
            "target_text": self.target_text,
            "preceding_context": self.preceding_context,
            "following_context": self.following_context,
            "full_context": self.full_context
        }


class ContextBuilder:
    """分层级上下文动态构建器"""
    
    def __init__(
        self,
        summary_length: int = 200,
        neighbor_paragraphs_before: int = 1,
        neighbor_paragraphs_after: int = 1,
        max_context_length: int = 8000
    ):
        """
        初始化上下文构建器
        
        Args:
            summary_length: 摘要长度
            neighbor_paragraphs_before: 目标段落前N段
            neighbor_paragraphs_after: 目标段落后M段
            max_context_length: 最大上下文长度
        """
        self.summary_length = summary_length
        self.neighbor_paragraphs_before = neighbor_paragraphs_before
        self.neighbor_paragraphs_after = neighbor_paragraphs_after
        self.max_context_length = max_context_length
        self._summary_cache: Dict[str, str] = {}
        
        logger.info(
            f"上下文构建器初始化, "
            f"摘要长度={summary_length}, "
            f"前N段={neighbor_paragraphs_before}, "
            f"后M段={neighbor_paragraphs_after}"
        )
    
    def build_context(
        self,
        document: Document,
        target_start: int,
        target_end: int,
        model_client: Optional[Any] = None,
        task_hint: Optional[str] = None
    ) -> ContextWindow:
        """
        构建上下文窗口
        
        Args:
            document: 文档对象
            target_start: 目标文本起始位置
            target_end: 目标文本结束位置
            model_client: 模型客户端（用于生成摘要）
            task_hint: 任务提示（用于定位相关段落）
            
        Returns:
            上下文窗口
        """
        text = document.text
        
        # 1. 获取或生成文档摘要
        summary = self._get_or_generate_summary(document, model_client)
        
        # 2. 解析章节结构
        sections = self._parse_sections(text)
        
        # 3. 定位目标段落所在章节
        section_path = self._get_section_path(sections, target_start, target_end)
        
        # 4. 按段落分割
        paragraphs = self._split_paragraphs(text)
        
        # 5. 找到目标段落索引
        target_para_idx = self._find_paragraph_index(paragraphs, target_start)
        
        # 6. 收集前后段落
        preceding_paras = paragraphs[
            max(0, target_para_idx - self.neighbor_paragraphs_before):
            target_para_idx
        ]
        following_paras = paragraphs[
            target_para_idx + 1:
            target_para_idx + 1 + self.neighbor_paragraphs_after
        ]
        
        # 7. 提取目标文本
        target_text = text[target_start:target_end]
        
        # 8. 组装上下文
        full_context = self._assemble_context(
            summary=summary,
            section_path=section_path,
            preceding_context='\n\n'.join(preceding_paras),
            target_text=target_text,
            following_context='\n\n'.join(following_paras)
        )
        
        # 9. 长度控制
        full_context = self._truncate_context(full_context)
        
        return ContextWindow(
            summary=summary,
            section_path=section_path,
            target_text=target_text,
            preceding_context='\n\n'.join(preceding_paras),
            following_context='\n\n'.join(following_paras),
            full_context=full_context
        )
    
    def build_context_for_chunk(
        self,
        document: Document,
        chunk_text: str,
        model_client: Optional[Any] = None
    ) -> ContextWindow:
        """
        为文本分块构建上下文
        
        Args:
            document: 文档对象
            chunk_text: 分块文本
            model_client: 模型客户端
            
        Returns:
            上下文窗口
        """
        # 在文档中定位分块
        text = document.text
        
        # 尝试精确匹配
        idx = text.find(chunk_text)
        if idx >= 0:
            return self.build_context(
                document=document,
                target_start=idx,
                target_end=idx + len(chunk_text),
                model_client=model_client
            )
        
        # 如果精确匹配失败，使用模糊匹配
        logger.warning("精确匹配失败，使用模糊匹配")
        return self._fuzzy_build_context(document, chunk_text, model_client)
    
    def _get_or_generate_summary(
        self,
        document: Document,
        model_client: Optional[Any]
    ) -> str:
        """获取或生成文档摘要"""
        # 检查缓存
        if document.doc_id in self._summary_cache:
            return self._summary_cache[document.doc_id]
        
        # 检查元数据
        if document.metadata and "summary" in document.metadata:
            return document.metadata["summary"]
        
        # 使用模型生成摘要
        if model_client is not None:
            summary = self._generate_summary(document.text, model_client)
            self._summary_cache[document.doc_id] = summary
            return summary
        
        # 回退：提取首段作为摘要
        first_para = document.text.split('\n\n')[0]
        summary = first_para[:self.summary_length]
        if len(first_para) > self.summary_length:
            summary += "..."
        
        self._summary_cache[document.doc_id] = summary
        return summary
    
    def _generate_summary(self, text: str, model_client: Any) -> str:
        """使用模型生成摘要"""
        # 截取前部分文本作为输入
        input_text = text[:3000]
        
        prompt = f"""请为以下文档生成一个简短的摘要（约{self.summary_length}字），概括文档的主要内容和结构：

文档内容：
{input_text}

摘要："""
        
        try:
            response = model_client.generate(prompt, max_tokens=self.summary_length)
            summary = response.strip()
            logger.debug(f"生成摘要: {summary[:50]}...")
            return summary
        except Exception as e:
            logger.error(f"生成摘要失败: {e}")
            return text[:self.summary_length]
    
    def _parse_sections(self, text: str) -> List[Dict[str, Any]]:
        """解析章节结构"""
        sections = []
        lines = text.split('\n')
        current_pos = 0
        
        for i, line in enumerate(lines):
            # 匹配Markdown标题
            match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if match:
                level = len(match.group(1))
                title = match.group(2).strip()
                sections.append({
                    "level": level,
                    "title": title,
                    "start_line": i,
                    "start_pos": current_pos
                })
            current_pos += len(line) + 1
        
        return sections
    
    def _get_section_path(
        self,
        sections: List[Dict[str, Any]],
        target_start: int,
        target_end: int
    ) -> List[str]:
        """获取目标位置所在的章节路径"""
        path = []
        current_levels = {}  # level -> title
        
        for section in sections:
            # 更新当前层级
            current_levels[section["level"]] = section["title"]
            
            # 清除更深层级
            levels_to_remove = [
                l for l in current_levels 
                if l > section["level"]
            ]
            for l in levels_to_remove:
                del current_levels[l]
            
            # 检查目标是否在当前章节之后
            if section["start_pos"] <= target_start:
                path = [
                    current_levels[l] 
                    for l in sorted(current_levels.keys())
                ]
        
        return path
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """按段落分割文本"""
        # 使用双换行符分割
        paragraphs = re.split(r'\n\s*\n', text)
        # 过滤空段落
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        return paragraphs
    
    def _find_paragraph_index(
        self,
        paragraphs: List[str],
        target_pos: int
    ) -> int:
        """找到目标位置所在的段落索引"""
        current_pos = 0
        
        for i, para in enumerate(paragraphs):
            para_start = self._find_in_text(para, current_pos)
            if para_start >= 0:
                para_end = para_start + len(para)
                if para_start <= target_pos < para_end:
                    return i
                current_pos = para_end
        
        return 0
    
    def _find_in_text(self, para: str, start_pos: int) -> int:
        """在原始文本中查找段落位置"""
        # 简化实现
        return start_pos
    
    def _assemble_context(
        self,
        summary: str,
        section_path: List[str],
        preceding_context: str,
        target_text: str,
        following_context: str
    ) -> str:
        """组装完整上下文"""
        parts = []
        
        # 1. 文档摘要
        parts.append(f"【文档摘要】\n{summary}\n")
        
        # 2. 章节路径
        if section_path:
            path_str = " > ".join(section_path)
            parts.append(f"【章节位置】\n{path_str}\n")
        
        # 3. 前文上下文
        if preceding_context:
            parts.append(f"【前文内容】\n{preceding_context}\n")
        
        # 4. 目标文本
        parts.append(f"【目标文本】\n{target_text}\n")
        
        # 5. 后文上下文
        if following_context:
            parts.append(f"【后文内容】\n{following_context}\n")
        
        return '\n'.join(parts)
    
    def _truncate_context(self, context: str) -> str:
        """截断上下文以控制长度"""
        if len(context) <= self.max_context_length:
            return context
        
        # 按比例截断各部分
        logger.warning(f"上下文过长({len(context)}), 进行截断")
        
        # 保留目标文本部分，截断前后文
        lines = context.split('\n')
        result_lines = []
        target_started = False
        
        for line in lines:
            if '【目标文本】' in line:
                target_started = True
            
            result_lines.append(line)
            
            if len('\n'.join(result_lines)) > self.max_context_length:
                break
        
        return '\n'.join(result_lines)[:self.max_context_length]
    
    def _fuzzy_build_context(
        self,
        document: Document,
        chunk_text: str,
        model_client: Optional[Any]
    ) -> ContextWindow:
        """模糊匹配构建上下文"""
        # 使用文本开头定位
        text = document.text
        chunk_start = chunk_text[:50] if len(chunk_text) > 50 else chunk_text
        
        for i in range(len(text) - len(chunk_start)):
            if text[i:i+len(chunk_start)] == chunk_start:
                return self.build_context(
                    document=document,
                    target_start=i,
                    target_end=i + len(chunk_text),
                    model_client=model_client
                )
        
        # 完全无法定位，返回基础上下文
        summary = self._get_or_generate_summary(document, model_client)
        return ContextWindow(
            summary=summary,
            section_path=[],
            target_text=chunk_text,
            preceding_context="",
            following_context="",
            full_context=f"【文档摘要】\n{summary}\n\n【目标文本】\n{chunk_text}"
        )
    
    def clear_cache(self):
        """清空摘要缓存"""
        self._summary_cache.clear()
        logger.info("摘要缓存已清空")
