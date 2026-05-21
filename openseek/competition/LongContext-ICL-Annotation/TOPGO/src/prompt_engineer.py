"""
提示词工程模块
"""
from typing import Dict, Any, List, Optional
from string import Template
from pathlib import Path
from loguru import logger

from .retriever import RetrievedExample
from .context_builder import ContextWindow


class PromptTemplate:
    """提示词模板"""
    
    # 系统提示词
    SYSTEM_PROMPT = """你是一个专业的数据标注助手，擅长从技术文档中提取结构化信息。

你的任务是：
1. 仔细阅读提供的文档内容
2. 根据标注任务要求，准确提取相关实体和关系
3. 以指定的JSON格式输出结果

标注规范：
- 实体类型包括：方法、技术、数据集、指标、机构、人员
- 确保提取的信息完整、准确
- 不要遗漏重要的实体
- 关系抽取需要基于明确的文本证据"""

    # 标注任务提示词模板
    ANNOTATION_TEMPLATE = """## 任务说明
请从以下文本中${task_description}

## 文档摘要
${summary}

## 章节位置
${section_path}

## 待标注文本
${context_text}

## 参考示例
${examples}

## 输出要求
请以JSON格式输出，格式如下：
{
  "entities": [
    {"text": "实体文本", "type": "实体类型", "start": 起始位置, "end": 结束位置}
  ],
  "relations": [
    {"head": "头实体", "relation": "关系类型", "tail": "尾实体"}
  ]
}

请输出标注结果："""

    # 验证提示词模板
    VALIDATION_TEMPLATE = """## 验证任务
请检查以下标注结果是否正确、完整。

## 原始文本
${context_text}

## 初始标注结果
${initial_annotation}

## 验证要点
1. 实体是否被遗漏？
2. 实体类型是否正确？
3. 关系是否准确？
4. 格式是否符合要求？

## 输出要求
如果标注结果正确无误，请输出：验证通过
如果需要修正，请输出修正后的完整JSON结果

请输出验证结果："""

    def __init__(self, template_dir: Optional[str] = None):
        """
        初始化提示词模板
        
        Args:
            template_dir: 模板目录路径
        """
        self.template_dir = Path(template_dir) if template_dir else None
        self._custom_templates: Dict[str, str] = {}
        
        if self.template_dir and self.template_dir.exists():
            self._load_custom_templates()
    
    def _load_custom_templates(self):
        """加载自定义模板"""
        for template_file in self.template_dir.glob("*.txt"):
            template_name = template_file.stem
            with open(template_file, 'r', encoding='utf-8') as f:
                self._custom_templates[template_name] = f.read()
            logger.info(f"加载自定义模板: {template_name}")
    
    def build_annotation_prompt(
        self,
        context: ContextWindow,
        task_description: str,
        examples: Optional[List[RetrievedExample]] = None,
        entity_types: Optional[List[str]] = None
    ) -> str:
        """
        构建标注提示词
        
        Args:
            context: 上下文窗口
            task_description: 任务描述
            examples: 示例列表
            entity_types: 实体类型列表
            
        Returns:
            完整提示词
        """
        # 格式化示例
        examples_str = ""
        if examples:
            examples_str = self._format_examples(examples)
        
        # 格式化章节路径
        section_path = " > ".join(context.section_path) if context.section_path else "文档主体"
        
        # 组装提示词
        prompt = self.ANNOTATION_TEMPLATE.substitute(
            task_description=task_description,
            summary=context.summary,
            section_path=section_path,
            context_text=context.full_context,
            examples=examples_str if examples_str else "（无参考示例）"
        )
        
        return prompt
    
    def build_validation_prompt(
        self,
        context_text: str,
        initial_annotation: str
    ) -> str:
        """
        构建验证提示词
        
        Args:
            context_text: 原始文本
            initial_annotation: 初始标注结果
            
        Returns:
            验证提示词
        """
        return self.VALIDATION_TEMPLATE.substitute(
            context_text=context_text,
            initial_annotation=initial_annotation
        )
    
    def _format_examples(
        self,
        examples: List[RetrievedExample],
        max_examples: int = 3
    ) -> str:
        """
        格式化示例
        
        Args:
            examples: 示例列表
            max_examples: 最大示例数
            
        Returns:
            格式化后的示例字符串
        """
        parts = []
        
        for i, retrieved in enumerate(examples[:max_examples]):
            example = retrieved.example
            parts.append(f"### 示例 {i + 1}")
            parts.append(f"文本片段：{example.text_chunk[:300]}...")
            
            if example.annotation:
                parts.append(f"标注结果：{example.annotation}")
            
            parts.append("")  # 空行
        
        return '\n'.join(parts)
    
    def get_system_prompt(self) -> str:
        """获取系统提示词"""
        return self.SYSTEM_PROMPT
    
    def build_few_shot_prompt(
        self,
        examples: List[RetrievedExample],
        query_text: str
    ) -> str:
        """
        构建Few-shot提示词
        
        Args:
            examples: 示例列表
            query_text: 查询文本
            
        Returns:
            Few-shot提示词
        """
        parts = [self.SYSTEM_PROMPT, ""]
        
        # 添加示例
        for i, retrieved in enumerate(examples[:3]):
            example = retrieved.example
            parts.append(f"示例 {i + 1}:")
            parts.append(f"输入: {example.text_chunk[:200]}...")
            parts.append(f"输出: {example.annotation}")
            parts.append("")
        
        # 添加当前任务
        parts.append("当前任务:")
        parts.append(f"输入: {query_text[:500]}...")
        parts.append("输出:")
        
        return '\n'.join(parts)


class PromptManager:
    """提示词管理器"""
    
    def __init__(self, template_dir: Optional[str] = None):
        """
        初始化提示词管理器
        
        Args:
            template_dir: 模板目录
        """
        self.template = PromptTemplate(template_dir)
        self._task_prompts: Dict[str, str] = {
            "entity_extraction": "提取文本中的所有实体，包括方法、技术、数据集、指标、机构和人员名称。",
            "relation_extraction": "识别文本中实体之间的关系，包括使用、提出、参与等关系。",
            "event_extraction": "从文本中提取事件信息，包括事件类型、参与者、时间、地点等要素。",
            "full_annotation": "进行完整的信息抽取，包括实体识别、关系抽取和事件抽取。"
        }
        
        logger.info("提示词管理器初始化完成")
    
    def get_task_description(self, task_type: str) -> str:
        """
        获取任务描述
        
        Args:
            task_type: 任务类型
            
        Returns:
            任务描述
        """
        return self._task_prompts.get(task_type, self._task_prompts["entity_extraction"])
    
    def build_prompt(
        self,
        context: ContextWindow,
        task_type: str = "entity_extraction",
        examples: Optional[List[RetrievedExample]] = None
    ) -> str:
        """
        构建完整提示词
        
        Args:
            context: 上下文窗口
            task_type: 任务类型
            examples: 示例列表
            
        Returns:
            完整提示词
        """
        task_description = self.get_task_description(task_type)
        return self.template.build_annotation_prompt(
            context=context,
            task_description=task_description,
            examples=examples
        )
    
    def build_messages(
        self,
        context: ContextWindow,
        task_type: str = "entity_extraction",
        examples: Optional[List[RetrievedExample]] = None
    ) -> List[Dict[str, str]]:
        """
        构建消息格式（用于Chat API）
        
        Args:
            context: 上下文窗口
            task_type: 任务类型
            examples: 示例列表
            
        Returns:
            消息列表
        """
        system_prompt = self.template.get_system_prompt()
        user_prompt = self.build_prompt(context, task_type, examples)
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def register_task_prompt(self, task_type: str, description: str):
        """
        注册任务提示词
        
        Args:
            task_type: 任务类型
            description: 任务描述
        """
        self._task_prompts[task_type] = description
        logger.info(f"注册任务提示词: {task_type}")
