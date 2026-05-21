"""
自洽性验证与修正模块
"""
import json
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from loguru import logger

from .models.qwen_client import QwenClient
from .prompt_engineer import PromptManager
from .context_builder import ContextWindow


@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    original_annotation: Dict[str, Any]
    corrected_annotation: Optional[Dict[str, Any]]
    validation_message: str
    round_number: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "original_annotation": self.original_annotation,
            "corrected_annotation": self.corrected_annotation,
            "validation_message": self.validation_message,
            "round_number": self.round_number
        }


class SelfConsistencyValidator:
    """自洽性验证器"""
    
    def __init__(
        self,
        model_client: QwenClient,
        prompt_manager: PromptManager,
        max_validation_rounds: int = 2
    ):
        """
        初始化验证器
        
        Args:
            model_client: 模型客户端
            prompt_manager: 提示词管理器
            max_validation_rounds: 最大验证轮数
        """
        self.model_client = model_client
        self.prompt_manager = prompt_manager
        self.max_validation_rounds = max_validation_rounds
        
        logger.info(f"自洽性验证器初始化, 最大验证轮数: {max_validation_rounds}")
    
    def validate(
        self,
        context: ContextWindow,
        initial_annotation: Dict[str, Any]
    ) -> ValidationResult:
        """
        验证标注结果
        
        Args:
            context: 上下文窗口
            initial_annotation: 初始标注结果
            
        Returns:
            验证结果
        """
        current_annotation = initial_annotation
        validation_message = ""
        
        for round_num in range(1, self.max_validation_rounds + 1):
            logger.debug(f"验证轮次 {round_num}")
            
            # 构建验证提示词
            annotation_str = json.dumps(current_annotation, ensure_ascii=False, indent=2)
            validation_prompt = self.prompt_manager.template.build_validation_prompt(
                context_text=context.full_context,
                initial_annotation=annotation_str
            )
            
            # 调用模型验证
            try:
                response = self.model_client.generate(
                    prompt=validation_prompt,
                    temperature=0.0  # 验证使用较低温度
                )
                
                # 解析验证结果
                is_valid, corrected, message = self._parse_validation_response(response)
                
                if is_valid:
                    logger.info(f"验证通过 (轮次 {round_num})")
                    return ValidationResult(
                        is_valid=True,
                        original_annotation=initial_annotation,
                        corrected_annotation=None,
                        validation_message=f"验证通过 (轮次 {round_num})",
                        round_number=round_num
                    )
                
                if corrected:
                    current_annotation = corrected
                    validation_message = message
                
            except Exception as e:
                logger.error(f"验证过程出错: {e}")
                return ValidationResult(
                    is_valid=False,
                    original_annotation=initial_annotation,
                    corrected_annotation=current_annotation,
                    validation_message=f"验证过程出错: {str(e)}",
                    round_number=round_num
                )
        
        # 达到最大轮次仍未通过
        logger.warning(f"达到最大验证轮次 {self.max_validation_rounds}")
        return ValidationResult(
            is_valid=False,
            original_annotation=initial_annotation,
            corrected_annotation=current_annotation,
            validation_message=f"达到最大验证轮次: {validation_message}",
            round_number=self.max_validation_rounds
        )
    
    def _parse_validation_response(
        self,
        response: str
    ) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """
        解析验证响应
        
        Args:
            response: 模型响应
            
        Returns:
            (是否通过, 修正后的标注, 消息)
        """
        response = response.strip()
        
        # 检查是否通过验证
        if "验证通过" in response or "正确无误" in response:
            return True, None, "标注结果验证通过"
        
        # 尝试解析JSON
        try:
            # 提取JSON部分
            json_str = self._extract_json(response)
            corrected = json.loads(json_str)
            
            return False, corrected, "标注结果已修正"
            
        except json.JSONDecodeError:
            logger.warning(f"无法解析验证响应: {response[:200]}")
            return False, None, f"无法解析响应"
    
    def _extract_json(self, text: str) -> str:
        """从文本中提取JSON"""
        start_idx = text.find('{')
        if start_idx < 0:
            return text
        
        depth = 0
        for i, char in enumerate(text[start_idx:]):
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    return text[start_idx:start_idx + i + 1]
        
        return text[start_idx:]
    
    def multi_round_validate(
        self,
        context: ContextWindow,
        initial_annotation: Dict[str, Any],
        num_rounds: int = 3
    ) -> Dict[str, Any]:
        """
        多轮验证（自洽性验证）
        
        Args:
            context: 上下文窗口
            initial_annotation: 初始标注
            num_rounds: 验证轮数
            
        Returns:
            最终标注结果
        """
        annotations = [initial_annotation]
        
        for _ in range(num_rounds - 1):
            # 重新生成标注
            prompt = self.prompt_manager.build_prompt(context)
            response = self.model_client.generate(prompt=prompt, temperature=0.3)
            
            try:
                annotation = json.loads(self._extract_json(response))
                annotations.append(annotation)
            except:
                pass
        
        # 合并结果
        return self._merge_annotations(annotations)
    
    def _merge_annotations(
        self,
        annotations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        合并多个标注结果（投票机制）
        
        Args:
            annotations: 标注结果列表
            
        Returns:
            合并后的标注
        """
        if not annotations:
            return {"entities": [], "relations": []}
        
        if len(annotations) == 1:
            return annotations[0]
        
        # 实体投票
        entity_counts: Dict[str, int] = {}
        entity_map: Dict[str, Dict[str, Any]] = {}
        
        for annotation in annotations:
            entities = annotation.get("entities", [])
            for entity in entities:
                key = f"{entity.get('text', '')}_{entity.get('type', '')}"
                entity_counts[key] = entity_counts.get(key, 0) + 1
                if key not in entity_map:
                    entity_map[key] = entity
        
        # 选择出现次数超过一半的实体
        merged_entities = []
        threshold = len(annotations) / 2
        
        for key, count in entity_counts.items():
            if count >= threshold:
                merged_entities.append(entity_map[key])
        
        # 关系投票
        relation_counts: Dict[str, int] = {}
        relation_map: Dict[str, Dict[str, Any]] = {}
        
        for annotation in annotations:
            relations = annotation.get("relations", [])
            for relation in relations:
                key = f"{relation.get('head', '')}_{relation.get('relation', '')}_{relation.get('tail', '')}"
                relation_counts[key] = relation_counts.get(key, 0) + 1
                if key not in relation_map:
                    relation_map[key] = relation
        
        merged_relations = []
        for key, count in relation_counts.items():
            if count >= threshold:
                merged_relations.append(relation_map[key])
        
        return {
            "entities": merged_entities,
            "relations": merged_relations
        }


class AnnotationValidator:
    """标注验证器（规则检查）"""
    
    def __init__(self, entity_types: List[str] = None):
        """
        初始化标注验证器
        
        Args:
            entity_types: 允许的实体类型列表
        """
        self.entity_types = entity_types or [
            "方法", "技术", "数据集", "指标", "机构", "人员"
        ]
    
    def validate_annotation(
        self,
        annotation: Dict[str, Any],
        text: str
    ) -> Tuple[bool, List[str]]:
        """
        验证标注格式和内容
        
        Args:
            annotation: 标注结果
            text: 原始文本
            
        Returns:
            (是否有效, 错误列表)
        """
        errors = []
        
        # 检查基本结构
        if "entities" not in annotation:
            errors.append("缺少 'entities' 字段")
        if "relations" not in annotation:
            errors.append("缺少 'relations' 字段")
        
        if errors:
            return False, errors
        
        # 检查实体
        entities = annotation.get("entities", [])
        for i, entity in enumerate(entities):
            entity_errors = self._validate_entity(entity, text, i)
            errors.extend(entity_errors)
        
        # 检查关系
        relations = annotation.get("relations", [])
        entity_texts = {e.get("text") for e in entities}
        
        for i, relation in enumerate(relations):
            relation_errors = self._validate_relation(relation, entity_texts, i)
            errors.extend(relation_errors)
        
        return len(errors) == 0, errors
    
    def _validate_entity(
        self,
        entity: Dict[str, Any],
        text: str,
        index: int
    ) -> List[str]:
        """验证单个实体"""
        errors = []
        
        # 检查必需字段
        if "text" not in entity:
            errors.append(f"实体 {index}: 缺少 'text' 字段")
        if "type" not in entity:
            errors.append(f"实体 {index}: 缺少 'type' 字段")
        
        # 检查实体类型
        entity_type = entity.get("type")
        if entity_type and entity_type not in self.entity_types:
            errors.append(f"实体 {index}: 未知实体类型 '{entity_type}'")
        
        # 检查位置
        start = entity.get("start")
        end = entity.get("end")
        
        if start is not None and end is not None:
            if start < 0 or end > len(text):
                errors.append(f"实体 {index}: 位置超出文本范围")
            elif start >= end:
                errors.append(f"实体 {index}: 起始位置大于结束位置")
            elif text[start:end] != entity.get("text", ""):
                errors.append(f"实体 {index}: 文本与位置不匹配")
        
        return errors
    
    def _validate_relation(
        self,
        relation: Dict[str, Any],
        entity_texts: set,
        index: int
    ) -> List[str]:
        """验证单个关系"""
        errors = []
        
        # 检查必需字段
        if "head" not in relation:
            errors.append(f"关系 {index}: 缺少 'head' 字段")
        if "tail" not in relation:
            errors.append(f"关系 {index}: 缺少 'tail' 字段")
        if "relation" not in relation:
            errors.append(f"关系 {index}: 缺少 'relation' 字段")
        
        # 检查头尾实体是否存在
        head = relation.get("head")
        tail = relation.get("tail")
        
        if head and head not in entity_texts:
            errors.append(f"关系 {index}: 头实体 '{head}' 不在实体列表中")
        if tail and tail not in entity_texts:
            errors.append(f"关系 {index}: 尾实体 '{tail}' 不在实体列表中")
        
        return errors
