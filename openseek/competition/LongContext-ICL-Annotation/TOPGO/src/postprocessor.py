"""
结构化输出后处理模块
"""
import json
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from loguru import logger


@dataclass
class AnnotationResult:
    """标注结果"""
    doc_id: str
    entities: List[Dict[str, Any]]
    relations: List[Dict[str, Any]]
    raw_response: Optional[str] = None
    is_valid: bool = True
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "entities": self.entities,
            "relations": self.relations,
            "is_valid": self.is_valid,
            "errors": self.errors
        }
    
    def to_jsonl(self) -> str:
        """转换为JSONL格式"""
        return json.dumps(self.to_dict(), ensure_ascii=False)


class OutputPostprocessor:
    """输出后处理器"""
    
    def __init__(
        self,
        entity_types: Optional[List[str]] = None,
        merge_duplicates: bool = True,
        normalize_text: bool = True
    ):
        """
        初始化后处理器
        
        Args:
            entity_types: 允许的实体类型列表
            merge_duplicates: 是否合并重复项
            normalize_text: 是否规范化文本
        """
        self.entity_types = entity_types or [
            "方法", "技术", "数据集", "指标", "机构", "人员"
        ]
        self.merge_duplicates = merge_duplicates
        self.normalize_text = normalize_text
        
        logger.info("输出后处理器初始化完成")
    
    def process(
        self,
        raw_output: str,
        doc_id: str,
        original_text: Optional[str] = None
    ) -> AnnotationResult:
        """
        处理原始输出
        
        Args:
            raw_output: 原始输出字符串
            doc_id: 文档ID
            original_text: 原始文本（用于位置验证）
            
        Returns:
            标注结果
        """
        errors = []
        
        # 1. 尝试解析JSON
        parsed_data, parse_errors = self._parse_json(raw_output)
        errors.extend(parse_errors)
        
        if parsed_data is None:
            return AnnotationResult(
                doc_id=doc_id,
                entities=[],
                relations=[],
                raw_response=raw_output,
                is_valid=False,
                errors=errors
            )
        
        # 2. 提取实体和关系
        entities = parsed_data.get("entities", [])
        relations = parsed_data.get("relations", [])
        
        # 3. 清洗实体
        entities, entity_errors = self._clean_entities(entities, original_text)
        errors.extend(entity_errors)
        
        # 4. 清洗关系
        relations, relation_errors = self._clean_relations(relations, entities)
        errors.extend(relation_errors)
        
        # 5. 合并重复项
        if self.merge_duplicates:
            entities = self._merge_duplicate_entities(entities)
            relations = self._merge_duplicate_relations(relations)
        
        # 6. 验证结果
        is_valid = len([e for e in errors if "错误" in e or "缺少" in e]) == 0
        
        return AnnotationResult(
            doc_id=doc_id,
            entities=entities,
            relations=relations,
            raw_response=raw_output,
            is_valid=is_valid,
            errors=errors
        )
    
    def _parse_json(self, text: str) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        """
        解析JSON
        
        Args:
            text: 包含JSON的文本
            
        Returns:
            (解析结果, 错误列表)
        """
        errors = []
        
        # 尝试直接解析
        try:
            return json.loads(text), []
        except json.JSONDecodeError:
            pass
        
        # 尝试提取JSON块
        json_str = self._extract_json_block(text)
        
        if json_str:
            try:
                return json.loads(json_str), []
            except json.JSONDecodeError as e:
                errors.append(f"JSON解析错误: {str(e)}")
        
        # 尝试修复常见问题
        fixed_json = self._fix_json(text)
        if fixed_json:
            try:
                return json.loads(fixed_json), ["JSON已自动修复"]
            except:
                pass
        
        errors.append("无法解析JSON输出")
        return None, errors
    
    def _extract_json_block(self, text: str) -> Optional[str]:
        """提取JSON块"""
        # 查找第一个 { 和最后一个 }
        start = text.find('{')
        if start < 0:
            return None
        
        depth = 0
        end = -1
        
        for i, char in enumerate(text[start:]):
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    end = start + i + 1
                    break
        
        if end > start:
            return text[start:end]
        
        return None
    
    def _fix_json(self, text: str) -> Optional[str]:
        """尝试修复JSON"""
        json_str = self._extract_json_block(text)
        if not json_str:
            return None
        
        # 修复常见问题
        # 1. 修复单引号
        json_str = json_str.replace("'", '"')
        
        # 2. 修复缺少引号的键
        json_str = re.sub(r'(\w+)(?=\s*:)', r'"\1"', json_str)
        
        # 3. 修复尾随逗号
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        return json_str
    
    def _clean_entities(
        self,
        entities: List[Dict[str, Any]],
        original_text: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        清洗实体列表
        
        Args:
            entities: 原始实体列表
            original_text: 原始文本
            
        Returns:
            (清洗后的实体列表, 错误列表)
        """
        errors = []
        cleaned = []
        
        for i, entity in enumerate(entities):
            if not isinstance(entity, dict):
                errors.append(f"实体 {i}: 格式错误，应为字典")
                continue
            
            # 检查必需字段
            if "text" not in entity:
                errors.append(f"实体 {i}: 缺少 'text' 字段")
                continue
            
            entity_text = entity.get("text", "")
            entity_type = entity.get("type", "未知")
            
            # 规范化文本
            if self.normalize_text:
                entity_text = entity_text.strip()
            
            # 验证实体类型
            if entity_type not in self.entity_types:
                entity_type = self._guess_entity_type(entity_text, entity_type)
            
            # 创建清洗后的实体
            cleaned_entity = {
                "text": entity_text,
                "type": entity_type
            }
            
            # 添加位置信息（如果有）
            if "start" in entity and "end" in entity:
                start = entity["start"]
                end = entity["end"]
                
                if original_text:
                    # 验证位置
                    if 0 <= start < end <= len(original_text):
                        actual_text = original_text[start:end]
                        if actual_text != entity_text:
                            # 尝试修正位置
                            corrected_start = original_text.find(entity_text)
                            if corrected_start >= 0:
                                cleaned_entity["start"] = corrected_start
                                cleaned_entity["end"] = corrected_start + len(entity_text)
                            else:
                                errors.append(f"实体 {i}: 文本 '{entity_text}' 在原文中未找到")
                        else:
                            cleaned_entity["start"] = start
                            cleaned_entity["end"] = end
                else:
                    cleaned_entity["start"] = start
                    cleaned_entity["end"] = end
            
            cleaned.append(cleaned_entity)
        
        return cleaned, errors
    
    def _guess_entity_type(self, text: str, original_type: str) -> str:
        """猜测实体类型"""
        # 基于文本特征猜测类型
        if any(kw in text for kw in ["方法", "算法", "模型"]):
            return "方法"
        elif any(kw in text for kw in ["数据集", "语料", "数据库"]):
            return "数据集"
        elif any(kw in text for kw in ["大学", "学院", "研究院", "公司"]):
            return "机构"
        elif any(kw in text for kw in ["准确率", "F1", "BLEU", "ROUGE"]):
            return "指标"
        elif original_type in self.entity_types:
            return original_type
        else:
            return "方法"  # 默认类型
    
    def _clean_relations(
        self,
        relations: List[Dict[str, Any]],
        entities: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        清洗关系列表
        
        Args:
            relations: 原始关系列表
            entities: 实体列表
            
        Returns:
            (清洗后的关系列表, 错误列表)
        """
        errors = []
        cleaned = []
        
        entity_texts = {e.get("text") for e in entities}
        
        for i, relation in enumerate(relations):
            if not isinstance(relation, dict):
                errors.append(f"关系 {i}: 格式错误，应为字典")
                continue
            
            # 检查必需字段
            head = relation.get("head")
            tail = relation.get("tail")
            rel_type = relation.get("relation")
            
            if not head or not tail:
                errors.append(f"关系 {i}: 缺少头实体或尾实体")
                continue
            
            # 检查实体是否存在
            if head not in entity_texts:
                errors.append(f"关系 {i}: 头实体 '{head}' 不在实体列表中")
                continue
            
            if tail not in entity_texts:
                errors.append(f"关系 {i}: 尾实体 '{tail}' 不在实体列表中")
                continue
            
            cleaned.append({
                "head": head,
                "tail": tail,
                "relation": rel_type or "相关"
            })
        
        return cleaned, errors
    
    def _merge_duplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """合并重复实体"""
        seen = {}
        merged = []
        
        for entity in entities:
            key = (entity.get("text"), entity.get("type"))
            
            if key not in seen:
                seen[key] = entity
                merged.append(entity)
        
        return merged
    
    def _merge_duplicate_relations(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """合并重复关系"""
        seen = set()
        merged = []
        
        for relation in relations:
            key = (
                relation.get("head"),
                relation.get("tail"),
                relation.get("relation")
            )
            
            if key not in seen:
                seen.add(key)
                merged.append(relation)
        
        return merged
    
    def batch_process(
        self,
        results: List[Tuple[str, str, Optional[str]]]
    ) -> List[AnnotationResult]:
        """
        批量处理
        
        Args:
            results: [(doc_id, raw_output, original_text), ...]
            
        Returns:
            标注结果列表
        """
        processed = []
        
        for doc_id, raw_output, original_text in results:
            result = self.process(raw_output, doc_id, original_text)
            processed.append(result)
        
        logger.info(f"批量处理完成: {len(processed)} 条记录")
        return processed
    
    def save_results(
        self,
        results: List[AnnotationResult],
        output_path: str,
        format: str = "jsonl"
    ):
        """
        保存结果
        
        Args:
            results: 结果列表
            output_path: 输出路径
            format: 输出格式 (jsonl/json)
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "jsonl":
            with open(path, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(result.to_jsonl() + '\n')
        
        elif format == "json":
            data = [r.to_dict() for r in results]
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"结果已保存: {output_path}, 共 {len(results)} 条记录")
    
    def get_statistics(self, results: List[AnnotationResult]) -> Dict[str, Any]:
        """
        获取统计信息
        
        Args:
            results: 结果列表
            
        Returns:
            统计信息
        """
        total_entities = 0
        total_relations = 0
        valid_count = 0
        entity_type_counts = {}
        
        for result in results:
            total_entities += len(result.entities)
            total_relations += len(result.relations)
            
            if result.is_valid:
                valid_count += 1
            
            for entity in result.entities:
                entity_type = entity.get("type", "未知")
                entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1
        
        return {
            "total_documents": len(results),
            "valid_documents": valid_count,
            "total_entities": total_entities,
            "total_relations": total_relations,
            "avg_entities_per_doc": total_entities / len(results) if results else 0,
            "entity_type_distribution": entity_type_counts
        }
