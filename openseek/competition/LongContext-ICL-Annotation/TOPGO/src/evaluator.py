"""
评测与结果生成模块
"""
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from loguru import logger


@dataclass
class EvaluationMetrics:
    """评测指标"""
    precision: float
    recall: float
    f1: float
    exact_match: float
    support: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "exact_match": round(self.exact_match, 4),
            "support": self.support
        }


@dataclass
class EvaluationResult:
    """评测结果"""
    doc_id: str
    predicted_entities: List[Dict[str, Any]]
    true_entities: List[Dict[str, Any]]
    predicted_relations: List[Dict[str, Any]]
    true_relations: List[Dict[str, Any]]
    entity_metrics: EvaluationMetrics
    relation_metrics: EvaluationMetrics
    match_details: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "entity_metrics": self.entity_metrics.to_dict(),
            "relation_metrics": self.relation_metrics.to_dict(),
            "match_details": self.match_details
        }


class Evaluator:
    """评测器"""
    
    def __init__(
        self,
        entity_types: Optional[List[str]] = None,
        strict_match: bool = False
    ):
        """
        初始化评测器
        
        Args:
            entity_types: 实体类型列表
            strict_match: 是否严格匹配（包括位置）
        """
        self.entity_types = entity_types or [
            "方法", "技术", "数据集", "指标", "机构", "人员"
        ]
        self.strict_match = strict_match
        
        logger.info(f"评测器初始化, strict_match={strict_match}")
    
    def evaluate(
        self,
        prediction: Dict[str, Any],
        ground_truth: Dict[str, Any],
        doc_id: str = ""
    ) -> EvaluationResult:
        """
        评估单个文档
        
        Args:
            prediction: 预测结果
            ground_truth: 真实标签
            doc_id: 文档ID
            
        Returns:
            评测结果
        """
        # 提取实体和关系
        pred_entities = prediction.get("entities", [])
        true_entities = ground_truth.get("entities", [])
        pred_relations = prediction.get("relations", [])
        true_relations = ground_truth.get("relations", [])
        
        # 计算实体指标
        entity_metrics, entity_matches = self._evaluate_entities(
            pred_entities, true_entities
        )
        
        # 计算关系指标
        relation_metrics, relation_matches = self._evaluate_relations(
            pred_relations, true_relations, pred_entities
        )
        
        return EvaluationResult(
            doc_id=doc_id,
            predicted_entities=pred_entities,
            true_entities=true_entities,
            predicted_relations=pred_relations,
            true_relations=true_relations,
            entity_metrics=entity_metrics,
            relation_metrics=relation_metrics,
            match_details={
                "entity_matches": entity_matches,
                "relation_matches": relation_matches
            }
        )
    
    def _evaluate_entities(
        self,
        predicted: List[Dict[str, Any]],
        true: List[Dict[str, Any]]
    ) -> Tuple[EvaluationMetrics, Dict[str, Any]]:
        """
        评估实体
        
        Args:
            predicted: 预测实体列表
            true: 真实实体列表
            
        Returns:
            (指标, 匹配详情)
        """
        # 转换为集合便于比较
        pred_set = self._entities_to_set(predicted)
        true_set = self._entities_to_set(true)
        
        # 计算TP, FP, FN
        tp = len(pred_set & true_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        
        # 计算指标
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        exact_match = 1.0 if tp == len(true_set) and fp == 0 else 0.0
        
        metrics = EvaluationMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            exact_match=exact_match,
            support=len(true_set)
        )
        
        # 匹配详情
        matches = {
            "true_positives": list(pred_set & true_set),
            "false_positives": list(pred_set - true_set),
            "false_negatives": list(true_set - pred_set),
            "tp_count": tp,
            "fp_count": fp,
            "fn_count": fn
        }
        
        return metrics, matches
    
    def _entities_to_set(self, entities: List[Dict[str, Any]]) -> set:
        """将实体列表转换为集合"""
        entity_set = set()
        
        for entity in entities:
            if self.strict_match:
                # 严格匹配：文本+类型+位置
                key = (
                    entity.get("text", ""),
                    entity.get("type", ""),
                    entity.get("start", -1),
                    entity.get("end", -1)
                )
            else:
                # 宽松匹配：文本+类型
                key = (
                    entity.get("text", "").strip(),
                    entity.get("type", "")
                )
            
            entity_set.add(key)
        
        return entity_set
    
    def _evaluate_relations(
        self,
        predicted: List[Dict[str, Any]],
        true: List[Dict[str, Any]],
        pred_entities: List[Dict[str, Any]]
    ) -> Tuple[EvaluationMetrics, Dict[str, Any]]:
        """
        评估关系
        
        Args:
            predicted: 预测关系列表
            true: 真实关系列表
            pred_entities: 预测实体列表
            
        Returns:
            (指标, 匹配详情)
        """
        # 转换为集合
        pred_set = self._relations_to_set(predicted)
        true_set = self._relations_to_set(true)
        
        # 计算TP, FP, FN
        tp = len(pred_set & true_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        
        # 计算指标
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        exact_match = 1.0 if tp == len(true_set) and fp == 0 else 0.0
        
        metrics = EvaluationMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            exact_match=exact_match,
            support=len(true_set)
        )
        
        matches = {
            "true_positives": list(pred_set & true_set),
            "false_positives": list(pred_set - true_set),
            "false_negatives": list(true_set - pred_set),
            "tp_count": tp,
            "fp_count": fp,
            "fn_count": fn
        }
        
        return metrics, matches
    
    def _relations_to_set(self, relations: List[Dict[str, Any]]) -> set:
        """将关系列表转换为集合"""
        relation_set = set()
        
        for relation in relations:
            key = (
                relation.get("head", ""),
                relation.get("tail", ""),
                relation.get("relation", "")
            )
            relation_set.add(key)
        
        return relation_set
    
    def evaluate_batch(
        self,
        predictions: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]],
        doc_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        批量评测
        
        Args:
            predictions: 预测结果列表
            ground_truths: 真实标签列表
            doc_ids: 文档ID列表
            
        Returns:
            评测报告
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("预测结果和真实标签数量不一致")
        
        if doc_ids is None:
            doc_ids = [f"doc_{i}" for i in range(len(predictions))]
        
        results = []
        total_entity_metrics = {"tp": 0, "fp": 0, "fn": 0}
        total_relation_metrics = {"tp": 0, "fp": 0, "fn": 0}
        
        for pred, truth, doc_id in zip(predictions, ground_truths, doc_ids):
            result = self.evaluate(pred, truth, doc_id)
            results.append(result)
            
            # 累计指标
            total_entity_metrics["tp"] += result.entity_metrics.support * result.entity_metrics.precision * \
                (result.entity_metrics.precision + result.entity_metrics.recall) / (2 * result.entity_metrics.recall) if result.entity_metrics.recall > 0 else 0
            total_entity_metrics["fp"] += result.match_details["entity_matches"]["fp_count"]
            total_entity_metrics["fn"] += result.match_details["entity_matches"]["fn_count"]
            
            total_relation_metrics["tp"] += result.match_details["relation_matches"]["tp_count"]
            total_relation_metrics["fp"] += result.match_details["relation_matches"]["fp_count"]
            total_relation_metrics["fn"] += result.match_details["relation_matches"]["fn_count"]
        
        # 计算整体指标
        overall_entity_metrics = self._compute_overall_metrics(total_entity_metrics)
        overall_relation_metrics = self._compute_overall_metrics(total_relation_metrics)
        
        return {
            "overall": {
                "entity_metrics": overall_entity_metrics.to_dict(),
                "relation_metrics": overall_relation_metrics.to_dict()
            },
            "per_document": [r.to_dict() for r in results],
            "summary": {
                "total_documents": len(results),
                "avg_entity_f1": sum(r.entity_metrics.f1 for r in results) / len(results),
                "avg_relation_f1": sum(r.relation_metrics.f1 for r in results) / len(results)
            }
        }
    
    def _compute_overall_metrics(self, counts: Dict[str, int]) -> EvaluationMetrics:
        """计算整体指标"""
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return EvaluationMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            exact_match=0.0,
            support=tp + fn
        )
    
    def generate_report(
        self,
        results: Dict[str, Any],
        output_path: str
    ):
        """
        生成评测报告
        
        Args:
            results: 评测结果
            output_path: 输出路径
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"评测报告已保存: {output_path}")
    
    def print_summary(self, results: Dict[str, Any]):
        """打印评测摘要"""
        print("\n" + "=" * 60)
        print("评测结果摘要")
        print("=" * 60)
        
        overall = results.get("overall", {})
        summary = results.get("summary", {})
        
        entity_metrics = overall.get("entity_metrics", {})
        relation_metrics = overall.get("relation_metrics", {})
        
        print(f"\n总文档数: {summary.get('total_documents', 0)}")
        
        print("\n--- 实体识别 ---")
        print(f"精确率 (Precision): {entity_metrics.get('precision', 0):.4f}")
        print(f"召回率 (Recall):    {entity_metrics.get('recall', 0):.4f}")
        print(f"F1分数:            {entity_metrics.get('f1', 0):.4f}")
        
        print("\n--- 关系抽取 ---")
        print(f"精确率 (Precision): {relation_metrics.get('precision', 0):.4f}")
        print(f"召回率 (Recall):    {relation_metrics.get('recall', 0):.4f}")
        print(f"F1分数:            {relation_metrics.get('f1', 0):.4f}")
        
        print(f"\n平均实体F1: {summary.get('avg_entity_f1', 0):.4f}")
        print(f"平均关系F1: {summary.get('avg_relation_f1', 0):.4f}")
        
        print("=" * 60 + "\n")


class ResultGenerator:
    """结果生成器"""
    
    def __init__(self, output_dir: str):
        """
        初始化结果生成器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"结果生成器初始化, 输出目录: {output_dir}")
    
    def generate_predictions_file(
        self,
        predictions: List[Dict[str, Any]],
        filename: str = "predictions.json"
    ):
        """
        生成预测结果文件
        
        Args:
            predictions: 预测结果列表
            filename: 文件名
        """
        path = self.output_dir / filename
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        
        logger.info(f"预测结果已保存: {path}")
    
    def generate_submission_file(
        self,
        predictions: List[Dict[str, Any]],
        format: str = "jsonl",
        filename: Optional[str] = None
    ):
        """
        生成提交文件
        
        Args:
            predictions: 预测结果列表
            format: 格式 (json/jsonl)
            filename: 文件名
        """
        if filename is None:
            filename = f"submission.{format}"
        
        path = self.output_dir / filename
        
        if format == "jsonl":
            with open(path, 'w', encoding='utf-8') as f:
                for pred in predictions:
                    f.write(json.dumps(pred, ensure_ascii=False) + '\n')
        else:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(predictions, f, ensure_ascii=False, indent=2)
        
        logger.info(f"提交文件已生成: {path}")
    
    def generate_error_analysis(
        self,
        evaluation_results: Dict[str, Any],
        filename: str = "error_analysis.json"
    ):
        """
        生成错误分析报告
        
        Args:
            evaluation_results: 评测结果
            filename: 文件名
        """
        errors = {
            "entity_errors": [],
            "relation_errors": []
        }
        
        for doc_result in evaluation_results.get("per_document", []):
            doc_id = doc_result.get("doc_id", "")
            
            # 实体错误
            entity_matches = doc_result.get("match_details", {}).get("entity_matches", {})
            
            for fp in entity_matches.get("false_positives", []):
                errors["entity_errors"].append({
                    "doc_id": doc_id,
                    "type": "false_positive",
                    "entity": fp
                })
            
            for fn in entity_matches.get("false_negatives", []):
                errors["entity_errors"].append({
                    "doc_id": doc_id,
                    "type": "false_negative",
                    "entity": fn
                })
        
        path = self.output_dir / filename
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)
        
        logger.info(f"错误分析报告已生成: {path}")
