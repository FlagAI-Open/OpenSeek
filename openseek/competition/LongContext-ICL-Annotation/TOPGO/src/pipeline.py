"""
主Pipeline - 长文档智能标注系统
"""
import os
import json
import time
from typing import Dict, Any, List, Optional
from pathlib import Path
from loguru import logger

from .utils.config import load_config, validate_config
from .utils.logger import setup_logger
from .data.loader import DataLoader, Document
from .data.preprocessor import DataPreprocessor
from .context_builder import ContextBuilder
from .models.embedding import EmbeddingModel
from .models.qwen_client import QwenClient, MockQwenClient
from .retriever import ExampleRetriever
from .prompt_engineer import PromptManager
from .validator import SelfConsistencyValidator, AnnotationValidator
from .postprocessor import OutputPostprocessor, AnnotationResult
from .evaluator import Evaluator, ResultGenerator


class AnnotationPipeline:
    """标注Pipeline"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化Pipeline
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        self.config = load_config(config_path)
        validate_config(self.config)
        
        # 设置日志
        log_config = self.config.get("logging", {})
        setup_logger(
            log_level=log_config.get("level", "INFO"),
            log_file=log_config.get("file"),
            log_format=log_config.get("format")
        )
        
        logger.info("=" * 60)
        logger.info("长文档智能标注系统初始化")
        logger.info("=" * 60)
        
        # 初始化各组件
        self._init_components()
        
        # 统计信息
        self.stats = {
            "total_documents": 0,
            "processed_documents": 0,
            "failed_documents": 0,
            "total_entities": 0,
            "total_relations": 0
        }
    
    def _init_components(self):
        """初始化各组件"""
        # 数据加载器
        self.data_loader = DataLoader()
        
        # 数据预处理器
        context_config = self.config.get("context", {})
        self.preprocessor = DataPreprocessor(
            chunk_size=context_config.get("chunk_size", 512),
            chunk_overlap=context_config.get("chunk_overlap", 50)
        )
        
        # 上下文构建器
        self.context_builder = ContextBuilder(
            summary_length=context_config.get("summary_length", 200),
            neighbor_paragraphs_before=context_config.get("neighbor_paragraphs_before", 1),
            neighbor_paragraphs_after=context_config.get("neighbor_paragraphs_after", 1),
            max_context_length=context_config.get("max_context_length", 8000)
        )
        
        # 嵌入模型
        retrieval_config = self.config.get("retrieval", {})
        self.embedding_model = EmbeddingModel(
            model_name=retrieval_config.get("embedding_model", "BAAI/bge-m3"),
            device=retrieval_config.get("embedding_device", "cpu")
        )
        
        # 示例检索器
        self.retriever = ExampleRetriever(
            embedding_model=self.embedding_model,
            top_k=retrieval_config.get("top_k", 3),
            similarity_threshold=retrieval_config.get("similarity_threshold", 0.5)
        )
        
        # 模型客户端
        model_config = self.config.get("model", {})
        if os.getenv("USE_MOCK_MODEL") == "true" or not model_config.get("api_base"):
            self.model_client = MockQwenClient()
        else:
            self.model_client = QwenClient(
                api_base=model_config.get("api_base"),
                api_key=model_config.get("api_key", ""),
                model_name=model_config.get("name", "Qwen3-4B"),
                temperature=model_config.get("temperature", 0.1),
                max_tokens=model_config.get("max_tokens", 2048),
                timeout=model_config.get("timeout", 300),
                max_retries=model_config.get("max_retries", 3)
            )
        
        # 提示词管理器
        self.prompt_manager = PromptManager()
        
        # 验证器
        validation_config = self.config.get("validation", {})
        self.validator = SelfConsistencyValidator(
            model_client=self.model_client,
            prompt_manager=self.prompt_manager,
            max_validation_rounds=validation_config.get("max_validation_rounds", 2)
        )
        
        # 后处理器
        annotation_config = self.config.get("annotation", {})
        self.postprocessor = OutputPostprocessor(
            entity_types=annotation_config.get("entity_types"),
            merge_duplicates=True,
            normalize_text=True
        )
        
        # 评测器
        self.evaluator = Evaluator(
            entity_types=annotation_config.get("entity_types")
        )
        
        # 结果生成器
        output_config = self.config.get("output", {})
        self.result_generator = ResultGenerator(
            output_dir=output_config.get("results_dir", "output/results")
        )
        
        logger.info("所有组件初始化完成")
    
    def run(
        self,
        train_file: str,
        test_file: str,
        output_file: str = "predictions.json"
    ) -> Dict[str, Any]:
        """
        运行完整Pipeline
        
        Args:
            train_file: 训练数据文件
            test_file: 测试数据文件
            output_file: 输出文件名
            
        Returns:
            处理结果统计
        """
        logger.info("开始运行Pipeline")
        start_time = time.time()
        
        try:
            # 1. 加载数据
            logger.info("步骤1: 加载数据")
            train_docs = self.data_loader.load_documents(train_file)
            test_docs = self.data_loader.load_documents(test_file)
            self.stats["total_documents"] = len(test_docs)
            
            # 2. 构建示例库
            logger.info("步骤2: 构建示例库")
            example_library = self.preprocessor.build_example_library(train_docs)
            self.retriever.build_index(example_library)
            logger.info(f"示例库构建完成，共 {len(example_library)} 个示例")
            
            # 3. 处理测试文档
            logger.info("步骤3: 处理测试文档")
            predictions = []
            
            for doc in test_docs:
                try:
                    result = self._process_document(doc)
                    predictions.append(result.to_dict())
                    self.stats["processed_documents"] += 1
                    self.stats["total_entities"] += len(result.entities)
                    self.stats["total_relations"] += len(result.relations)
                    
                except Exception as e:
                    logger.error(f"处理文档 {doc.doc_id} 失败: {e}")
                    self.stats["failed_documents"] += 1
                    predictions.append({
                        "doc_id": doc.doc_id,
                        "entities": [],
                        "relations": [],
                        "error": str(e)
                    })
            
            # 4. 保存结果
            logger.info("步骤4: 保存结果")
            self.result_generator.generate_predictions_file(predictions, output_file)
            
            # 5. 生成统计报告
            elapsed_time = time.time() - start_time
            self.stats["elapsed_time"] = elapsed_time
            self.stats["avg_time_per_doc"] = elapsed_time / len(test_docs) if test_docs else 0
            
            logger.info("Pipeline运行完成")
            self._print_stats()
            
            return {
                "predictions": predictions,
                "stats": self.stats
            }
            
        except Exception as e:
            logger.error(f"Pipeline运行失败: {e}")
            raise
    
    def _process_document(self, document: Document) -> AnnotationResult:
        """
        处理单个文档
        
        Args:
            document: 文档对象
            
        Returns:
            标注结果
        """
        logger.debug(f"处理文档: {document.doc_id}")
        
        # 1. 文档预处理
        cleaned_text = self.preprocessor.clean_text(document.text)
        document.text = cleaned_text
        
        # 2. 文档分块
        chunks = self.preprocessor.chunk_document(document)
        logger.debug(f"文档分为 {len(chunks)} 个块")
        
        # 3. 处理每个块
        all_entities = []
        all_relations = []
        
        for chunk in chunks:
            # 构建上下文
            context = self.context_builder.build_context_for_chunk(
                document=document,
                chunk_text=chunk.text,
                model_client=self.model_client
            )
            
            # 检索相似示例
            examples = self.retriever.retrieve(chunk.text)
            
            # 构建提示词
            task_type = self.config.get("annotation", {}).get("task_type", "entity_extraction")
            prompt = self.prompt_manager.build_prompt(
                context=context,
                task_type=task_type,
                examples=examples
            )
            
            # 调用模型
            raw_response = self.model_client.generate(prompt=prompt)
            
            # 后处理
            result = self.postprocessor.process(
                raw_output=raw_response,
                doc_id=f"{document.doc_id}_{chunk.chunk_id}",
                original_text=chunk.text
            )
            
            # 验证（可选）
            if self.config.get("validation", {}).get("enable", True) and result.is_valid:
                validation_result = self.validator.validate(
                    context=context,
                    initial_annotation={"entities": result.entities, "relations": result.relations}
                )
                
                if validation_result.corrected_annotation:
                    result.entities = validation_result.corrected_annotation.get("entities", [])
                    result.relations = validation_result.corrected_annotation.get("relations", [])
            
            all_entities.extend(result.entities)
            all_relations.extend(result.relations)
        
        # 4. 合并结果
        merged_entities = self.postprocessor._merge_duplicate_entities(all_entities)
        merged_relations = self.postprocessor._merge_duplicate_relations(all_relations)
        
        return AnnotationResult(
            doc_id=document.doc_id,
            entities=merged_entities,
            relations=merged_relations,
            is_valid=True
        )
    
    def evaluate(
        self,
        predictions_file: str,
        ground_truth_file: str,
        output_report: str = "evaluation_report.json"
    ) -> Dict[str, Any]:
        """
        评测结果
        
        Args:
            predictions_file: 预测结果文件
            ground_truth_file: 真实标签文件
            output_report: 输出报告文件名
            
        Returns:
            评测结果
        """
        logger.info("开始评测")
        
        # 加载预测结果
        with open(predictions_file, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
        
        # 加载真实标签
        ground_truths = self.data_loader.load_jsonl(ground_truth_file)
        
        # 执行评测
        doc_ids = [p.get("doc_id") for p in predictions]
        results = self.evaluator.evaluate_batch(predictions, ground_truths, doc_ids)
        
        # 保存报告
        self.evaluator.generate_report(results, output_report)
        
        # 打印摘要
        self.evaluator.print_summary(results)
        
        return results
    
    def _print_stats(self):
        """打印统计信息"""
        logger.info("=" * 60)
        logger.info("处理统计")
        logger.info("=" * 60)
        logger.info(f"总文档数: {self.stats['total_documents']}")
        logger.info(f"成功处理: {self.stats['processed_documents']}")
        logger.info(f"处理失败: {self.stats['failed_documents']}")
        logger.info(f"提取实体总数: {self.stats['total_entities']}")
        logger.info(f"提取关系总数: {self.stats['total_relations']}")
        logger.info(f"总耗时: {self.stats['elapsed_time']:.2f} 秒")
        logger.info(f"平均每文档: {self.stats['avg_time_per_doc']:.2f} 秒")
        logger.info("=" * 60)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="长文档智能标注系统")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--train", type=str, help="训练数据文件")
    parser.add_argument("--test", type=str, help="测试数据文件")
    parser.add_argument("--output", type=str, default="predictions.json", help="输出文件")
    parser.add_argument("--evaluate", action="store_true", help="是否执行评测")
    parser.add_argument("--ground-truth", type=str, help="真实标签文件")
    
    args = parser.parse_args()
    
    # 创建Pipeline
    pipeline = AnnotationPipeline(args.config)
    
    # 运行
    if args.train and args.test:
        pipeline.run(
            train_file=args.train,
            test_file=args.test,
            output_file=args.output
        )
    
    # 评测
    if args.evaluate and args.ground_truth:
        pipeline.evaluate(
            predictions_file=args.output,
            ground_truth_file=args.ground_truth
        )


if __name__ == "__main__":
    main()
