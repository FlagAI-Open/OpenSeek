#!/usr/bin/env python3
"""
四智能体流水线示例程序

展示如何使用 FourAgentPipeline 完成长文档标注任务

运行方式:
    python four_agent_main.py --task_description "找出所有责任条款及其主体" --doc_id "doc_001"
"""

import argparse
import json
import os
from typing import List

# 设置国内镜像源
os.environ['HF_ENDPOINT'] = os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com')

from src.data.loader import Document
from src.models.embedding import EmbeddingModel
from src.models.qwen_client import QwenClient, MockQwenClient
from src.retriever import ExampleRetriever
from src.four_agent_pipeline import FourAgentPipeline, PipelineConfig


def create_sample_document() -> Document:
    """创建示例文档"""
    sample_text = """
# 合同条款

## 第一章 总则

第一条 本合同由甲方（委托方）和乙方（受托方）共同签订。

甲方责任：
1. 甲方应按照合同约定支付服务费用
2. 甲方应提供必要的技术资料和文档
3. 甲方有权对乙方的工作进行监督和检查

乙方责任：
1. 乙方应按照甲方要求完成开发任务
2. 乙方应保证代码质量和交付时间
3. 乙方应遵守甲方的保密协议

## 第二章 违约责任

第二条 任何一方违反本合同约定，应承担相应的违约责任。

甲方违约责任：
- 如甲方未按时付款，每逾期一天应向乙方支付合同金额的0.1%作为违约金
- 如甲方提供的资料不完整导致项目延期，甲方应承担相应责任

乙方违约责任：
- 如乙方未按时交付，每逾期一天应向甲方支付合同金额的0.1%作为违约金
- 如乙方交付的代码存在严重质量问题，乙方应免费进行修复

## 第三章 争议解决

第三条 因本合同引起的任何争议，双方应友好协商解决。

第四条 如协商不成，任何一方可向合同签订地的人民法院提起诉讼。
    """
    
    return Document(
        doc_id="sample_contract_001",
        text=sample_text,
        title="示例合同",
        metadata={"type": "contract", "language": "zh"}
    )


def create_sample_example_library() -> List[Document]:
    """创建示例示例库"""
    examples = [
        Document(
            doc_id="example_001",
            text="甲方应按照合同约定支付服务费用。甲方逾期付款的，每逾期一天应支付合同金额的0.1%作为违约金。",
            metadata={"type": "example"}
        ),
        Document(
            doc_id="example_002",
            text="乙方应按照甲方要求完成开发任务。乙方交付的代码应符合行业标准，并经过测试验证。",
            metadata={"type": "example"}
        ),
        Document(
            doc_id="example_003",
            text="任何一方违反本合同约定，应承担相应的违约责任。违约方应赔偿守约方因此遭受的全部损失。",
            metadata={"type": "example"}
        )
    ]
    return examples


def main():
    parser = argparse.ArgumentParser(description="四智能体流水线示例")
    parser.add_argument("--task_description", type=str, 
                        default="找出所有责任条款及其主体",
                        help="任务描述")
    parser.add_argument("--doc_id", type=str, 
                        default="sample_contract_001",
                        help="文档ID")
    parser.add_argument("--use_mock_model", action="store_true",
                        help="使用模拟模型")
    parser.add_argument("--api_base", type=str,
                        default=os.getenv("API_BASE", "http://localhost:8000/v1"),
                        help="API地址")
    parser.add_argument("--api_key", type=str,
                        default=os.getenv("API_KEY", ""),
                        help="API密钥")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("四智能体流水线示例")
    print("=" * 60)
    print(f"任务描述: {args.task_description}")
    print(f"文档ID: {args.doc_id}")
    print()
    
    # 1. 创建文档
    document = create_sample_document()
    print(f"文档加载完成: {document.doc_id}")
    print(f"文档长度: {len(document.text)} 字符")
    print()
    
    # 2. 初始化模型客户端
    if args.use_mock_model:
        model_client = MockQwenClient()
        print("使用模拟模型客户端")
    else:
        model_client = QwenClient(
            api_base=args.api_base,
            api_key=args.api_key,
            model_name="Qwen3-4B",
            temperature=0.1
        )
        print(f"使用真实模型客户端: {args.api_base}")
    print()
    
    # 3. 初始化嵌入模型和检索器
    embedding_model = EmbeddingModel(
        model_name="BAAI/bge-m3",
        device="cpu"
    )
    print("嵌入模型初始化完成")
    
    retriever = ExampleRetriever(
        embedding_model=embedding_model,
        top_k=3,
        similarity_threshold=0.5
    )
    
    # 4. 构建示例库索引
    example_docs = create_sample_example_library()
    from src.data.preprocessor import DataPreprocessor
    preprocessor = DataPreprocessor()
    example_library = preprocessor.build_example_library(example_docs)
    retriever.build_index(example_library)
    print(f"示例库索引构建完成: {len(example_library)} 个示例")
    print()
    
    # 5. 配置流水线
    config = PipelineConfig(
        max_context_tokens=8000,
        top_k_examples=3,
        similarity_threshold=0.5,
        max_validation_rounds=2,
        enable_cross_validation=True,
        enable_knowledge_graph=True
    )
    
    # 6. 创建流水线
    pipeline = FourAgentPipeline(
        model_client=model_client,
        retriever=retriever,
        config=config
    )
    print("四智能体流水线初始化完成")
    print()
    
    # 7. 运行流水线
    print("开始处理...")
    result = pipeline.run(
        document=document,
        task_description=args.task_description
    )
    
    # 8. 输出结果
    print()
    print("=" * 60)
    print("处理结果")
    print("=" * 60)
    print(f"文档ID: {result.doc_id}")
    print(f"任务描述: {result.task_description}")
    print(f"处理块数: {result.chunks_processed}")
    print(f"处理时间: {result.processing_time:.2f}秒")
    print(f"提取实体数: {len(result.entities)}")
    print(f"提取关系数: {len(result.relations)}")
    print(f"验证问题数: {len(result.issues)}")
    print()
    
    if result.entities:
        print("提取的实体:")
        for i, entity in enumerate(result.entities[:10], 1):
            print(f"  {i}. {entity.get('text', '')} ({entity.get('type', '未知')})")
        if len(result.entities) > 10:
            print(f"  ... 还有 {len(result.entities) - 10} 个实体")
        print()
    
    if result.relations:
        print("提取的关系:")
        for i, relation in enumerate(result.relations[:10], 1):
            print(f"  {i}. {relation.get('head', '')} --{relation.get('relation', '')}--> {relation.get('tail', '')}")
        if len(result.relations) > 10:
            print(f"  ... 还有 {len(result.relations) - 10} 个关系")
        print()
    
    if result.issues:
        print("验证问题:")
        for issue in result.issues[:5]:
            print(f"  - {issue}")
        if len(result.issues) > 5:
            print(f"  ... 还有 {len(result.issues) - 5} 个问题")
        print()
    
    # 9. 保存结果
    output_file = f"output_{result.doc_id}.json"
    os.makedirs("output", exist_ok=True)
    with open(f"output/{output_file}", "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
    print(f"结果已保存到: output/{output_file}")
    
    # 10. 输出统计
    stats = pipeline.get_stats()
    print()
    print("流水线统计:")
    print(f"  总文档数: {stats['total_documents']}")
    print(f"  处理成功: {stats['processed_documents']}")
    print(f"  处理失败: {stats['failed_documents']}")
    print(f"  总块数: {stats['total_chunks']}")
    print(f"  总实体数: {stats['total_entities']}")
    print(f"  总关系数: {stats['total_relations']}")


if __name__ == "__main__":
    main()