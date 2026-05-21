#!/usr/bin/env python3
"""
四智能体流水线测试

测试各个智能体组件的功能
"""

import unittest
from src.orchestrator import OrchestratorAgent, TaskPlan, TextChunk
from src.playwright import PlaywrightAgent, ChainOfThoughtStep
from src.quality_inspector import QualityInspectorAgent, KnowledgeGraph, ValidationResult
from src.data.loader import Document


class TestOrchestratorAgent(unittest.TestCase):
    """测试管理员智能体"""
    
    def setUp(self):
        self.orchestrator = OrchestratorAgent(
            max_context_tokens=8000,
            min_chunk_size=100,
            max_chunk_size=2000
        )
        self.document = Document(
            doc_id="test_doc_001",
            text="这是一个测试文档。" * 100,  # 较长文档
            title="测试文档"
        )
    
    def test_task_planning(self):
        """测试任务规划"""
        task_description = "找出所有责任条款及其主体"
        task_plan = self.orchestrator.plan_task(task_description, self.document)
        
        self.assertIsInstance(task_plan, TaskPlan)
        self.assertEqual(task_plan.task_description, task_description)
        self.assertIn("responsibility", task_plan.constraints.get("task_type", ""))
        self.assertGreater(len(task_plan.target_entity_types), 0)
    
    def test_document_chunking(self):
        """测试文档切分"""
        task_plan = TaskPlan(
            task_id="test_task",
            task_description="测试任务",
            target_entity_types=["实体"],
            target_relation_types=["关系"],
            estimated_chunks=5,
            priority="medium",
            constraints={"task_type": "general"}
        )
        
        chunks = self.orchestrator.chunk_document(self.document, task_plan)
        
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertIsInstance(chunk, TextChunk)
            self.assertGreater(len(chunk.text), 0)
    
    def test_work_memory_allocation(self):
        """测试工作记忆分配"""
        task_plan = TaskPlan(
            task_id="test_task",
            task_description="测试任务",
            target_entity_types=["实体"],
            target_relation_types=["关系"],
            estimated_chunks=5,
            priority="medium",
            constraints={"task_type": "general"}
        )
        
        chunks = self.orchestrator.chunk_document(self.document, task_plan)
        work_memories = self.orchestrator.allocate_work_memory(chunks, task_plan)
        
        self.assertIsInstance(work_memories, list)
        self.assertGreater(len(work_memories), 0)
        for wm in work_memories:
            self.assertLessEqual(wm.used_tokens, wm.max_tokens)


class TestPlaywrightAgent(unittest.TestCase):
    """测试编剧智能体"""
    
    def setUp(self):
        self.playwright = PlaywrightAgent()
        self.task_plan = TaskPlan(
            task_id="test_task",
            task_description="找出所有责任条款及其主体",
            target_entity_types=["责任主体", "责任内容"],
            target_relation_types=["承担", "履行"],
            estimated_chunks=3,
            priority="high",
            constraints={"task_type": "responsibility"}
        )
        self.chunks = [
            TextChunk(
                chunk_id="chunk_1",
                text="甲方应按照合同约定支付服务费用。",
                start_pos=0,
                end_pos=20,
                section_path=["第一章"],
                chunk_type="paragraph"
            )
        ]
    
    def test_script_generation(self):
        """测试提示剧本生成"""
        script = self.playwright.generate_script(
            task_plan=self.task_plan,
            context_chunks=self.chunks,
            examples=None
        )
        
        self.assertIsNotNone(script)
        self.assertIsNotNone(script.system_prompt)
        self.assertIsNotNone(script.output_template)
        self.assertGreater(len(script.task_decomposition), 0)
    
    def test_cot_steps_defined(self):
        """测试思考链步骤定义"""
        # 验证responsibility类型的思考链
        cot_steps = self.playwright.COT_TEMPLATES.get("responsibility", [])
        self.assertGreater(len(cot_steps), 0)
        
        # 验证步骤结构
        for step in cot_steps:
            self.assertIsInstance(step, ChainOfThoughtStep)
            self.assertGreater(step.step_id, 0)
            self.assertIsNotNone(step.step_name)
            self.assertIsNotNone(step.action)
    
    def test_full_prompt_building(self):
        """测试完整提示构建"""
        script = self.playwright.generate_script(
            task_plan=self.task_plan,
            context_chunks=self.chunks,
            examples=None
        )
        
        full_prompt = self.playwright.build_full_prompt(
            script=script,
            context_text="这是测试文本。",
            knowledge_graph_context=""
        )
        
        self.assertIsNotNone(full_prompt)
        self.assertIn("这是测试文本", full_prompt)


class TestQualityInspectorAgent(unittest.TestCase):
    """测试质检智能体"""
    
    def setUp(self):
        self.inspector = QualityInspectorAgent(max_validation_rounds=2)
        self.task_plan = TaskPlan(
            task_id="test_task",
            task_description="测试任务",
            target_entity_types=["责任主体", "责任内容"],
            target_relation_types=["承担", "履行"],
            estimated_chunks=3,
            priority="high",
            constraints={"task_type": "responsibility"}
        )
    
    def test_basic_validation(self):
        """测试基础验证"""
        annotation = {
            "entities": [
                {"text": "甲方", "type": "责任主体", "start": 0, "end": 2},
                {"text": "付款", "type": "责任内容", "start": 5, "end": 7}
            ],
            "relations": [
                {"head": "甲方", "relation": "承担", "tail": "付款"}
            ]
        }
        
        result = self.inspector.validate(annotation, self.task_plan, "chunk_1")
        
        self.assertIsInstance(result, ValidationResult)
    
    def test_entity_consistency(self):
        """测试实体一致性"""
        # 第一次验证
        annotation1 = {
            "entities": [
                {"text": "甲方", "type": "责任主体", "start": 0, "end": 2}
            ],
            "relations": []
        }
        self.inspector.validate(annotation1, self.task_plan, "chunk_1")
        
        # 第二次验证相同实体
        annotation2 = {
            "entities": [
                {"text": "甲方", "type": "责任主体", "start": 10, "end": 12}
            ],
            "relations": []
        }
        result2 = self.inspector.validate(annotation2, self.task_plan, "chunk_2")
        
        # 实体应该保持一致
        self.assertEqual(len(result2.issues), 0)
    
    def test_knowledge_graph(self):
        """测试知识图谱"""
        kg = KnowledgeGraph()
        
        # 添加实体
        from src.quality_inspector import EntityRecord
        entity = EntityRecord(
            text="甲方",
            normalized_text="甲方",
            entity_type="责任主体",
            chunk_id="chunk_1",
            start_pos=0,
            end_pos=2
        )
        kg.add_entity(entity)
        
        # 查询实体
        canonical = kg.get_canonical_name("甲方")
        self.assertEqual(canonical, "甲方")
        
        # 解析实体
        resolved = kg.resolve_entity("甲方")
        self.assertEqual(resolved, "甲方")
    
    def test_cross_chunk_validation(self):
        """测试跨块验证"""
        annotations = [
            {
                "entities": [
                    {"text": "甲方", "type": "责任主体", "start": 0, "end": 2}
                ],
                "relations": []
            },
            {
                "entities": [
                    {"text": "甲方", "type": "责任主体", "start": 10, "end": 12}
                ],
                "relations": []
            }
        ]
        
        results = self.inspector.cross_validate(
            annotations=annotations,
            chunk_ids=["chunk_1", "chunk_2"]
        )
        
        self.assertEqual(len(results), 2)


class TestKnowledgeGraph(unittest.TestCase):
    """测试知识图谱"""
    
    def test_entity_normalization(self):
        """测试实体标准化"""
        kg = KnowledgeGraph()
        
        # 添加带空格的实体
        from src.quality_inspector import EntityRecord
        entity = EntityRecord(
            text="甲 方",  # 中间有空格
            normalized_text="",
            entity_type="测试",
            chunk_id="chunk_1",
            start_pos=0,
            end_pos=3
        )
        kg.add_entity(entity)
        
        # 查询时应该能匹配
        canonical = kg.get_canonical_name("甲方")
        self.assertIsNotNone(canonical)
    
    def test_consistency_check(self):
        """测试一致性检查"""
        kg = KnowledgeGraph()
        
        from src.quality_inspector import EntityRecord
        
        # 添加同一实体但不同类型
        entity1 = EntityRecord(
            text="测试",
            normalized_text="",
            entity_type="类型A",
            chunk_id="chunk_1",
            start_pos=0,
            end_pos=2
        )
        entity2 = EntityRecord(
            text="测试",
            normalized_text="",
            entity_type="类型B",  # 不同类型
            chunk_id="chunk_2",
            start_pos=10,
            end_pos=12
        )
        
        kg.add_entity(entity1)
        kg.add_entity(entity2)
        
        issues = kg.check_consistency()
        self.assertGreater(len(issues), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)