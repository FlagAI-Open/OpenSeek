#!/usr/bin/env python3
"""
编剧智能体 (Playwright Agent)
生成动态、结构化的提示剧本，严格遵循"角色定义-任务分解-输出格式"框架，
并为模型预设"思考链"（Chain-of-Thought）路径
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from string import Template
from loguru import logger

from .orchestrator import TextChunk, TaskPlan, WorkMemory
from .retriever import RetrievedExample


@dataclass
class ChainOfThoughtStep:
    """思考链步骤"""
    step_id: int
    step_name: str
    description: str
    action: str  # 模型需要执行的动作
    output_format: str  # 该步骤的输出格式
    validation_rules: List[str] = field(default_factory=list)


@dataclass
class PromptScript:
    """提示剧本"""
    system_prompt: str
    task_decomposition: List[ChainOfThoughtStep]
    few_shot_examples: List[Dict[str, str]]
    output_template: str
    constraints: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "system_prompt": self.system_prompt,
            "task_decomposition": [
                {
                    "step_id": s.step_id,
                    "step_name": s.step_name,
                    "description": s.description,
                    "action": s.action,
                    "output_format": s.output_format
                } for s in self.task_decomposition
            ],
            "few_shot_examples": self.few_shot_examples,
            "output_template": self.output_template,
            "constraints": self.constraints
        }


class PlaywrightAgent:
    """
    编剧智能体
    
    核心职责：
    1. 生成动态、结构化的提示剧本
    2. 预设Chain-of-Thought路径
    3. 整合示例和上下文
    """
    
    # 任务类型对应的思考链模板
    COT_TEMPLATES = {
        "responsibility": [
            ChainOfThoughtStep(
                step_id=1,
                step_name="识别责任陈述",
                description="判断文本中是否包含责任条款",
                action="仔细阅读文本，判断是否包含责任声明、义务要求或约束性条款",
                output_format="是/否",
                validation_rules=["如果答案为否，跳过后续步骤"]
            ),
            ChainOfThoughtStep(
                step_id=2,
                step_name="提取责任主体",
                description="识别承担责任的主体",
                action="从文本中找出承担责任的实体（如个人、组织、部门等）",
                output_format="责任主体列表",
                validation_rules=["主体必须明确出现在文本中", "排除假设性主体"]
            ),
            ChainOfThoughtStep(
                step_id=3,
                step_name="提取责任内容",
                description="识别具体责任内容",
                action="明确责任主体需要做什么或避免做什么",
                output_format="责任内容列表",
                validation_rules=["责任内容必须具体可执行", "排除模糊描述"]
            ),
            ChainOfThoughtStep(
                step_id=4,
                step_name="判断责任性质",
                description="判断是积极责任还是消极责任",
                action="积极责任=必须做某事；消极责任=必须避免做某事",
                output_format="积极/消极",
                validation_rules=["根据责任表述的措辞判断"]
            ),
            ChainOfThoughtStep(
                step_id=5,
                step_name="提取责任对象",
                description="识别责任指向的对象",
                action="责任是对谁/什么承担的",
                output_format="责任对象列表",
                validation_rules=["对象可以是个人、群体、行为或结果"]
            )
        ],
        "entity_extraction": [
            ChainOfThoughtStep(
                step_id=1,
                step_name="识别实体边界",
                description="找出文本中的实体提及",
                action="扫描文本，识别所有可能的实体提及",
                output_format="实体提及列表",
                validation_rules=["实体必须具有明确边界", "排除修饰性短语"]
            ),
            ChainOfThoughtStep(
                step_id=2,
                step_name="分类实体类型",
                description="为每个实体分配类型",
                action="根据预设的类型体系，为每个实体分配类型标签",
                output_format="{实体: 类型} 映射",
                validation_rules=["每个实体只能属于一个类型", "使用标准类型名称"]
            ),
            ChainOfThoughtStep(
                step_id=3,
                step_name="消歧处理",
                description="处理实体歧义",
                action="对于同名实体，根据上下文判断是否指代同一实体",
                output_format="消歧后的实体列表",
                validation_rules=["同名不同义需要区分", "同名同义需要合并"]
            )
        ],
        "relation_extraction": [
            ChainOfThoughtStep(
                step_id=1,
                step_name="识别实体对",
                description="找出文本中存在的实体对",
                action="扫描文本，识别同时出现的两个实体",
                output_format="实体对列表",
                validation_rules=["实体对必须在同一句子或相邻句子中"]
            ),
            ChainOfThoughtStep(
                step_id=2,
                step_name="判断关系类型",
                description="确定实体对之间的关系",
                action="根据上下文和预设的关系类型，判断实体对之间的关系",
                output_format="关系类型标签",
                validation_rules=["关系必须有文本证据支持", "排除虚假关系"]
            ),
            ChainOfThoughtStep(
                step_id=3,
                step_name="验证关系完整性",
                description="确保关系信息完整",
                action="检查每个关系是否有头实体、尾实体和关系类型",
                output_format="完整关系列表",
                validation_rules=["每个关系必须包含头尾实体和类型"]
            )
        ],
        "general": [
            ChainOfThoughtStep(
                step_id=1,
                step_name="理解任务要求",
                description="明确需要提取什么信息",
                action="回顾任务描述，确定提取目标",
                output_format="任务目标列表",
                validation_rules=["目标必须具体可衡量"]
            ),
            ChainOfThoughtStep(
                step_id=2,
                step_name="定位相关信息",
                description="在文本中找出与任务相关的内容",
                action="扫描文本，标记与任务相关的内容区域",
                output_format="相关文本片段列表",
                validation_rules=["相关性判断要准确", "排除干扰信息"]
            ),
            ChainOfThoughtStep(
                step_id=3,
                step_name="提取目标信息",
                description="从相关片段中提取信息",
                action="按照任务要求，从相关片段中提取信息",
                output_format="结构化信息",
                validation_rules=["信息必须准确", "格式必须规范"]
            )
        ]
    }
    
    # 系统提示词模板
    SYSTEM_PROMPT_TEMPLATE = Template("""你是一个专业的数据标注助手，擅长从文档中提取结构化信息。

你的角色：
- 严格遵循预定义的思考链步骤进行推理
- 确保提取的信息准确、完整
- 保持命名一致性

当前任务类型：${task_type}
实体类型：${entity_types}
关系类型：${relation_types}

标注规范：
${annotation_rules}

重要约束：
1. 只提取有明确文本证据支持的信息
2. 实体命名在整个文档中必须一致
3. 关系必须基于实际文本关系
4. 严格遵循输出格式要求""")
    
    # 思考链提示词模板
    COT_PROMPT_TEMPLATE = Template("""## 思考链执行

请严格按照以下步骤执行，每个步骤都要给出明确的推理过程：

${cot_steps}

## 步骤执行要求

1. 每个步骤都要在<step_N>标签中展示你的推理过程
2. 推理过程必须基于提供的文本内容
3. 不要跳过任何步骤
4. 如果某步骤不适用，明确说明原因

## 输出格式

完成所有步骤后，按以下JSON格式输出结果：
${output_template}

请开始执行思考链：""")
    
    def __init__(self):
        """初始化编剧智能体"""
        logger.info("编剧智能体初始化")
    
    def generate_script(
        self,
        task_plan: TaskPlan,
        context_chunks: List[TextChunk],
        examples: Optional[List[RetrievedExample]] = None
    ) -> PromptScript:
        """
        生成提示剧本
        
        Args:
            task_plan: 任务规划
            context_chunks: 上下文块
            examples: 检索到的示例
            
        Returns:
            提示剧本
        """
        logger.info(f"生成提示剧本: {task_plan.task_id}")
        
        # 1. 获取任务类型对应的思考链模板
        task_type = task_plan.constraints.get("task_type", "general")
        cot_steps = self.COT_TEMPLATES.get(task_type, self.COT_TEMPLATES["general"])
        
        # 2. 生成系统提示词
        system_prompt = self._build_system_prompt(task_plan)
        
        # 3. 构建思考链提示词
        cot_prompt = self._build_cot_prompt(cot_steps, task_plan)
        
        # 4. 整合示例
        few_shot_examples = self._format_examples(examples, task_type) if examples else []
        
        # 5. 生成输出模板
        output_template = self._generate_output_template(task_plan)
        
        # 6. 生成约束条件
        constraints = self._generate_constraints(task_plan)
        
        script = PromptScript(
            system_prompt=system_prompt + "\n\n" + cot_prompt,
            task_decomposition=cot_steps,
            few_shot_examples=few_shot_examples,
            output_template=output_template,
            constraints=constraints
        )
        
        logger.info(f"提示剧本生成完成: {len(cot_steps)} 个思考链步骤")
        
        return script
    
    def build_full_prompt(
        self,
        script: PromptScript,
        context_text: str,
        knowledge_graph_context: Optional[str] = None
    ) -> str:
        """
        构建完整提示词
        
        Args:
            script: 提示剧本
            context_text: 上下文文本
            knowledge_graph_context: 知识图谱上下文（用于实体一致性）
            
        Returns:
            完整提示词
        """
        prompt_parts = []
        
        # 1. 系统提示词
        prompt_parts.append(f"# 系统提示\n{script.system_prompt}")
        
        # 2. 知识图谱上下文（如果存在）
        if knowledge_graph_context:
            prompt_parts.append(f"\n# 全局实体一致性参考\n{knowledge_graph_context}")
        
        # 3. Few-shot示例
        if script.few_shot_examples:
            prompt_parts.append("\n# 参考示例")
            for i, example in enumerate(script.few_shot_examples):
                prompt_parts.append(f"\n## 示例 {i+1}")
                prompt_parts.append(f"输入：{example.get('input', '')}")
                prompt_parts.append(f"输出：{example.get('output', '')}")
        
        # 4. 待处理文本
        prompt_parts.append(f"\n# 待标注文本\n{context_text}")
        
        # 5. 输出要求
        prompt_parts.append(f"\n# 输出要求\n{script.output_template}")
        
        # 6. 约束条件
        if script.constraints:
            prompt_parts.append("\n# 约束条件")
            for constraint in script.constraints:
                prompt_parts.append(f"- {constraint}")
        
        return "\n\n".join(prompt_parts)
    
    def _build_system_prompt(self, task_plan: TaskPlan) -> str:
        """构建系统提示词"""
        task_type = task_plan.constraints.get("task_type", "general")
        
        # 生成标注规则
        annotation_rules = []
        for entity_type in task_plan.target_entity_types:
            annotation_rules.append(f"- {entity_type}：定义和边界要清晰")
        for relation_type in task_plan.target_relation_types:
            annotation_rules.append(f"- {relation_type}：必须有明确的文本证据")
        
        return self.SYSTEM_PROMPT_TEMPLATE.substitute(
            task_type=task_type,
            entity_types=", ".join(task_plan.target_entity_types),
            relation_types=", ".join(task_plan.target_relation_types),
            annotation_rules="\n".join(annotation_rules)
        )
    
    def _build_cot_prompt(
        self,
        cot_steps: List[ChainOfThoughtStep],
        task_plan: TaskPlan
    ) -> str:
        """构建思考链提示词"""
        steps_text = []
        
        for step in cot_steps:
            steps_text.append(f"""### 步骤{step.step_id}：{step.step_name}
描述：{step.description}
动作：{step.action}
输出格式：{step.output_format}
验证规则：{', '.join(step.validation_rules) if step.validation_rules else '无'}

<step_{step.step_id}>
[在此处展示你的推理过程]
</step_{step.step_id}>""")
        
        output_template = self._generate_output_template(task_plan)
        
        return self.COT_PROMPT_TEMPLATE.substitute(
            cot_steps="\n\n".join(steps_text),
            output_template=output_template
        )
    
    def _format_examples(
        self,
        examples: List[RetrievedExample],
        task_type: str
    ) -> List[Dict[str, str]]:
        """格式化示例"""
        formatted = []
        
        for example in examples[:3]:  # 最多3个示例
            formatted.append({
                "input": example.example.text_chunk[:200] + "..." if len(example.example.text_chunk) > 200 else example.example.text_chunk,
                "output": json.dumps(example.example.annotation, ensure_ascii=False, indent=2) if example.example.annotation else "{}"
            })
        
        return formatted
    
    def _generate_output_template(self, task_plan: TaskPlan) -> str:
        """生成输出模板"""
        # 构建实体模板
        entity_template_parts = []
        for entity_type in task_plan.target_entity_types:
            entity_template_parts.append(f'{{"text": "实体文本", "type": "{entity_type}", "start": 0, "end": 0}}')
        
        # 构建关系模板
        relation_template_parts = []
        for relation_type in task_plan.target_relation_types:
            relation_template_parts.append(f'{{"head": "头实体", "relation": "{relation_type}", "tail": "尾实体"}}')
        
        template = {
            "entities": [
                json.loads(et) for et in entity_template_parts
            ],
            "relations": [
                json.loads(rt) for rt in relation_template_parts
            ] if relation_template_parts else []
        }
        
        return json.dumps(template, ensure_ascii=False, indent=2)
    
    def _generate_constraints(self, task_plan: TaskPlan) -> List[str]:
        """生成约束条件"""
        constraints = [
            "只提取有明确文本证据支持的信息",
            "实体命名在整个文档中必须保持一致",
            "关系必须基于实际文本关系",
            "不要遗漏重要的实体和关系",
            "不要添加文本中没有的信息"
        ]
        
        # 根据任务类型添加特定约束
        task_type = task_plan.constraints.get("task_type", "general")
        
        if task_type == "responsibility":
            constraints.extend([
                "责任主体必须明确",
                "责任内容必须具体可执行",
                "区分积极责任和消极责任"
            ])
        elif task_type == "entity_extraction":
            constraints.extend([
                "实体边界必须清晰",
                "每个实体只能属于一种类型",
                "同名实体需要消歧"
            ])
        elif task_type == "relation_extraction":
            constraints.extend([
                "关系必须有头尾实体",
                "避免虚假关系",
                "关系类型必须准确"
            ])
        
        return constraints
    
    def extract_cot_steps(self, task_description: str) -> List[ChainOfThoughtStep]:
        """
        从任务描述中提取自定义思考链步骤
        
        Args:
            task_description: 任务描述
            
        Returns:
            思考链步骤列表
        """
        # 尝试从任务描述中解析步骤
        steps = []
        
        # 查找"步骤一"、"步骤二"等模式
        step_pattern = r'步骤[一二三四五六七八九十]+[：:]\s*(.+)'
        matches = re.findall(step_pattern, task_description)
        
        if matches:
            for i, match in enumerate(matches):
                steps.append(ChainOfThoughtStep(
                    step_id=i + 1,
                    step_name=f"步骤{i+1}",
                    description=match.strip(),
                    action=f"执行{match.strip()}",
                    output_format="待确定",
                    validation_rules=[]
                ))
        
        return steps