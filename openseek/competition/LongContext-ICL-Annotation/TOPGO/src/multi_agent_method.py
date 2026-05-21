#!/usr/bin/env python3
"""
多智能体协作的答案提取系统
解决推理过程污染、答案不完整、格式错误等问题
"""

import re
import json
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class AgentResult:
    """智能体结果"""
    success: bool
    content: str
    confidence: float
    message: str


class ReasoningAgent:
    """推理智能体 - 负责思考和推理"""
    
    @staticmethod
    def build_prompt(task_description: str, text2annotate: str, task_id: int) -> str:
        """构建推理提示词"""
        return f"""### 任务
{task_description}

### 待标注文本
{text2annotate}

### 要求
1. 仔细分析问题
2. 进行必要的推理和思考
3. 在<thinking>标签中展示你的思考过程
4. 在<answer>标签中给出最终答案
5. 答案要简洁准确

### 输出格式
<thinking>
[你的推理过程]
</thinking>

<answer>
[最终答案]
</answer>
"""


class ValidationAgent:
    """验证智能体 - 负责验证答案的正确性"""
    
    @staticmethod
    def validate(original_input: str, reasoning: str, answer: str, task_id: int) -> AgentResult:
        """
        验证答案的正确性
        
        返回：
        - success: 是否通过验证
        - content: 修正后的答案（如果需要）
        - confidence: 置信度
        - message: 验证信息
        """
        # 检查答案是否为空
        if not answer or answer.strip() == '':
            return AgentResult(False, '', 0.0, '答案为空')
        
        # 检查答案是否包含推理关键词（污染）
        reasoning_keywords = ['因为', '所以', '首先', '然后', '我认为', '因此', '答案是']
        has_pollution = any(kw in answer for kw in reasoning_keywords)
        
        if has_pollution:
            # 尝试清理
            cleaned = ValidationAgent._clean_pollution(answer)
            if cleaned != answer:
                return AgentResult(True, cleaned, 0.7, f'清理了污染，原答案: {answer[:50]}')
            return AgentResult(False, answer, 0.3, '答案包含推理过程污染')
        
        # 检查答案长度
        if task_id == 7:  # 阅读理解任务，答案应该很短
            if len(answer) > 100:
                return AgentResult(False, answer[:100], 0.5, '答案过长，可能包含额外内容')
        
        # 检查答案格式
        if task_id == 2:  # 计数任务，应该是数字
            if not answer.strip().isdigit():
                # 尝试提取数字
                numbers = re.findall(r'\d+', answer)
                if numbers:
                    return AgentResult(True, numbers[-1], 0.8, f'提取了数字: {numbers[-1]}')
                return AgentResult(False, answer, 0.3, '答案不是有效数字')
        
        if task_id == 6:  # MNLI任务，应该是Y或N
            answer_upper = answer.strip().upper()
            if answer_upper not in ['Y', 'N', 'YES', 'NO']:
                return AgentResult(False, answer, 0.3, '答案不是Y/N')
            if answer_upper in ['YES', 'NO']:
                return AgentResult(True, 'Y' if answer_upper == 'YES' else 'N', 0.9, '标准化为Y/N')
        
        return AgentResult(True, answer, 0.9, '答案通过验证')
    
    @staticmethod
    def _clean_pollution(text: str) -> str:
        """清理答案中的推理过程污染"""
        # 移除常见的推理句式
        patterns = [
            r'因为.*?所以[，,]?',
            r'首先.*?然后[，,]?',
            r'我认为.*?[，,]?',
            r'因此[，,]?',
            r'答案是[：:]?',
        ]
        
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.DOTALL)
        
        return text.strip()


class ExtractionAgent:
    """提取智能体 - 负责从推理结果中提取最终答案"""
    
    @staticmethod
    def extract(text: str, task_id: int) -> AgentResult:
        """
        从文本中提取最终答案
        
        优先级：
        1. <answer>标签
        2. <label>标签
        3. 代码块
        4. 列表格式
        5. 最后一行（对于短答案任务）
        """
        if not text:
            return AgentResult(False, '', 0.0, '输入为空')
        
        # 1. 尝试提取 <answer> 标签
        answer_pattern = r'<answer>\s*(.*?)\s*</answer>'
        answer_matches = re.findall(answer_pattern, text, re.DOTALL)
        if answer_matches:
            return AgentResult(True, answer_matches[-1].strip(), 0.95, '从<answer>标签提取')
        
        # 2. 尝试提取 <label> 标签
        label_pattern = r'<label>\s*(.*?)\s*</label>'
        label_matches = re.findall(label_pattern, text, re.DOTALL)
        if label_matches:
            return AgentResult(True, label_matches[-1].strip(), 0.9, '从<label>标签提取')
        
        # 3. 尝试提取代码块（针对任务8）
        code_pattern = r'```(?:python)?\s*\n(.*?)\n```'
        code_matches = re.findall(code_pattern, text, re.DOTALL)
        if code_matches:
            return AgentResult(True, code_matches[-1].strip(), 0.95, '从代码块提取')
        
        # 4. 尝试提取列表格式
        list_pattern = r'\[[\d\s,\-]+\]'
        list_matches = re.findall(list_pattern, text)
        if list_matches:
            return AgentResult(True, list_matches[-1], 0.85, '从列表格式提取')
        
        # 5. 对于短答案任务，提取最后一行或最后一个短句
        if task_id in [2, 4, 6, 7]:  # 这些任务的答案通常很短
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            
            # 过滤推理过程
            clean_lines = []
            for line in reversed(lines):
                if not any(kw in line for kw in ['因为', '所以', '首先', '然后', '我认为', '因此']):
                    clean_lines.append(line)
                    if len(clean_lines) >= 3:
                        break
            
            if clean_lines:
                # 取最后一个短行
                for line in clean_lines:
                    if len(line) < 50:
                        return AgentResult(True, line, 0.7, f'从最后一行提取: {line[:30]}')
        
        # 6. 返回整个文本（作为最后的手段）
        return AgentResult(True, text.strip(), 0.5, '返回完整文本')


class FormattingAgent:
    """格式化智能体 - 负责确保输出格式正确"""
    
    @staticmethod
    def format(answer: str, task_id: int) -> str:
        """
        确保答案格式正确
        
        规则：
        - 移除多余的空格和换行
        - 移除包裹标记
        - 根据任务类型进行特定格式化
        """
        if not answer:
            return answer
        
        # 移除包裹标记
        answer = re.sub(r'</?(?:label|answer|think)>', '', answer)
        
        # 移除多余的空格
        answer = ' '.join(answer.split())
        
        # 任务特定的格式化
        if task_id == 2:  # 计数任务
            # 确保是数字
            numbers = re.findall(r'\d+', answer)
            if numbers:
                return numbers[-1]
        
        elif task_id == 4:  # 字符串连接任务
            # 移除可能的多余内容
            answer = answer.strip()
            # 移除末尾标点
            answer = answer.rstrip('。，,.!?')
            # 如果包含中文，尝试提取非中文部分
            if re.search(r'[\u4e00-\u9fff]', answer):
                parts = re.split(r'[，。！？,.]', answer)
                for part in parts:
                    if part.strip() and not re.search(r'[\u4e00-\u9fff]', part.strip()):
                        return part.strip()
        
        elif task_id == 6:  # MNLI任务
            # 确保是Y或N
            answer_upper = answer.strip().upper()
            if answer_upper in ['YES', 'Y']:
                return 'Y'
            elif answer_upper in ['NO', 'N']:
                return 'N'
        
        elif task_id == 7:  # 阅读理解任务
            # 移除末尾标点
            answer = answer.rstrip('。，,.!?')
            # 确保答案简洁
            if len(answer) > 100:
                # 尝试提取最后一个短句
                sentences = re.split(r'[。！？.!?]', answer)
                for sentence in reversed(sentences):
                    if sentence.strip() and len(sentence.strip()) < 50:
                        return sentence.strip()
        
        return answer.strip()


class MultiAgentCoordinator:
    """多智能体协调器"""
    
    @staticmethod
    def process(model_output: str, original_input: str, task_id: int) -> Optional[str]:
        """
        使用多智能体协作处理模型输出
        
        流程：
        1. 提取智能体：从模型输出中提取答案
        2. 验证智能体：验证答案的正确性
        3. 格式化智能体：确保输出格式正确
        
        返回：最终答案
        """
        # 步骤1：提取答案
        extraction_result = ExtractionAgent.extract(model_output, task_id)
        
        if not extraction_result.success:
            # 如果提取失败，尝试其他方法
            print(f"[提取失败] {extraction_result.message}")
            return None
        
        answer = extraction_result.content
        print(f"[提取成功] {extraction_result.message}: {answer[:50]}")
        
        # 步骤2：验证答案
        validation_result = ValidationAgent.validate(
            original_input, 
            model_output, 
            answer, 
            task_id
        )
        
        if validation_result.content != answer:
            print(f"[验证修正] {validation_result.message}")
            answer = validation_result.content
        
        if not validation_result.success and validation_result.confidence < 0.5:
            print(f"[验证失败] 置信度过低: {validation_result.confidence}")
            # 但仍然继续处理，因为可能是部分正确
        
        # 步骤3：格式化答案
        final_answer = FormattingAgent.format(answer, task_id)
        
        print(f"[格式化完成] 最终答案: {final_answer[:50]}")
        
        return final_answer


# 用于替换原有的 count_answer 函数
def count_answer_multi_agent(text: str, task_id: int = None, original_input: str = '') -> Optional[str]:
    """
    使用多智能体协作提取答案
    
    Args:
        text: 模型输出文本
        task_id: 任务ID
        original_input: 原始输入（用于验证）
    
    Returns:
        提取的答案
    """
    if not text:
        return None
    
    return MultiAgentCoordinator.process(text, original_input, task_id or 0)


if __name__ == "__main__":
    # 测试用例
    test_cases = [
        {
            'task_id': 7,
            'output': '<thinking>让我想想，法国的首都是哪里呢？应该是巴黎。</thinking><answer>Paris</answer>',
            'input': '法国的首都是哪里？',
            'expected': 'Paris'
        },
        {
            'task_id': 2,
            'output': '因为这个句子有3个名词，所以答案是3。',
            'input': 'Count the nouns',
            'expected': '3'
        },
        {
            'task_id': 6,
            'output': '<label>YES</label>',
            'input': 'Is this true?',
            'expected': 'Y'
        }
    ]
    
    print("="*70)
    print("多智能体答案提取测试")
    print("="*70)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n测试用例 {i}:")
        print(f"任务ID: {test['task_id']}")
        print(f"模型输出: {test['output'][:50]}...")
        print(f"期望输出: {test['expected']}")
        
        result = count_answer_multi_agent(
            test['output'],
            test['task_id'],
            test['input']
        )
        
        print(f"实际输出: {result}")
        print(f"测试结果: {'✅ 通过' if result == test['expected'] else '❌ 失败'}")
    
    print("\n" + "="*70)