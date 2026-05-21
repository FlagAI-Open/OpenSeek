#!/usr/bin/env python3
"""
轻量级多智能体答案提取系统 - 2.0版本

设计哲学：
1. 不过度约束模型的推理过程
2. 在底线处（结果输出）建立护栏
3. 动态纠偏而非硬性阻断
4. 给模型自主纠偏的机会

参考：自主搜索agent反思 - 阿里云Data+AI工程师大奖赛
"""

import re
import json
from typing import Optional, Dict, Any, List
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    cleaned_answer: str
    confidence: float
    issues: List[str]


class LightweightAnswerExtractor:
    """
    轻量级答案提取器
    
    设计原则：
    - 最小干预：只在必要时介入
    - 动态纠偏：检测问题后引导模型修正
    - 底线思维：确保输出格式正确即可
    """
    
    # 最小化的过滤规则（底线）
    MINIMAL_FILTERS = {
        # 必须过滤的内容（硬底线）
        'hard_blocks': [
            r'</?think>',  # 思考标记
            r'</?thinking>',
            r'<label>.*?</label>',  # 包裹标记（但提取后会移除）
        ],
        # 不建议但允许的内容（软底线）
        'soft_warnings': [
            r'因为.*?所以',  # 推理句式
            r'首先.*?然后',
            r'我认为',
        ]
    }
    
    @staticmethod
    def extract_with_minimal_intervention(
        model_output: str, 
        task_id: int,
        original_input: str = ''
    ) -> Optional[str]:
        """
        最小干预的答案提取
        
        流程：
        1. 提取答案（优先使用标签）
        2. 验证底线规则
        3. 如果触犯底线，尝试清理
        4. 返回结果
        """
        if not model_output:
            return None
        
        # Step 1: 提取答案（优先级顺序）
        answer = LightweightAnswerExtractor._extract_answer(model_output, task_id)
        
        if not answer:
            return None
        
        # Step 2: 验证底线规则
        validation = LightweightAnswerExtractor._validate_answer(
            answer, 
            task_id, 
            original_input
        )
        
        # Step 3: 如果有问题，尝试清理
        if not validation.is_valid:
            # 只在触犯硬底线时才清理
            if validation.issues:
                cleaned = LightweightAnswerExtractor._clean_answer(
                    answer, 
                    task_id,
                    validation.issues
                )
                return cleaned
        
        return validation.cleaned_answer
    
    @staticmethod
    def _extract_answer(text: str, task_id: int) -> Optional[str]:
        """
        提取答案 - 优先级驱动
        
        优先级：
        1. <answer> 标签（模型明确标记的答案）
        2. <label> 标签（包裹标记）
        3. 代码块（任务8）
        4. 列表格式
        5. 自由文本（对于短答案任务，取最后一段）
        """
        # 优先级1: <answer> 标签
        match = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # 优先级2: <label> 标签
        match = re.search(r'<label>\s*(.*?)\s*</label>', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # 优先级3: 代码块（仅任务8）
        if task_id == 8:
            match = re.search(r'```(?:python)?\s*\n(.*?)\n```', text, re.DOTALL)
            if match:
                return match.group(1).strip()
            
            # 如果包含import语句，直接返回从import开始的内容
            if 'import torch' in text or 'import triton' in text:
                match = re.search(r'(import torch.*)', text, re.DOTALL)
                if match:
                    return match.group(1).strip()
        
        # 优先级4: 列表格式
        match = re.search(r'\[[\d\s,\-]+\]', text)
        if match:
            return match.group(0)
        
        # 优先级5: 自由文本
        # 对于短答案任务（2, 4, 6, 7），尝试提取关键信息
        if task_id in [2, 4, 6, 7]:
            # 先尝试提取数字（任务2）
            if task_id == 2:
                numbers = re.findall(r'\d+', text)
                if numbers:
                    return numbers[-1]
            
            # 尝试提取Y/N（任务6）
            if task_id == 6:
                text_upper = text.upper()
                if 'YES' in text_upper or ' Y ' in text_upper:
                    return 'Y'
                elif 'NO' in text_upper or ' N ' in text_upper:
                    return 'N'
            
            # 对于任务7，提取最后一个短句
            if task_id == 7:
                lines = [l.strip() for l in text.split('\n') if l.strip()]
                if lines:
                    # 取最后一个短行（通常是答案）
                    for line in reversed(lines):
                        if len(line) < 100:  # 短答案
                            return line
        
        # 最后返回整个文本
        return text.strip()
    
    @staticmethod
    def _validate_answer(
        answer: str, 
        task_id: int,
        original_input: str = ''
    ) -> ValidationResult:
        """
        验证答案 - 底线思维
        
        只检查底线规则：
        1. 答案是否为空
        2. 是否触犯硬底线（必须修复）
        3. 任务特定格式
        """
        issues = []
        
        # 检查1: 空答案
        if not answer or not answer.strip():
            return ValidationResult(False, '', 0.0, ['答案为空'])
        
        confidence = 0.9  # 初始置信度
        cleaned = answer.strip()
        
        # 检查2: 硬底线 - 思考标记
        if re.search(r'</?think(?:ing)?>', cleaned):
            issues.append('包含思考标记')
            # 移除思考标记
            cleaned = re.sub(r'</?think(?:ing)?>', '', cleaned, flags=re.DOTALL)
        
        # 检查3: 任务特定格式（底线）
        if task_id == 2:  # 计数任务
            if not cleaned.strip().isdigit():
                # 尝试提取数字
                numbers = re.findall(r'\d+', cleaned)
                if numbers:
                    cleaned = numbers[-1]
                    issues.append(f'从文本中提取数字: {cleaned}')
                else:
                    issues.append('答案不是有效数字')
                    confidence -= 0.3
        
        elif task_id == 6:  # MNLI任务
            cleaned_upper = cleaned.strip().upper()
            if cleaned_upper in ['YES', 'Y']:
                cleaned = 'Y'
            elif cleaned_upper in ['NO', 'N']:
                cleaned = 'N'
            else:
                issues.append('答案不是Y/N')
                confidence -= 0.3
        
        elif task_id == 7:  # 阅读理解
            # 移除末尾标点
            cleaned = cleaned.rstrip('。，,.!?')
            # 检查是否过长
            if len(cleaned) > 200:
                issues.append('答案可能过长')
                confidence -= 0.2
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            cleaned_answer=cleaned,
            confidence=confidence,
            issues=issues
        )
    
    @staticmethod
    def _clean_answer(
        answer: str, 
        task_id: int,
        issues: List[str]
    ) -> str:
        """
        清理答案 - 动态纠偏
        
        根据具体问题进行针对性清理
        """
        cleaned = answer
        
        # 移除包裹标记
        cleaned = re.sub(r'</?(?:label|answer|think(?:ing)?)>', '', cleaned)
        
        # 移除推理句式（动态纠偏）
        for pattern in LightweightAnswerExtractor.MINIMAL_FILTERS['soft_warnings']:
            cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL)
        
        # 任务特定清理
        if task_id == 7:
            # 移除末尾标点
            cleaned = cleaned.rstrip('。，,.!?')
            # 如果仍然很长，提取最后一个短句
            if len(cleaned) > 100:
                sentences = re.split(r'[。！？.!?]', cleaned)
                for sentence in reversed(sentences):
                    if sentence.strip() and len(sentence.strip()) < 50:
                        cleaned = sentence.strip()
                        break
        
        # 清理多余空格
        cleaned = ' '.join(cleaned.split())
        
        return cleaned.strip()


# 向后兼容的接口
def extract_answer_lightweight(
    model_output: str,
    task_id: int,
    original_input: str = ''
) -> Optional[str]:
    """
    轻量级答案提取接口
    
    Args:
        model_output: 模型输出
        task_id: 任务ID
        original_input: 原始输入
    
    Returns:
        提取的答案
    """
    return LightweightAnswerExtractor.extract_with_minimal_intervention(
        model_output,
        task_id,
        original_input
    )


if __name__ == "__main__":
    # 测试用例
    test_cases = [
        {
            'task_id': 7,
            'output': '因为法国的首都是巴黎，所以答案是Paris。',
            'expected': 'Paris'
        },
        {
            'task_id': 2,
            'output': '经过分析，这个句子包含3个名词。',
            'expected': '3'
        },
        {
            'task_id': 6,
            'output': '<label>YES</label>',
            'expected': 'Y'
        },
    ]
    
    print("="*70)
    print("轻量级答案提取系统测试")
    print("="*70)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n测试 {i}: 任务{test['task_id']}")
        print(f"输入: {test['output'][:50]}...")
        print(f"期望: {test['expected']}")
        
        result = extract_answer_lightweight(
            test['output'],
            test['task_id']
        )
        
        print(f"输出: {result}")
        print(f"状态: {'✅' if result == test['expected'] else '❌'}")