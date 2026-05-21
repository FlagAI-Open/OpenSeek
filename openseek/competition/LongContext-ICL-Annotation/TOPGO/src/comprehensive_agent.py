#!/usr/bin/env python3
"""
综合答案提取系统 - 融合最佳实践

参考：
1. 阿里云Data+AI大奖赛 - 轻量级设计
2. Agent实现方式详解 - 答案提取与归一化
3. 错误处理与重试机制
4. 性能优化技巧
"""

import re
import json
import time
import logging
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from functools import lru_cache
from collections import defaultdict


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """提取结果"""
    answer: str
    confidence: float
    extraction_method: str
    issues: List[str]
    normalized: bool


class ComprehensiveAnswerExtractor:
    """
    综合答案提取器
    
    融合最佳实践：
    1. 多格式支持（JSON、label、finish等）
    2. 答案归一化（数值、多实体）
    3. 答案验证（空值、过长、推理过程）
    4. 重试机制（提取失败时尝试不同策略）
    """
    
    # 任务特定配置
    TASK_CONFIGS = {
        1: {  # 最接近整数
            'max_length': 50,
            'type': 'list',
            'validate': lambda x: re.match(r'\[[\d\s,\-]+\]', x)
        },
        2: {  # 名词/动词计数
            'max_length': 10,
            'type': 'number',
            'validate': lambda x: x.isdigit()
        },
        3: {  # Collatz猜想
            'max_length': 50,
            'type': 'number',
            'validate': lambda x: x.isdigit()
        },
        4: {  # 字符串连接
            'max_length': 200,
            'type': 'string',
            'validate': lambda x: len(x) <= 200
        },
        5: {  # 情感分析
            'max_length': 20,
            'type': 'string',
            'validate': lambda x: x in ['Sad', 'Not sad']
        },
        6: {  # MNLI分类
            'max_length': 5,
            'type': 'enum',
            'validate': lambda x: x in ['Y', 'N']
        },
        7: {  # 阅读理解
            'max_length': 200,
            'type': 'string',
            'validate': lambda x: len(x) <= 200
        },
        8: {  # Triton代码
            'max_length': 20000,
            'type': 'code',
            'validate': lambda x: 'import' in x
        }
    }
    
    @staticmethod
    def extract(
        model_output: str,
        task_id: int,
        original_input: str = '',
        max_retries: int = 3
    ) -> Optional[str]:
        """
        综合答案提取
        
        流程：
        1. 尝试多种格式提取
        2. 归一化答案
        3. 验证答案
        4. 如果失败，重试不同策略
        
        Args:
            model_output: 模型输出
            task_id: 任务ID
            original_input: 原始输入
            max_retries: 最大重试次数
        
        Returns:
            提取的答案
        """
        if not model_output:
            return None
        
        # 重试机制
        strategies = [
            ComprehensiveAnswerExtractor._extract_with_label_tags,
            ComprehensiveAnswerExtractor._extract_with_json_format,
            ComprehensiveAnswerExtractor._extract_with_finish_format,
            ComprehensiveAnswerExtractor._extract_task_specific,
            ComprehensiveAnswerExtractor._extract_last_paragraph
        ]
        
        for attempt in range(max_retries):
            strategy = strategies[attempt % len(strategies)]
            
            try:
                result = strategy(model_output, task_id)
                
                if result:
                    # 归一化
                    normalized = ComprehensiveAnswerExtractor._normalize(
                        result, task_id
                    )
                    
                    # 验证
                    validation = ComprehensiveAnswerExtractor._validate(
                        normalized, task_id
                    )
                    
                    if validation['is_valid']:
                        logger.info(
                            f"提取成功 (尝试{attempt+1}, 策略{strategy.__name__}): "
                            f"{normalized[:50]}"
                        )
                        return normalized
                    
                    elif validation.get('fixable', False):
                        # 可修复的问题
                        fixed = ComprehensiveAnswerExtractor._fix(
                            normalized, validation['issues'], task_id
                        )
                        if fixed:
                            return fixed
            
            except Exception as e:
                logger.warning(f"策略{strategy.__name__}失败: {e}")
                continue
        
        # 所有策略都失败，返回最后尝试的结果
        logger.warning(f"所有提取策略失败，返回原始输出")
        return model_output.strip()[:500]  # 限制长度防止过长
    
    @staticmethod
    def _extract_with_label_tags(text: str, task_id: int) -> Optional[str]:
        """提取<label>或<answer>标签内容"""
        # 优先<answer>
        match = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # 其次<label>
        match = re.search(r'<label>\s*(.*?)\s*</label>', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        return None
    
    @staticmethod
    def _extract_with_json_format(text: str, task_id: int) -> Optional[str]:
        """提取JSON格式答案"""
        try:
            # 尝试提取JSON对象
            json_match = re.search(r'\{[^}]+\}', text)
            if json_match:
                data = json.loads(json_match.group())
                for key in ['answer', 'prediction', 'result', 'output']:
                    if key in data:
                        return str(data[key])
        except:
            pass
        
        return None
    
    @staticmethod
    def _extract_with_finish_format(text: str, task_id: int) -> Optional[str]:
        """提取finish格式答案"""
        match = re.search(r'finish\(["\']?(.+?)["\']?\)', text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # 尝试"答案:"格式
        match = re.search(r'(?:答案|Answer)[：:]\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        return None
    
    @staticmethod
    def _extract_task_specific(text: str, task_id: int) -> Optional[str]:
        """任务特定提取"""
        
        # 任务2：计数 - 提取数字
        if task_id == 2:
            numbers = re.findall(r'\d+', text)
            if numbers:
                return numbers[-1]
        
        # 任务6：MNLI - 提取Y/N
        elif task_id == 6:
            text_upper = text.upper()
            if 'YES' in text_upper or (' Y ' in text_upper):
                return 'Y'
            elif 'NO' in text_upper or (' N ' in text_upper):
                return 'N'
        
        # 任务8：代码 - 提取代码块
        elif task_id == 8:
            # 尝试提取代码块
            match = re.search(r'```(?:python)?\s*\n(.*?)\n```', text, re.DOTALL)
            if match:
                return match.group(1).strip()
            
            # 尝试提取从import开始的内容
            if 'import torch' in text or 'import triton' in text:
                match = re.search(r'(import (?:torch|triton).*)', text, re.DOTALL)
                if match:
                    return match.group(1).strip()
        
        # 任务7：阅读理解 - 提取最后一个短句
        elif task_id == 7:
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            # 过滤推理过程
            clean_lines = [
                l for l in lines 
                if not any(kw in l for kw in ['因为', '所以', '首先', '然后', '我认为'])
            ]
            if clean_lines:
                # 取最后一个短行
                for line in reversed(clean_lines):
                    if len(line) < 100:
                        return line
        
        return None
    
    @staticmethod
    def _extract_last_paragraph(text: str, task_id: int) -> Optional[str]:
        """提取最后一段（兜底策略）"""
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        return paragraphs[-1] if paragraphs else text.strip()
    
    @staticmethod
    def _normalize(answer: str, task_id: int) -> str:
        """
        答案归一化
        
        参考：Agent实现方式详解
        """
        if not answer:
            return ""
        
        # 移除包裹标记
        answer = re.sub(r'</?(?:label|answer|think(?:ing)?)>', '', answer)
        
        # 移除末尾标点（任务7）
        if task_id == 7:
            answer = answer.rstrip('。，,.!?')
        
        # 数值归一化（任务2、3）
        if task_id in [2, 3]:
            numbers = re.findall(r'[-+]?\d+', answer)
            if numbers:
                return numbers[-1]
        
        # Y/N归一化（任务6）
        if task_id == 6:
            answer_upper = answer.strip().upper()
            if answer_upper in ['YES', 'Y']:
                return 'Y'
            elif answer_upper in ['NO', 'N']:
                return 'N'
        
        # 清理多余空格
        answer = ' '.join(answer.split())
        
        return answer.strip()
    
    @staticmethod
    def _validate(answer: str, task_id: int) -> Dict[str, Any]:
        """
        答案验证
        
        返回：
        - is_valid: 是否有效
        - issues: 问题列表
        - fixable: 是否可修复
        """
        issues = []
        fixable = False
        
        # 空答案
        if not answer or len(answer.strip()) == 0:
            return {'is_valid': False, 'issues': ['空答案'], 'fixable': False}
        
        # 获取任务配置
        config = ComprehensiveAnswerExtractor.TASK_CONFIGS.get(task_id, {})
        
        # 过长答案
        max_length = config.get('max_length', 500)
        if len(answer) > max_length:
            issues.append(f'答案过长(>{max_length})')
            fixable = True
        
        # 包含推理过程
        if any(kw in answer.lower() for kw in ['thought', 'action', 'search', '推理', '搜索']):
            issues.append('答案包含推理过程')
            fixable = True
        
        # 任务特定验证
        if 'validate' in config and not config['validate'](answer):
            issues.append('格式不符合要求')
            fixable = True
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'fixable': fixable
        }
    
    @staticmethod
    def _fix(answer: str, issues: List[str], task_id: int) -> Optional[str]:
        """
        修复答案
        """
        fixed = answer
        
        for issue in issues:
            if '推理过程' in issue:
                # 移除推理关键词
                for kw in ['因为', '所以', '首先', '然后', '我认为', '因此']:
                    fixed = re.sub(rf'{kw}[^。！？.!?]*[。！？.!?]', '', fixed)
            
            elif '过长' in issue:
                # 截断
                config = ComprehensiveAnswerExtractor.TASK_CONFIGS.get(task_id, {})
                max_len = config.get('max_length', 500)
                fixed = fixed[:max_len]
        
        return fixed.strip() if fixed.strip() else None


# 便捷接口
def extract_answer_comprehensive(
    model_output: str,
    task_id: int,
    original_input: str = ''
) -> Optional[str]:
    """
    综合答案提取接口
    """
    return ComprehensiveAnswerExtractor.extract(
        model_output,
        task_id,
        original_input
    )


if __name__ == "__main__":
    # 测试
    test_cases = [
        {
            'task_id': 2,
            'output': '经过分析，这个句子包含3个名词。<label>3</label>',
            'expected': '3'
        },
        {
            'task_id': 6,
            'output': '根据推理，答案应该是YES。',
            'expected': 'Y'
        },
        {
            'task_id': 7,
            'output': '因为法国的首都是巴黎，所以答案是Paris。',
            'expected': 'Paris'
        },
    ]
    
    print("="*70)
    print("综合答案提取系统测试")
    print("="*70)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n测试 {i}: 任务{test['task_id']}")
        print(f"输入: {test['output'][:50]}...")
        print(f"期望: {test['expected']}")
        
        result = extract_answer_comprehensive(
            test['output'],
            test['task_id']
        )
        
        print(f"输出: {result}")
        print(f"状态: {'✅' if result == test['expected'] else '❌'}")