#!/usr/bin/env python3
"""
Agent最佳实践工具模块

集成以下最佳实践：
1. 重试机制 (Retry with exponential backoff)
2. 搜索缓存 (Search/Retrieval caching)
3. 性能监控 (Performance monitoring)
4. 答案提取与归一化 (Answer extraction & normalization)
5. 错误处理 (Error handling)
"""

import re
import json
import time
import hashlib
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple
from functools import wraps
from dataclasses import dataclass, field
from collections import defaultdict
from loguru import logger


# ============== 1. 重试机制 ==============

def retry_with_exponential_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    multiplier: float = 2.0,
    exceptions: Tuple[type, ...] = (Exception,)
):
    """
    指数退避重试装饰器
    
    Args:
        max_attempts: 最大尝试次数
        base_delay: 基础延迟（秒）
        max_delay: 最大延迟（秒）
        multiplier: 延迟倍增因子
        exceptions: 需要重试的异常类型
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts:
                        logger.error(f"达到最大重试次数 {max_attempts}, 放弃执行")
                        raise
                    
                    delay = min(base_delay * (multiplier ** (attempt - 1)), max_delay)
                    logger.warning(
                        f"尝试 {attempt}/{max_attempts} 失败: {e}, "
                        f"{delay:.1f}秒后重试..."
                    )
                    time.sleep(delay)
            
            raise last_exception
        
        return wrapper
    return decorator


def retry_with_fallback(
    fallback_value: Any = None,
    max_attempts: int = 3,
    exceptions: Tuple[type, ...] = (Exception,)
):
    """
    带回退值的重试装饰器
    
    Args:
        fallback_value: 失败时返回的默认值
        max_attempts: 最大尝试次数
        exceptions: 需要重试的异常类型
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    logger.warning(f"尝试 {attempt}/{max_attempts} 失败: {e}")
            
            logger.error(f"所有重试失败，返回默认值: {fallback_value}")
            return fallback_value
        
        return wrapper
    return decorator


# ============== 2. 缓存机制 ==============

class CacheEntry:
    """缓存条目"""
    def __init__(self, value: Any, ttl: float = 3600):
        self.value = value
        self.timestamp = time.time()
        self.ttl = ttl
    
    def is_expired(self) -> bool:
        return time.time() - self.timestamp > self.ttl


class SearchCache:
    """
    检索结果缓存
    
    使用LRU淘汰策略，支持TTL过期
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: float = 3600):
        """
        初始化缓存
        
        Args:
            max_size: 最大缓存条目数
            default_ttl: 默认过期时间（秒）
        """
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.hits = 0
        self.misses = 0
    
    def _make_key(self, query: str) -> str:
        """生成缓存键"""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    def get(self, query: str) -> Optional[Any]:
        """获取缓存值"""
        key = self._make_key(query)
        
        if key in self.cache:
            entry = self.cache[key]
            
            if entry.is_expired():
                del self.cache[key]
                self.misses += 1
                return None
            
            self.hits += 1
            logger.debug(f"缓存命中: {query[:50]}...")
            return entry.value
        
        self.misses += 1
        return None
    
    def set(self, query: str, value: Any, ttl: Optional[float] = None):
        """设置缓存值"""
        key = self._make_key(query)
        
        # LRU淘汰
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest_key = min(
                self.cache.keys(),
                key=lambda k: self.cache[k].timestamp
            )
            del self.cache[oldest_key]
            logger.debug(f"缓存淘汰: {oldest_key[:20]}...")
        
        self.cache[key] = CacheEntry(value, ttl or self.default_ttl)
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate
        }


# ============== 3. 性能监控 ==============

class PerformanceMonitor:
    """
    性能监控器
    
    记录和报告各环节的性能指标
    """
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.counters: Dict[str, int] = defaultdict(int)
        self.start_times: Dict[str, float] = {}
    
    def record(self, metric_name: str, value: float):
        """记录指标值"""
        self.metrics[metric_name].append(value)
    
    def increment(self, counter_name: str, delta: int = 1):
        """增加计数器"""
        self.counters[counter_name] += delta
    
    def start_timer(self, timer_name: str):
        """启动计时器"""
        self.start_times[timer_name] = time.time()
    
    def stop_timer(self, timer_name: str) -> float:
        """停止计时器并记录"""
        if timer_name not in self.start_times:
            logger.warning(f"计时器 {timer_name} 未启动")
            return 0.0
        
        elapsed = time.time() - self.start_times[timer_name]
        self.record(timer_name, elapsed)
        del self.start_times[timer_name]
        return elapsed
    
    def timer(self, metric_name: str) -> Callable:
        """计时器装饰器"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                start = time.time()
                try:
                    return func(*args, **kwargs)
                finally:
                    elapsed = time.time() - start
                    self.record(metric_name, elapsed)
            return wrapper
        return decorator
    
    def report(self) -> Dict[str, Any]:
        """生成性能报告"""
        report = {
            "metrics": {},
            "counters": dict(self.counters)
        }
        
        for name, values in self.metrics.items():
            if values:
                report["metrics"][name] = {
                    "count": len(values),
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "total": sum(values)
                }
        
        return report
    
    def reset(self):
        """重置所有指标"""
        self.metrics.clear()
        self.counters.clear()
        self.start_times.clear()


# ============== 4. 答案提取与归一化 ==============

class AnswerExtractor:
    """
    答案提取器
    
    从LLM输出中提取答案，支持多种格式
    """
    
    @staticmethod
    def extract_finish(output: str) -> Optional[str]:
        """提取 finish() 格式的答案"""
        match = re.search(r'finish\(["\']?(.+?)["\']?\)', output, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
    
    @staticmethod
    def extract_json_answer(output: str) -> Optional[Dict[str, Any]]:
        """从JSON中提取答案"""
        try:
            # 尝试直接解析
            data = json.loads(output)
            if isinstance(data, dict):
                for key in ['answer', 'prediction', 'result', 'output']:
                    if key in data:
                        return data[key]
        except json.JSONDecodeError:
            pass
        
        # 尝试提取JSON块
        json_start = output.find('{')
        if json_start >= 0:
            depth = 0
            json_end = json_start
            for i, char in enumerate(output[json_start:]):
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        json_end = json_start + i + 1
                        break
            
            try:
                data = json.loads(output[json_start:json_end])
                return data
            except json.JSONDecodeError:
                pass
        
        return None
    
    @staticmethod
    def extract_label_tag(output: str) -> Optional[str]:
        """提取 <label> 或 <answer> 标签内容"""
        patterns = [
            r'<answer>\s*(.*?)\s*</answer>',
            r'<label>\s*(.*?)\s*</label>'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, output, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        return None
    
    @staticmethod
    def extract_answer_colon(output: str) -> Optional[str]:
        """提取 '答案:' 或 'Answer:' 格式"""
        patterns = [
            r'(?:答案|Answer)[:：]\s*(.+?)(?:\n|$)',
            r'^(.+?)$'  # 最后一行作为兜底
        ]
        
        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()
        
        return output.strip()
    
    @staticmethod
    def extract(output: str) -> str:
        """
        综合答案提取
        
        按优先级尝试各种提取方式
        """
        # 1. 尝试 finish() 格式
        result = AnswerExtractor.extract_finish(output)
        if result:
            return result
        
        # 2. 尝试 <answer> 或 <label> 标签
        result = AnswerExtractor.extract_label_tag(output)
        if result:
            return result
        
        # 3. 尝试 JSON 格式
        json_data = AnswerExtractor.extract_json_answer(output)
        if json_data:
            if isinstance(json_data, dict):
                for key in ['answer', 'prediction', 'result']:
                    if key in json_data:
                        return str(json_data[key])
            return str(json_data)
        
        # 4. 尝试 "答案:" 格式
        result = AnswerExtractor.extract_answer_colon(output)
        if result:
            return result
        
        # 5. 兜底：返回最后一行
        paragraphs = [p.strip() for p in output.split('\n') if p.strip()]
        return paragraphs[-1] if paragraphs else output.strip()


class AnswerNormalizer:
    """
    答案归一化器
    
    与评测脚本保持一致的归一化处理
    """
    
    @staticmethod
    def normalize(answer: str) -> str:
        """
        归一化答案
        
        处理步骤：
        1. 转小写
        2. 去除首尾空格
        3. 数值处理
        4. 多实体格式处理
        5. 去除多余空格
        """
        if not answer:
            return ""
        
        # 转小写
        answer = str(answer).lower()
        
        # 去除首尾空格
        answer = answer.strip()
        
        # 处理数值（转为整数）
        number_match = re.search(r'[-+]?[\d,]+\.?\d*', answer)
        if number_match:
            num_str = number_match.group().replace(',', '')
            try:
                num = float(num_str)
                if num == int(num):
                    return str(int(num))
                return str(num)
            except ValueError:
                pass
        
        # 多实体格式处理（逗号/分号后接空格）
        answer = re.sub(r',\s*', ', ', answer)
        answer = re.sub(r';\s*', '; ', answer)
        
        # 去除多余空格
        answer = re.sub(r'\s+', ' ', answer)
        
        return answer
    
    @staticmethod
    def normalize_list(answer: str) -> str:
        """归一化列表格式答案"""
        # 移除列表括号
        answer = answer.strip()
        if answer.startswith('[') and answer.endswith(']'):
            answer = answer[1:-1]
        
        # 分割并归一化每个元素
        items = re.split(r'[,;]', answer)
        items = [AnswerNormalizer.normalize(item) for item in items if item.strip()]
        
        return ', '.join(items)


class AnswerValidator:
    """
    答案验证器
    
    验证答案的合理性
    """
    
    def __init__(self):
        self.validation_rules = []
    
    def add_rule(self, rule: Callable[[str], Tuple[bool, str]]):
        """
        添加验证规则
        
        Args:
            rule: 规则函数，返回 (是否通过, 错误信息)
        """
        self.validation_rules.append(rule)
    
    def validate(self, answer: str, question: str = "") -> Dict[str, Any]:
        """
        验证答案
        
        Returns:
            包含 is_valid 和 issues 的字典
        """
        issues = []
        
        # 空答案检查
        if not answer or len(answer.strip()) == 0:
            issues.append("空答案")
        
        # 过长答案检查
        if len(answer) > 500:
            issues.append("答案过长")
        
        # 包含推理过程检查
        reasoning_keywords = ['thought', 'action', 'search', '推理', '搜索', '思考']
        if any(kw in answer.lower() for kw in reasoning_keywords):
            issues.append("答案包含推理过程")
        
        # 数值范围检查
        if re.match(r'^\d+$', answer):
            num = int(answer)
            if num > 1000000 or num < -1000000:
                issues.append("数值异常")
        
        # 应用自定义规则
        for rule in self.validation_rules:
            try:
                passed, message = rule(answer)
                if not passed:
                    issues.append(message)
            except Exception as e:
                logger.warning(f"验证规则执行失败: {e}")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues
        }


# ============== 5. 错误处理 ==============

class ErrorHandler:
    """
    错误处理器
    
    分类和处理不同类型的错误
    """
    
    ERROR_TYPES = {
        'timeout': ['timeout', 'timed out', '超时'],
        'network': ['connection', 'network', '网络', '连接'],
        'model': ['model', 'llm', 'api'],
        'parse': ['parse', 'json', 'decode', '解析'],
        'empty': ['empty', 'none', 'null', '空']
    }
    
    @classmethod
    def classify_error(cls, error: Exception or str) -> str:
        """分类错误类型"""
        error_str = str(error).lower()
        
        for error_type, keywords in cls.ERROR_TYPES.items():
            if any(kw in error_str for kw in keywords):
                return error_type
        
        return 'unknown'
    
    @classmethod
    def get_recovery_action(cls, error_type: str) -> str:
        """获取恢复动作建议"""
        actions = {
            'timeout': '增加超时时间或简化查询',
            'network': '检查网络连接或使用备用API',
            'model': '切换模型或调整参数',
            'parse': '增强答案解析逻辑',
            'empty': '检查输入或使用默认值',
            'unknown': '记录日志并使用兜底方案'
        }
        return actions.get(error_type, actions['unknown'])


# ============== 6. 日志记录 ==============

class ExecutionLogger:
    """
    执行日志记录器
    
    记录Agent执行轨迹，用于调试和分析
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.execution_trace: List[Dict[str, Any]] = []
    
    def log_step(self, step_type: str, content: Any, metadata: Optional[Dict] = None):
        """记录执行步骤"""
        step = {
            'timestamp': time.time(),
            'type': step_type,
            'content': content,
            'metadata': metadata or {}
        }
        self.execution_trace.append(step)
        
        if self.verbose:
            if isinstance(content, str) and len(content) > 200:
                content = content[:200] + "..."
            logger.info(f"[{step_type}] {content}")
    
    def save_trace(self, file_path: str):
        """保存执行轨迹"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.execution_trace, f, ensure_ascii=False, indent=2)
        logger.info(f"执行轨迹已保存到: {file_path}")
    
    def get_trace(self) -> List[Dict[str, Any]]:
        """获取执行轨迹"""
        return self.execution_trace
    
    def clear(self):
        """清空轨迹"""
        self.execution_trace.clear()


# ============== 便捷函数 ==============

def safe_execute(func: Callable, args: tuple = (), kwargs: dict = None, 
                 fallback: Any = None, timeout_sec: float = 30) -> Any:
    """
    安全执行函数，带超时控制
    
    Args:
        func: 要执行的函数
        args: 位置参数
        kwargs: 关键字参数
        fallback: 失败时的返回值
        timeout_sec: 超时时间（秒）
    
    Returns:
        函数返回值或兜底值
    """
    import signal
    
    kwargs = kwargs or {}
    
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {timeout_sec} seconds")
    
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(timeout_sec))
    
    try:
        result = func(*args, **kwargs)
        return result
    except TimeoutError:
        logger.warning(f"函数执行超时: {func.__name__}")
        return fallback
    except Exception as e:
        logger.error(f"函数执行失败: {e}")
        return fallback
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# 全局性能监控器实例
global_monitor = PerformanceMonitor()