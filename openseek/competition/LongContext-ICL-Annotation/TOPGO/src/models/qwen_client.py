"""
Qwen模型客户端
"""
import json
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import requests
from loguru import logger


@dataclass
class ModelResponse:
    """模型响应"""
    text: str
    usage: Dict[str, int]
    finish_reason: str
    raw_response: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "usage": self.usage,
            "finish_reason": self.finish_reason
        }


class QwenClient:
    """Qwen模型客户端"""
    
    def __init__(
        self,
        api_base: str,
        api_key: str = "",
        model_name: str = "Qwen3-4B",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        top_p: float = 0.9,
        timeout: int = 300,
        max_retries: int = 3
    ):
        """
        初始化Qwen客户端
        
        Args:
            api_base: API基础URL
            api_key: API密钥
            model_name: 模型名称
            temperature: 温度参数
            max_tokens: 最大token数
            top_p: Top-p采样参数
            timeout: 超时时间（秒）
            max_retries: 最大重试次数
        """
        self.api_base = api_base.rstrip('/')
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.timeout = timeout
        self.max_retries = max_retries
        
        self.headers = {
            "Content-Type": "application/json"
        }
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        
        logger.info(f"Qwen客户端初始化: {model_name}, API: {api_base}")
    
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None
    ) -> str:
        """
        生成文本
        
        Args:
            prompt: 提示词
            temperature: 温度
            max_tokens: 最大token数
            stop: 停止词列表
            
        Returns:
            生成的文本
        """
        response = self._call_api(
            prompt=prompt,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            stop=stop
        )
        return response.text
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None
    ) -> str:
        """
        对话式生成
        
        Args:
            messages: 消息列表
            temperature: 温度
            max_tokens: 最大token数
            stop: 停止词列表
            
        Returns:
            生成的文本
        """
        response = self._call_api(
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            stop=stop
        )
        return response.text
    
    def _call_api(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None
    ) -> ModelResponse:
        """
        调用API
        
        Args:
            prompt: 提示词
            messages: 消息列表
            temperature: 温度
            max_tokens: 最大token数
            stop: 停止词列表
            
        Returns:
            模型响应
        """
        # 构建请求体
        payload = {
            "model": self.model_name,
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "top_p": self.top_p
        }
        
        if prompt:
            payload["prompt"] = prompt
        if messages:
            payload["messages"] = messages
        if stop:
            payload["stop"] = stop
        
        # 重试机制
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.api_base}/v1/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # 解析响应
                    if "choices" in data and len(data["choices"]) > 0:
                        choice = data["choices"][0]
                        text = choice.get("text", choice.get("message", {}).get("content", ""))
                        
                        return ModelResponse(
                            text=text,
                            usage=data.get("usage", {}),
                            finish_reason=choice.get("finish_reason", "stop"),
                            raw_response=data
                        )
                    else:
                        raise ValueError(f"API响应格式错误: {data}")
                
                elif response.status_code == 429:
                    # 速率限制，等待后重试
                    wait_time = 2 ** attempt
                    logger.warning(f"速率限制，等待 {wait_time} 秒后重试")
                    time.sleep(wait_time)
                    continue
                
                else:
                    raise ValueError(f"API调用失败: {response.status_code} - {response.text}")
                    
            except requests.exceptions.Timeout:
                last_error = "请求超时"
                logger.warning(f"请求超时，尝试 {attempt + 1}/{self.max_retries}")
                time.sleep(2 ** attempt)
                
            except requests.exceptions.RequestException as e:
                last_error = str(e)
                logger.error(f"请求异常: {e}")
                time.sleep(2 ** attempt)
        
        raise RuntimeError(f"API调用失败，已重试 {self.max_retries} 次: {last_error}")
    
    def generate_json(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        生成JSON格式输出
        
        Args:
            prompt: 提示词
            temperature: 温度
            max_tokens: 最大token数
            
        Returns:
            JSON字典
        """
        # 添加JSON格式提示
        json_prompt = f"{prompt}\n\n请以JSON格式输出结果。"
        
        response_text = self.generate(
            prompt=json_prompt,
            temperature=temperature or 0.0,  # JSON输出使用较低温度
            max_tokens=max_tokens
        )
        
        # 解析JSON
        try:
            # 尝试提取JSON部分
            json_str = self._extract_json(response_text)
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}\n原始响应: {response_text}")
            return {"error": "JSON解析失败", "raw_response": response_text}
    
    def _extract_json(self, text: str) -> str:
        """
        从文本中提取JSON
        
        Args:
            text: 包含JSON的文本
            
        Returns:
            JSON字符串
        """
        # 尝试找到JSON块
        start_markers = ['{', '[']
        end_markers = ['}', ']']
        
        start_idx = -1
        end_idx = -1
        
        for marker in start_markers:
            idx = text.find(marker)
            if idx >= 0:
                start_idx = idx
                break
        
        if start_idx < 0:
            return text
        
        # 找到匹配的结束标记
        depth = 0
        for i, char in enumerate(text[start_idx:]):
            if char in start_markers:
                depth += 1
            elif char in end_markers:
                depth -= 1
                if depth == 0:
                    end_idx = start_idx + i + 1
                    break
        
        if end_idx > start_idx:
            return text[start_idx:end_idx]
        
        return text
    
    def batch_generate(
        self,
        prompts: List[str],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> List[str]:
        """
        批量生成
        
        Args:
            prompts: 提示词列表
            temperature: 温度
            max_tokens: 最大token数
            
        Returns:
            生成结果列表
        """
        results = []
        for prompt in prompts:
            result = self.generate(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            results.append(result)
        return results


class MockQwenClient(QwenClient):
    """模拟客户端（用于测试）"""
    
    def __init__(self, **kwargs):
        super().__init__(
            api_base="http://mock",
            api_key="mock",
            **kwargs
        )
        logger.info("使用模拟客户端")
    
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None
    ) -> str:
        """模拟生成"""
        # 返回简单的JSON格式响应
        return json.dumps({
            "entities": [
                {"text": "示例实体", "type": "方法"}
            ],
            "relations": []
        })
    
    def generate_json(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """模拟JSON生成"""
        return {
            "entities": [
                {"text": "示例实体", "type": "方法"}
            ],
            "relations": []
        }
