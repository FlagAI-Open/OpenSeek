"""
配置加载工具
"""
import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径，默认为configs/config.yaml
        
    Returns:
        配置字典
    """
    if config_path is None:
        # 默认配置文件路径
        base_dir = Path(__file__).parent.parent.parent
        config_path = base_dir / "configs" / "config.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 从环境变量覆盖敏感配置
    if os.getenv("QWEN_API_BASE"):
        config["model"]["api_base"] = os.getenv("QWEN_API_BASE")
    if os.getenv("QWEN_API_KEY"):
        config["model"]["api_key"] = os.getenv("QWEN_API_KEY")
    
    return config


def merge_config(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并配置（深度合并）
    
    Args:
        base_config: 基础配置
        override_config: 覆盖配置
        
    Returns:
        合并后的配置
    """
    result = base_config.copy()
    
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = value
    
    return result


def validate_config(config: Dict[str, Any]) -> bool:
    """
    验证配置是否完整
    
    Args:
        config: 配置字典
        
    Returns:
        是否有效
    """
    required_keys = ["model", "context", "retrieval", "validation", "output"]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"配置缺少必要字段: {key}")
    
    # 验证模型配置
    if not config["model"].get("name"):
        raise ValueError("模型名称不能为空")
    
    return True


def get_absolute_path(relative_path: str, base_dir: Optional[Path] = None) -> Path:
    """
    获取绝对路径
    
    Args:
        relative_path: 相对路径
        base_dir: 基础目录
        
    Returns:
        绝对路径
    """
    if base_dir is None:
        base_dir = Path(__file__).parent.parent.parent
    
    path = Path(relative_path)
    if path.is_absolute():
        return path
    return base_dir / path
