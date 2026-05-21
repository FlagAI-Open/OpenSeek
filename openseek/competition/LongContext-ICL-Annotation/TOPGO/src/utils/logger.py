"""
日志工具
"""
import os
import sys
from pathlib import Path
from typing import Optional
from loguru import logger


def setup_logger(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
):
    """
    设置日志系统
    
    Args:
        log_level: 日志级别
        log_file: 日志文件路径
        log_format: 日志格式
    """
    # 移除默认处理器
    logger.remove()
    
    # 默认格式
    if log_format is None:
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    
    # 控制台输出
    logger.add(
        sys.stdout,
        format=log_format,
        level=log_level,
        colorize=True,
        enqueue=True
    )
    
    # 文件输出
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level=log_level,
            rotation="10 MB",
            retention="7 days",
            compression="zip",
            enqueue=True,
            encoding="utf-8"
        )
    
    return logger


def get_logger(name: str):
    """
    获取命名日志器
    
    Args:
        name: 日志器名称
        
    Returns:
        日志器实例
    """
    return logger.bind(name=name)
