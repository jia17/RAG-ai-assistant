"""
日志配置
"""

import logging
import os
import sys
from typing import Optional
from src.config import LOG_LEVEL

# 配置日志格式
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

def setup_logger(level: Optional[str] = None) -> logging.Logger:
    """设置根日志器
    
    Args:
        level: 日志级别，如果为None则使用配置中的LOG_LEVEL
        
    Returns:
        配置好的根日志器
    """
    log_level = level or LOG_LEVEL
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    # 配置根日志器
    logging.basicConfig(
        level=numeric_level,
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # 设置第三方库的日志级别
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("pymilvus").setLevel(logging.WARNING)
    
    return logging.getLogger()

def get_logger(name: str) -> logging.Logger:
    """获取命名日志器
    
    Args:
        name: 日志器名称，通常为__name__
        
    Returns:
        配置好的日志器
    """    

    return logging.getLogger(name)
