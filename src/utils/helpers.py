"""
辅助函数
"""

import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import hashlib

def generate_document_id(content: str, source: Optional[str] = None) -> str:
    """为文档内容生成唯一ID
    
    Args:
        content: 文档内容
        source: 可选的源信息
        
    Returns:
        唯一ID字符串
    """
    text_for_hash = f"{content}{source or ''}{datetime.now().isoformat()}"
    return hashlib.md5(text_for_hash.encode()).hexdigest()

def ensure_directory(directory_path: str) -> str:
    """确保目录存在，如果不存在则创建
    
    Args:
        directory_path: 目录路径
        
    Returns:
        目录路径
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    return directory_path

def save_json(data: Any, file_path: str) -> None:
    """将数据保存为JSON文件
    
    Args:
        data: 要保存的数据
        file_path: 文件路径
    """
    ensure_directory(os.path.dirname(file_path))
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(file_path: str) -> Any:
    """从JSON文件加载数据
    
    Args:
        file_path: 文件路径
        
    Returns:
        加载的数据
    """
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
