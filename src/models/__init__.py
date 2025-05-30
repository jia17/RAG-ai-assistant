"""
模型服务层 - 封装各种模型和服务的调用接口
"""

from .llm_service import LLMService
from .embedding_service import EmbeddingService
from .vector_store_service import VectorStoreService
from src.config import ANTHROPIC_API_KEY, OPENAI_API_KEY, LLM_MODEL, EMBEDDING_MODEL, QWEN_API_KEY, GLM_API_KEY

# 创建默认实例
try:
    default_llm = LLMService(
        model_name=LLM_MODEL,
        anthropic_api_key=ANTHROPIC_API_KEY,
        openai_api_key=OPENAI_API_KEY,
        qwen_api_key=QWEN_API_KEY,
        glm_api_key=GLM_API_KEY,
    )
except Exception as e:
    print(f"警告: 无法初始化默认LLM服务: {e}")
    default_llm = None

try:
    default_embedding = EmbeddingService(model_name=EMBEDDING_MODEL)
except Exception as e:
    print(f"警告: 无法初始化默认嵌入服务: {e}")
    default_embedding = None

try:
    default_vector_store = VectorStoreService()
except Exception as e:
    print(f"警告: 无法初始化默认向量存储服务: {e}")
    default_vector_store = None

__all__ = [
    'LLMService',
    'EmbeddingService', 
    'VectorStoreService',
    'default_llm',
    'default_embedding',
    'default_vector_store'
]
