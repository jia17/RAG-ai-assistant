"""
模型服务模块
"""

from .llm_service import LLMService
from .embedding_service import EmbeddingService
from .vector_store_service import VectorStoreService
from src.config import ANTHROPIC_API_KEY, OPENAI_API_KEY, LLM_MODEL, EMBEDDING_MODEL, QWEN_API_KEY, GLM_API_KEY

# 创建默认服务实例
#TODO:后面需要根据具体的应用场景进行调整，
#不同的场景使用不同的LLM，比如生成答案和答案评估LLM侧重点不同
default_llm = LLMService(
    model_name=LLM_MODEL,
    anthropic_api_key=ANTHROPIC_API_KEY,
    openai_api_key=OPENAI_API_KEY,
    qwen_api_key=QWEN_API_KEY,
    glm_api_key=GLM_API_KEY,
)
default_embedding = EmbeddingService(model_name=EMBEDDING_MODEL)

__all__ = [
    'LLMService', 
    'EmbeddingService', 
    'VectorStoreService',
    'default_llm',
    'default_embedding'
]
