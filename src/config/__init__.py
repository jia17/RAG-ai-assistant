# """
# 配置管理模块
# """

import os
from .settings import DEFAULT_SETTINGS

# 从环境变量加载配置，如果不存在则使用默认值
LLM_MODEL = os.environ.get("LLM_MODEL", DEFAULT_SETTINGS["llm_model"])
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", DEFAULT_SETTINGS["embedding_model"])
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", DEFAULT_SETTINGS.get("anthropic_api_key", ""))
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", DEFAULT_SETTINGS.get("openai_api_key", ""))
QWEN_API_KEY = os.environ.get("QWEN_API_KEY", DEFAULT_SETTINGS.get("qwen_api_key", ""))
GLM_API_KEY = os.environ.get("GLM_API_KEY", DEFAULT_SETTINGS.get("glm_api_key", ""))
MILVUS_HOST = os.environ.get("MILVUS_HOST", DEFAULT_SETTINGS["milvus_host"])
MILVUS_PORT = int(os.environ.get("MILVUS_PORT", DEFAULT_SETTINGS["milvus_port"]))
MILVUS_COLLECTION = os.environ.get("MILVUS_COLLECTION", DEFAULT_SETTINGS["milvus_collection"])
MAX_CONTEXT_CHUNKS = int(os.environ.get("MAX_CONTEXT_CHUNKS", DEFAULT_SETTINGS["max_context_chunks"]))
TEMPERATURE = float(os.environ.get("TEMPERATURE", DEFAULT_SETTINGS["temperature"]))
LOG_LEVEL = os.environ.get("LOG_LEVEL", DEFAULT_SETTINGS["log_level"])

MAX_ITERATIONS = 7

__all__ = [
    "LLM_MODEL", 
    "EMBEDDING_MODEL", 
    "ANTHROPIC_API_KEY", 
    "OPENAI_API_KEY",
    "QWEN_API_KEY",
    "GLM_API_KEY",
    "MILVUS_HOST", 
    "MILVUS_PORT", 
    "MILVUS_COLLECTION",
    "MAX_CONTEXT_CHUNKS",
    "TEMPERATURE",
    "LOG_LEVEL",
    "DEFAULT_SETTINGS"
]
