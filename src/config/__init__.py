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

LANGCHAIN_TRACING_V2 = os.environ.get("LANGCHAIN_TRACING_V2", "false").lower() == "true"
LANGCHAIN_ENDPOINT = os.environ.get("LANGCHAIN_ENDPOINT", "http://localhost:8000")
LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY", "your_langchain_api_key")

# 数据库配置
DATABASE_URL = os.environ.get("DATABASE_URL", DEFAULT_SETTINGS["database_url"])
DATABASE_POOL_SIZE = int(os.environ.get("DATABASE_POOL_SIZE", DEFAULT_SETTINGS["database_pool_size"]))
DATABASE_MAX_OVERFLOW = int(os.environ.get("DATABASE_MAX_OVERFLOW", DEFAULT_SETTINGS["database_max_overflow"]))
DATABASE_POOL_TIMEOUT = int(os.environ.get("DATABASE_POOL_TIMEOUT", DEFAULT_SETTINGS["database_pool_timeout"]))
DATABASE_POOL_RECYCLE = int(os.environ.get("DATABASE_POOL_RECYCLE", DEFAULT_SETTINGS["database_pool_recycle"]))
DATABASE_ECHO = os.environ.get("DATABASE_ECHO", str(DEFAULT_SETTINGS["database_echo"])).lower() == "true"

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
    "DEFAULT_SETTINGS",
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_ENDPOINT",
    "LANGCHAIN_API_KEY",
    "DATABASE_URL",
    "DATABASE_POOL_SIZE",
    "DATABASE_MAX_OVERFLOW",
    "DATABASE_POOL_TIMEOUT",
    "DATABASE_POOL_RECYCLE",
    "DATABASE_ECHO"
]
