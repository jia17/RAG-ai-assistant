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

# Milvus连接重试配置
MILVUS_MAX_RETRIES = int(os.environ.get("MILVUS_MAX_RETRIES", DEFAULT_SETTINGS["milvus_max_retries"]))
MILVUS_INITIAL_DELAY = float(os.environ.get("MILVUS_INITIAL_DELAY", DEFAULT_SETTINGS["milvus_initial_delay"]))
MILVUS_MAX_DELAY = float(os.environ.get("MILVUS_MAX_DELAY", DEFAULT_SETTINGS["milvus_max_delay"]))
MILVUS_BACKOFF_FACTOR = float(os.environ.get("MILVUS_BACKOFF_FACTOR", DEFAULT_SETTINGS["milvus_backoff_factor"]))
MILVUS_CONNECTION_TIMEOUT = float(os.environ.get("MILVUS_CONNECTION_TIMEOUT", DEFAULT_SETTINGS["milvus_connection_timeout"]))
MILVUS_HEALTH_CHECK_INTERVAL = int(os.environ.get("MILVUS_HEALTH_CHECK_INTERVAL", DEFAULT_SETTINGS["milvus_health_check_interval"]))

# Milvus搜索优化配置
MILVUS_DEFAULT_METRIC_TYPE = os.environ.get("MILVUS_DEFAULT_METRIC_TYPE", DEFAULT_SETTINGS["milvus_default_metric_type"])
MILVUS_CONSISTENCY_LEVEL = os.environ.get("MILVUS_CONSISTENCY_LEVEL", DEFAULT_SETTINGS["milvus_consistency_level"])
MILVUS_SEARCH_TIMEOUT = int(os.environ.get("MILVUS_SEARCH_TIMEOUT", DEFAULT_SETTINGS["milvus_search_timeout"]))
MILVUS_SMALL_DATASET_THRESHOLD = int(os.environ.get("MILVUS_SMALL_DATASET_THRESHOLD", DEFAULT_SETTINGS["milvus_small_dataset_threshold"]))
MILVUS_MEDIUM_DATASET_THRESHOLD = int(os.environ.get("MILVUS_MEDIUM_DATASET_THRESHOLD", DEFAULT_SETTINGS["milvus_medium_dataset_threshold"]))
MILVUS_ENABLE_SEARCH_OPTIMIZATION = os.environ.get("MILVUS_ENABLE_SEARCH_OPTIMIZATION", str(DEFAULT_SETTINGS["milvus_enable_search_optimization"])).lower() == "true"
MILVUS_AUTO_INDEX_OPTIMIZATION = os.environ.get("MILVUS_AUTO_INDEX_OPTIMIZATION", str(DEFAULT_SETTINGS["milvus_auto_index_optimization"])).lower() == "true"

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
    "MILVUS_MAX_RETRIES",
    "MILVUS_INITIAL_DELAY", 
    "MILVUS_MAX_DELAY",
    "MILVUS_BACKOFF_FACTOR",
    "MILVUS_CONNECTION_TIMEOUT",
    "MILVUS_HEALTH_CHECK_INTERVAL",
    "MILVUS_DEFAULT_METRIC_TYPE",
    "MILVUS_CONSISTENCY_LEVEL",
    "MILVUS_SEARCH_TIMEOUT",
    "MILVUS_SMALL_DATASET_THRESHOLD",
    "MILVUS_MEDIUM_DATASET_THRESHOLD",
    "MILVUS_ENABLE_SEARCH_OPTIMIZATION",
    "MILVUS_AUTO_INDEX_OPTIMIZATION",
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
