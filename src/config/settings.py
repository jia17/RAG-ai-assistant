"""
默认设置和配置
"""

# 默认配置值
DEFAULT_SETTINGS = {
    # 模型配置
    "llm_model": "qwen-plus", #"claude-3-sonnet-20240229",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",

    "openai_api_key": "",
    "anthropic_api_key": "",
    "qwen_api_key": "your_QWEN_API_KEY",
    "glm_api_key": "",
    
    # Milvus配置
    "milvus_host": "localhost",
    "milvus_port": 19530,
    "milvus_collection": "kubesphere_docs",
    
    # Milvus连接重试配置
    "milvus_max_retries": 5,
    "milvus_initial_delay": 1.0,
    "milvus_max_delay": 60.0,
    "milvus_backoff_factor": 2.0,
    "milvus_connection_timeout": 10.0,
    "milvus_health_check_interval": 30,
    
    # Milvus搜索优化配置
    "milvus_default_metric_type": "COSINE",
    "milvus_consistency_level": "Strong",
    "milvus_search_timeout": 30,
    "milvus_small_dataset_threshold": 10000,
    "milvus_medium_dataset_threshold": 100000,
    "milvus_enable_search_optimization": True,
    "milvus_auto_index_optimization": True,
    
    # RAG配置
    "max_context_chunks": 10,
    "temperature": 0.2,
    "top_k": 5,
    "similarity_threshold": 0.7,
    
    # 日志配置
    "log_level": "INFO",
    
    # 文档处理配置
    "chunk_size": 1000,
    "chunk_overlap": 200,

    # 数据库配置
    "database_url": "sqlite:///./data/conversation_history.db",  # 默认使用SQLite
    "database_pool_size": 5,
    "database_max_overflow": 10,
    "database_pool_timeout": 30,
    "database_pool_recycle": -1,
    "database_echo": False,  # 设为True可以看到SQL日志
}

# 模型映射配置
LLM_PROVIDER_MAP = {
    "claude-3-sonnet-20240229": "anthropic",
    "claude-3-opus-20240229": "anthropic",
    "gpt-4o": "openai",
    "gpt-4-turbo": "openai",
    "qwen-plus": "qwen",
    "glm-4": "glm",
}

# 向量维度配置
EMBEDDING_DIMENSIONS = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "BAAI/bge-large-zh-v1.5": 1024,
}

# 向量维度配置
Checkpointer_Conf = {
    # Checkpointer Configuration
    "CHECKPOINTER_TYPE": "sqlite", # "sqlite" or "memory"
    "SQLITE_DB_PATH":"data/conversation_history.db" # Path for SQLite DB
}
