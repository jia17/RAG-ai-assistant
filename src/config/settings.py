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
    "qwen_api_key": "sk-c1c95e661b1443f78f10c86fe570585e",
    "glm_api_key": "",
    
    # Milvus配置
    "milvus_host": "localhost",
    "milvus_port": 19530,
    "milvus_collection": "kubesphere_docs",
    
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
