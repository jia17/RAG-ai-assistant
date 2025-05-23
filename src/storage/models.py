"""
数据库模型定义
"""

from sqlalchemy import Column, String, Text, DateTime, Integer, JSON, Index, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime
from typing import Dict, Any, Optional

Base = declarative_base()


class ConversationState(Base):
    """对话状态表 - 存储LangGraph的状态数据"""
    
    __tablename__ = "conversation_states"
    
    # 主键：conversation_id (thread_id)
    conversation_id = Column(String(255), primary_key=True, index=True)
    
    # 状态数据 (JSON格式存储完整的LangGraph状态)
    state_data = Column(JSON, nullable=False)
    
    # 当前查询
    current_query = Column(Text, nullable=True)
    
    # 最新答案
    latest_answer = Column(Text, nullable=True)
    
    # 迭代计数
    iteration_count = Column(Integer, default=0)
    query_rewrite_count = Column(Integer, default=0)
    
    # 时间戳
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    
    # 索引
    __table_args__ = (
        Index('idx_conversation_updated', 'updated_at'),
        Index('idx_conversation_created', 'created_at'),
    )


class ConversationHistory(Base):
    """对话历史表 - 存储详细的对话记录，包含丰富的分析信息"""
    
    __tablename__ = "conversation_history"
    
    # 主键
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # 外键：关联对话状态
    conversation_id = Column(String(255), nullable=False, index=True)
    
    # 消息角色：user 或 assistant
    role = Column(String(20), nullable=False)
    
    # 原始查询内容（仅对user消息有效）
    original_query = Column(Text, nullable=True)
    
    # 分析后的查询内容（仅对user消息有效）
    analyzed_query = Column(Text, nullable=True)
    
    # 消息内容（用户原始输入或助手的回答）
    content = Column(Text, nullable=False)
    
    # 查询意图（如：寻求概念解释、功能比较等）
    intent = Column(String(100), nullable=True)
    
    # 置信度分数
    confidence_score = Column(Float, nullable=True)
    
    # 检索方法（adaptive、semantic、keyword等）
    retrieval_method = Column(String(50), nullable=True)
    
    # 过滤后的文档块数量
    filtered_chunks_count = Column(Integer, default=0)
    
    # 迭代次数
    iteration_count = Column(Integer, default=0)
    
    # 消息的完整元数据（JSON格式，包含所有详细信息）
    message_metadata = Column(JSON, nullable=True)
    
    # 时间戳
    timestamp = Column(DateTime, default=func.now(), nullable=False)
    
    # 索引
    __table_args__ = (
        Index('idx_history_conversation_time', 'conversation_id', 'timestamp'),
        Index('idx_history_role', 'role'),
        Index('idx_history_intent', 'intent'),
        Index('idx_history_retrieval_method', 'retrieval_method'),
    )


class StateCheckpoint(Base):
    """LangGraph兼容的检查点表"""
    
    __tablename__ = "state_checkpoints"
    
    # 主键
    thread_id = Column(String(255), primary_key=True)
    checkpoint_id = Column(String(255), primary_key=True)
    
    # 检查点数据
    state = Column(JSON, nullable=False)
    config = Column(JSON, nullable=True)
    
    # 检查点元数据 - 重命名避免与SQLAlchemy的metadata冲突
    checkpoint_metadata = Column(JSON, nullable=True)
    
    # 时间戳
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    # 索引
    __table_args__ = (
        Index('idx_checkpoint_thread_time', 'thread_id', 'created_at'),
    )


class DatabaseVersion(Base):
    """数据库版本信息表"""
    
    __tablename__ = "database_versions"
    
    # 主键
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # 版本号
    version = Column(String(50), nullable=False, unique=True)
    
    # 描述
    description = Column(Text, nullable=True)
    
    # 应用时间
    applied_at = Column(DateTime, default=func.now(), nullable=False) 