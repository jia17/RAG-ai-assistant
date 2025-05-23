"""
数据库连接和管理
"""

import os
import json
import traceback
import time
import logging
from typing import Dict, Any, Optional, List
from sqlalchemy import create_engine, Engine, event, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
from datetime import datetime, timedelta
from collections import deque, defaultdict

from .models import Base, ConversationState, ConversationHistory, StateCheckpoint, DatabaseVersion
from ..config import (
    DATABASE_URL, 
    DATABASE_POOL_SIZE, 
    DATABASE_MAX_OVERFLOW,
    DATABASE_POOL_TIMEOUT,
    DATABASE_POOL_RECYCLE,
    DATABASE_ECHO
)
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self, database_url: Optional[str] = None):
        """初始化数据库管理器
        
        Args:
            database_url: 数据库连接URL，如果为None则使用配置中的默认值
        """
        self.database_url = database_url or DATABASE_URL
        self.engine: Optional[Engine] = None
        self.SessionLocal: Optional[sessionmaker] = None
        self._initialize_engine()
        
    def _initialize_engine(self):
        """初始化数据库引擎"""
        try:
            # 根据数据库类型配置引擎参数
            engine_kwargs = {
                "echo": DATABASE_ECHO,
            }
            
            if self.database_url.startswith("sqlite"):
                # SQLite特殊配置以解决并发问题
                engine_kwargs.update({
                    "poolclass": StaticPool,
                    "connect_args": {
                        "check_same_thread": False,  # 允许多线程访问
                        "timeout": 60,  # 增加超时时间到60秒
                        "isolation_level": None,  # 使用自动提交模式
                    },
                    "pool_pre_ping": True,  # 连接前检查
                    "pool_recycle": 300,  # 5分钟回收连接
                })
            else:
                # PostgreSQL配置
                engine_kwargs.update({
                    "pool_size": DATABASE_POOL_SIZE,
                    "max_overflow": DATABASE_MAX_OVERFLOW,
                    "pool_timeout": DATABASE_POOL_TIMEOUT,
                    "pool_recycle": DATABASE_POOL_RECYCLE,
                })
            
            self.engine = create_engine(self.database_url, **engine_kwargs)
            
            # 为 SQLite 设置额外的 PRAGMA
            if self.database_url.startswith("sqlite"):
                @event.listens_for(self.engine, "connect")
                def set_sqlite_pragma(dbapi_connection, connection_record):
                    cursor = dbapi_connection.cursor()
                    # 启用 WAL 模式以提高并发性能
                    cursor.execute("PRAGMA journal_mode=WAL")
                    # 设置忙等待超时
                    cursor.execute("PRAGMA busy_timeout=30000")
                    # 启用外键约束
                    cursor.execute("PRAGMA foreign_keys=ON")
                    # 设置同步模式为NORMAL以提高性能
                    cursor.execute("PRAGMA synchronous=NORMAL")
                    # 设置临时存储在内存中
                    cursor.execute("PRAGMA temp_store=MEMORY")
                    # 增加缓存大小
                    cursor.execute("PRAGMA cache_size=-64000")  # 64MB
                    cursor.close()
            
            self.SessionLocal = sessionmaker(
                bind=self.engine, 
                autoflush=False, 
                autocommit=False,
                expire_on_commit=False  # 避免访问已过期的对象
            )
            
            logger.info(f"数据库引擎初始化成功: {self.database_url}")
            
        except Exception as e:
            logger.error(f"数据库引擎初始化失败: {e}")
            raise
    
    def create_tables(self):
        """创建所有表"""
        try:
            # 确保数据目录存在
            if self.database_url.startswith("sqlite"):
                db_path = self.database_url.replace("sqlite:///", "")
                db_dir = os.path.dirname(db_path)
                if db_dir and not os.path.exists(db_dir):
                    os.makedirs(db_dir, exist_ok=True)
            
            Base.metadata.create_all(bind=self.engine)
            logger.info("数据库表创建成功")
        except Exception as e:
            logger.error(f"创建数据库表失败: {e}")
            raise
    
    @contextmanager
    def get_session(self):
        """获取数据库会话的上下文管理器（改进版）"""
        session = self.SessionLocal()
        try:
            yield session
            # 只在没有错误时提交
            if session.is_active:
                session.commit()
        except Exception as e:
            # 确保回滚
            if session.is_active:
                session.rollback()
            logger.error(f"数据库操作失败: {e}")
            raise
        finally:
            # 确保会话被正确关闭
            try:
                session.close()
            except Exception as e:
                logger.warning(f"关闭数据库会话时出错: {e}")
    
    def _retry_db_operation(self, operation_func, max_retries: int = 3):
        """重试数据库操作
        
        Args:
            operation_func: 要执行的数据库操作函数
            max_retries: 最大重试次数
            
        Returns:
            操作结果
        """
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                return operation_func()
            except Exception as e:
                last_exception = e
                
                # 检查是否是数据库连接问题
                if "result object does not return rows" in str(e).lower():
                    logger.warning(f"数据库结果集问题，重试 {attempt + 1}/{max_retries}")
                elif "database is locked" in str(e).lower():
                    logger.warning(f"数据库锁定，重试 {attempt + 1}/{max_retries}")
                else:
                    logger.warning(f"数据库操作重试 {attempt + 1}/{max_retries}: {e}")
                
                if attempt == max_retries - 1:
                    # 最后一次重试失败，抛出异常
                    break
                
                # 等待一小段时间后重试，递增等待时间
                time.sleep(0.2 * (attempt + 1))
        
        # 如果所有重试都失败，抛出最后一个异常
        if last_exception:
            raise last_exception
        
        return None
    
    def save_conversation_state(
        self, 
        conversation_id: str, 
        state_data: Dict[str, Any],
        current_query: Optional[str] = None,
        latest_answer: Optional[str] = None
    ) -> bool:
        """保存对话状态（彻底简化版本）
        
        Args:
            conversation_id: 对话ID
            state_data: 状态数据（暂时忽略）
            current_query: 当前查询
            latest_answer: 最新答案
            
        Returns:
            总是返回True，不让状态保存失败影响主流程
        """
        try:
            # 简化处理：仅记录基本信息，避免复杂状态序列化
            logger.debug(f"记录对话基本信息: {conversation_id[:8]}... - {current_query[:20] if current_query else 'None'}...")
            
            # 暂时禁用复杂的状态保存，专注于对话历史保存
            # 对话历史保存在 conversation_history 表中是正常工作的
            
            return True  # 总是返回成功，不影响主流程
            
        except Exception as e:
            logger.debug(f"状态保存跳过: {e}")
            return True  # 即使出错也返回True，不影响主流程
    
    def load_conversation_state(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """加载对话状态（简化版本）
        
        Args:
            conversation_id: 对话ID
            
        Returns:
            返回None，让系统使用默认状态
        """
        try:
            # 简化处理：不加载复杂状态，让系统使用默认状态
            # 对话历史会从 conversation_history 表中正确加载
            logger.debug(f"跳过状态加载: {conversation_id[:8]}...")
            return None
            
        except Exception as e:
            logger.debug(f"状态加载跳过: {e}")
            return None
    
    def save_conversation_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        original_query: Optional[str] = None,
        analyzed_query: Optional[str] = None,
        intent: Optional[str] = None,
        confidence_score: Optional[float] = None,
        retrieval_method: Optional[str] = None,
        filtered_chunks_count: Optional[int] = None,
        iteration_count: Optional[int] = None
    ) -> bool:
        """保存对话消息（增强版本）
        
        Args:
            conversation_id: 对话ID
            role: 消息角色 (user/assistant)
            content: 消息内容
            metadata: 完整的元数据
            original_query: 原始查询（仅user消息）
            analyzed_query: 分析后的查询（仅user消息）
            intent: 查询意图
            confidence_score: 置信度分数
            retrieval_method: 检索方法
            filtered_chunks_count: 过滤后的文档块数量
            iteration_count: 迭代次数
            
        Returns:
            是否保存成功
        """
        try:
            with self.get_session() as session:
                # 深度序列化元数据以确保JSON兼容性
                serialized_metadata = self._deep_serialize(metadata) if metadata else None
                
                # 从metadata中提取增强信息（如果没有直接提供的话）
                if metadata and not original_query and role == "user":
                    original_query = metadata.get("original_query")
                
                if metadata and not analyzed_query and role == "user":
                    analyzed_query = metadata.get("analyzed_query")
                
                if metadata and not intent:
                    analysis = metadata.get("analysis", {})
                    if isinstance(analysis, dict):
                        intent = analysis.get("intent")
                
                if metadata and confidence_score is None:
                    if role == "assistant":
                        confidence_score = metadata.get("confidence_score")
                        if confidence_score is None:
                            # 尝试从generation_metadata中获取
                            gen_meta = metadata.get("generation_metadata", {})
                            confidence_score = gen_meta.get("confidence_score")
                
                if metadata and not retrieval_method:
                    gen_meta = metadata.get("generation_metadata", {})
                    if isinstance(gen_meta, dict):
                        retrieval_method = gen_meta.get("retrieval_method")
                
                if metadata and filtered_chunks_count is None:
                    gen_meta = metadata.get("generation_metadata", {})
                    if isinstance(gen_meta, dict):
                        filtered_chunks_count = gen_meta.get("filtered_chunks_count", 0)
                
                if metadata and iteration_count is None:
                    iteration_count = metadata.get("iteration_count", 0)
                
                message = ConversationHistory(
                    conversation_id=conversation_id,
                    role=role,
                    original_query=original_query,
                    analyzed_query=analyzed_query,
                    content=content,
                    intent=intent,
                    confidence_score=confidence_score,
                    retrieval_method=retrieval_method,
                    filtered_chunks_count=filtered_chunks_count or 0,
                    iteration_count=iteration_count or 0,
                    message_metadata=serialized_metadata
                )
                session.add(message)
                
                logger.debug(f"保存增强对话消息成功: {conversation_id} - {role} - intent: {intent}")
                return True
                
        except Exception as e:
            logger.error(f"保存对话消息失败: {e}")
            return False
    
    def get_conversation_history(
        self, 
        conversation_id: str, 
        limit: Optional[int] = None,
        include_analysis: bool = True
    ) -> List[Dict[str, Any]]:
        """获取对话历史（增强版本）
        
        Args:
            conversation_id: 对话ID
            limit: 限制返回的消息数量
            include_analysis: 是否包含分析信息
            
        Returns:
            对话历史列表
        """
        def _get_history_operation():
            with self.get_session() as session:
                try:
                    query = session.query(ConversationHistory).filter(
                        ConversationHistory.conversation_id == conversation_id
                    ).order_by(ConversationHistory.timestamp.asc())
                    
                    if limit:
                        query = query.limit(limit)
                    
                    messages = query.all()
                    
                    history = []
                    for message in messages:
                        try:
                            # 安全的时间戳处理
                            timestamp_str = ""
                            if message.timestamp:
                                timestamp_str = message.timestamp.isoformat()
                        except Exception as e:
                            logger.warning(f"时间戳格式化失败: {e}")
                            timestamp_str = ""
                        
                        # 安全地构建基本历史项
                        history_item = {
                            "role": getattr(message, 'role', 'unknown'),
                            "content": getattr(message, 'content', '') or "",
                            "timestamp": timestamp_str
                        }
                        
                        # 如果需要包含分析信息，添加增强字段
                        if include_analysis:
                            history_item.update({
                                "original_query": getattr(message, 'original_query', None),
                                "analyzed_query": getattr(message, 'analyzed_query', None),
                                "intent": getattr(message, 'intent', None),
                                "confidence_score": getattr(message, 'confidence_score', None),
                                "retrieval_method": getattr(message, 'retrieval_method', None),
                                "filtered_chunks_count": getattr(message, 'filtered_chunks_count', None),
                                "iteration_count": getattr(message, 'iteration_count', None),
                            })
                            
                            # 安全处理metadata
                            try:
                                metadata = getattr(message, 'message_metadata', None)
                                if metadata:
                                    history_item["metadata"] = metadata
                            except Exception as e:
                                logger.warning(f"解析metadata失败: {e}")
                                history_item["metadata"] = None
                        else:
                            # 保持向后兼容，只添加基本的增强信息
                            analyzed_query = getattr(message, 'analyzed_query', None)
                            if message.role == "user" and analyzed_query:
                                history_item["analyzed_query"] = analyzed_query
                            
                            intent = getattr(message, 'intent', None)
                            if intent:
                                history_item["intent"] = intent
                            
                            # 安全处理metadata
                            try:
                                metadata = getattr(message, 'message_metadata', None)
                                if metadata:
                                    history_item["metadata"] = metadata
                            except Exception as e:
                                logger.warning(f"解析metadata失败: {e}")
                        
                        history.append(history_item)
                    
                    logger.debug(f"获取增强对话历史成功: {conversation_id}, {len(history)} 条消息")
                    return history
                    
                except Exception as e:
                    logger.error(f"查询对话历史时出错: {e}")
                    logger.debug(f"查询对话历史详细错误: {traceback.format_exc()}")
                    return []
        
        try:
            return self._retry_db_operation(_get_history_operation)
        except Exception as e:
            logger.error(f"获取对话历史失败: {e}")
            logger.debug(f"获取对话历史详细错误: {traceback.format_exc()}")
            return []
    
    def get_conversation_summary(self, conversation_id: str) -> Dict[str, Any]:
        """获取对话摘要统计信息
        
        Args:
            conversation_id: 对话ID
            
        Returns:
            对话摘要信息
        """
        try:
            with self.get_session() as session:
                messages = session.query(ConversationHistory).filter(
                    ConversationHistory.conversation_id == conversation_id
                ).all()
                
                if not messages:
                    return {"error": "没有找到对话记录"}
                
                # 统计信息
                stats = {
                    "conversation_id": conversation_id,
                    "total_messages": len(messages),
                    "user_messages": 0,
                    "assistant_messages": 0,
                    "unique_intents": set(),
                    "retrieval_methods": set(),
                    "avg_confidence": 0,
                    "total_iterations": 0,
                    "total_chunks": 0,
                    "first_message_time": None,
                    "last_message_time": None
                }
                
                confidence_scores = []
                
                for msg in messages:
                    if msg.role == "user":
                        stats["user_messages"] += 1
                        if msg.intent:
                            stats["unique_intents"].add(msg.intent)
                    elif msg.role == "assistant":
                        stats["assistant_messages"] += 1
                        if msg.confidence_score is not None:
                            confidence_scores.append(msg.confidence_score)
                        if msg.retrieval_method:
                            stats["retrieval_methods"].add(msg.retrieval_method)
                    
                    stats["total_iterations"] += (msg.iteration_count or 0)
                    stats["total_chunks"] += (msg.filtered_chunks_count or 0)
                    
                    # 更新时间范围
                    if msg.timestamp:
                        if not stats["first_message_time"] or msg.timestamp < stats["first_message_time"]:
                            stats["first_message_time"] = msg.timestamp.isoformat()
                        if not stats["last_message_time"] or msg.timestamp > stats["last_message_time"]:
                            stats["last_message_time"] = msg.timestamp.isoformat()
                
                # 计算平均置信度
                if confidence_scores:
                    stats["avg_confidence"] = sum(confidence_scores) / len(confidence_scores)
                
                # 转换集合为列表以便JSON序列化
                stats["unique_intents"] = list(stats["unique_intents"])
                stats["retrieval_methods"] = list(stats["retrieval_methods"])
                
                return stats
                
        except Exception as e:
            logger.error(f"获取对话摘要失败: {e}")
            return {"error": str(e)}
    
    def cleanup_old_conversations(self, days_to_keep: int = 30) -> int:
        """清理旧的对话记录
        
        Args:
            days_to_keep: 保留的天数
            
        Returns:
            删除的记录数
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            with self.get_session() as session:
                # 删除旧的对话状态
                deleted_states = session.query(ConversationState).filter(
                    ConversationState.updated_at < cutoff_date
                ).delete()
                
                # 删除旧的对话历史
                deleted_history = session.query(ConversationHistory).filter(
                    ConversationHistory.timestamp < cutoff_date
                ).delete()
                
                total_deleted = deleted_states + deleted_history
                logger.info(f"清理旧对话记录: 删除 {total_deleted} 条记录")
                return total_deleted
                
        except Exception as e:
            logger.error(f"清理旧对话记录失败: {e}")
            return 0

    def _deep_serialize(self, obj):
        """深度序列化对象，处理不能JSON序列化的类型，确保中文字符正确处理"""
        if obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [self._deep_serialize(item) for item in obj]
        elif isinstance(obj, deque):
            # 将deque转换为列表
            return [self._deep_serialize(item) for item in list(obj)]
        elif isinstance(obj, defaultdict):
            # 将defaultdict转换为普通字典
            return {str(k): self._deep_serialize(v) for k, v in dict(obj).items()}
        elif isinstance(obj, dict):
            return {str(k): self._deep_serialize(v) for k, v in obj.items()}
        elif hasattr(obj, '__dict__'):
            # 对于自定义对象，尝试序列化其属性
            return self._deep_serialize(obj.__dict__)
        else:
            # 对于其他不能序列化的对象，转换为字符串
            try:
                # 尝试JSON序列化测试（确保中文字符不被转义）
                json.dumps(obj, ensure_ascii=False)
                return obj
            except (TypeError, ValueError):
                logger.warning(f"无法序列化对象类型 {type(obj)}，转换为字符串: {str(obj)[:100]}")
                return str(obj)


# 全局数据库管理器实例
_db_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """获取全局数据库管理器实例"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
        _db_manager.create_tables()
    return _db_manager


def init_database(database_url: Optional[str] = None) -> DatabaseManager:
    """初始化数据库
    
    Args:
        database_url: 数据库连接URL
        
    Returns:
        数据库管理器实例
    """
    global _db_manager
    _db_manager = DatabaseManager(database_url)
    _db_manager.create_tables()
    return _db_manager 