"""
上下文管理节点 - 处理多轮对话的状态管理，使用数据库持久化
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid
import os

from langgraph.checkpoint.memory import MemorySaver
from src.utils.logger import get_logger
from src.storage import get_database_checkpointer, get_database_manager

logger = get_logger(__name__)

# 全局检查点保存器实例
checkpointer = None

def manage_context(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    管理对话上下文，更新对话历史
    
    Args:
        state: 当前状态
        
    Returns:
        更新后的状态
    """
    # 获取当前查询和生成的答案
    query = state["original_query"]
    answer = state.get("generation", {}).get("answer", "")
    
    # 获取分析后的查询和其他分析结果
    analyzed_query = state.get("analyzed_query", query)
    analysis_result = state.get("analysis", {})
    
    # 初始化上下文
    if "context" not in state:
        state["context"] = {
            "conversation_history": [],
            "current_query": query,
            "conversation_id": state.get("context", {}).get("conversation_id") or str(uuid.uuid4())
        }
    
    # 初始化工作流控制状态
    if "workflow" not in state:
        state["workflow"] = {
            "iteration_count": 0,
            "query_rewrite_count": 0,
            "needs_web_search": False
        }
    else:
        # 如果是新查询，重置迭代计数
        if state.get("original_query") != state.get("context", {}).get("current_query", ""):
            state["workflow"]["iteration_count"] = 0
            state["workflow"]["query_rewrite_count"] = 0
    
    # 获取数据库管理器
    db_manager = get_database_manager()
    conversation_id = state["context"]["conversation_id"]
    
    # 获取现有对话历史
    conversation_history = state["context"].get("conversation_history", [])
    
    # 如果没有历史记录，尝试从数据库加载（仅查询操作）
    if not conversation_history:
        try:
            db_history = db_manager.get_conversation_history(conversation_id)
            conversation_history = []
            for msg in db_history:
                # 构建增强的历史记录
                history_item = {"role": msg["role"], "content": msg["content"]}
                
                # 如果有分析信息，添加到历史记录中
                if msg.get("analyzed_query"):
                    history_item["analyzed_query"] = msg["analyzed_query"]
                if msg.get("intent"):
                    history_item["analysis"] = {"intent": msg["intent"]}
                if msg.get("metadata"):
                    metadata = msg["metadata"]
                    if msg["role"] == "assistant" and metadata.get("sources"):
                        history_item["sources"] = metadata["sources"]
                
                conversation_history.append(history_item)
            logger.debug(f"从数据库加载增强的对话历史: {len(conversation_history)} 条消息")
        except Exception as e:
            logger.warning(f"加载对话历史失败: {e}")
            conversation_history = []
    
    # 更新对话历史（仅在内存中）
    # 添加用户消息（如果不是重复的）
    user_message = {
        "role": "user",
        "content": query
    }
    
    # 为用户消息添加分析信息
    if analyzed_query and analyzed_query != query:
        user_message["analyzed_query"] = analyzed_query
    if analysis_result:
        user_message["analysis"] = analysis_result
    
    # 检查是否为新消息
    is_new_message = True
    if conversation_history:
        last_message = conversation_history[-1]
        if last_message.get("role") == "user" and last_message.get("content") == query:
            is_new_message = False
    
    if is_new_message:
        conversation_history.append(user_message)
        logger.debug(f"添加用户消息到内存历史: {query[:50]}...")
    
    # 如果有答案，添加助手消息
    if answer and answer.strip():
        assistant_message = {
            "role": "assistant",
            "content": answer
        }
        
        # 为助手消息添加生成信息
        generation_info = state.get("generation", {})
        if generation_info.get("sources"):
            assistant_message["sources"] = generation_info["sources"]
        
        # 检查是否为新答案
        is_new_answer = True
        if conversation_history:
            last_message = conversation_history[-1]
            if (last_message.get("role") == "assistant" and 
                last_message.get("content") == answer):
                is_new_answer = False
        
        if is_new_answer:
            conversation_history.append(assistant_message)
            logger.debug(f"添加助手消息到内存历史: {answer[:50]}...")
    
    # 更新状态（主要状态保存由DatabaseCheckpointer和专门的保存节点负责）
    updated_context = {
        "conversation_id": conversation_id,
        "conversation_history": conversation_history,
        "current_query": query,
        "current_answer": answer
    }
    
    return {
        "context": updated_context
    }


def setup_checkpointer(use_database: bool = True) -> None:
    """设置检查点机制进行状态持久化"""
    global checkpointer
    
    try:
        if use_database:
            # 使用改进后的数据库检查点保存器
            checkpointer = get_database_checkpointer()
            logger.info("已启用数据库状态持久化（DatabaseCheckpointer）")
        else:
            # 使用内存检查点保存器（兼容性选项）
            checkpointer = MemorySaver()
            logger.info("已启用内存状态持久化")
            
    except Exception as e:
        logger.error(f"设置检查点保存器失败: {e}")
        # 回退到内存保存器
        checkpointer = MemorySaver()
        logger.warning("回退到内存状态持久化")


def cleanup_old_conversations(days_to_keep: int = 30) -> int:
    """清理旧的对话记录
    
    Args:
        days_to_keep: 保留的天数
        
    Returns:
        删除的记录数
    """
    try:
        db_manager = get_database_manager()
        deleted_count = db_manager.cleanup_old_conversations(days_to_keep)
        logger.info(f"清理完成，删除了 {deleted_count} 条旧记录")
        return deleted_count
    except Exception as e:
        logger.error(f"清理旧对话记录失败: {e}")
        return 0


# 初始化检查点保存器
# 检查环境变量决定是否使用数据库持久化
USE_DATABASE_PERSISTENCE = os.environ.get("USE_DATABASE_PERSISTENCE", "true").lower() == "true"
setup_checkpointer(USE_DATABASE_PERSISTENCE)

