"""
上下文管理节点 - 处理多轮对话的状态管理，使用LangGraph标准检查点机制
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid
from src.utils.logger import get_logger
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver
import os

logger = get_logger(__name__)

# 全局变量，用于存储检查点配置
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
    
    # 初始化上下文
    if "context" not in state:
        state["context"] = {
            "conversation_history": [],
            "current_query": query
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
    
    # 获取现有对话历史
    conversation_history = state["context"].get("conversation_history", [])
    
    # 添加用户消息
    user_message = {
        "role": "user",
        "content": query
    }
    conversation_history.append(user_message)
    
    # 如果有答案，添加助手消息
    if answer:
        assistant_message = {
            "role": "assistant",
            "content": answer
        }
        conversation_history.append(assistant_message)
    
    # 更新状态
    return {
        "context": {
            "conversation_history": conversation_history,
            "current_query": query,
            "current_answer": answer
        }
    }

def setup_checkpointer(directory_path: str) -> None:
    """设置检查点机制进行状态持久化"""
    print(f"设置对话状态持久化: {directory_path}")

    global checkpointer
    try:
        # 确保目录存在
        os.makedirs(directory_path, exist_ok=True)
        
        # 根据路径选择合适的保存器
        if directory_path == ":memory:":
            # 使用 with 语句正确获取 SqliteSaver 实例
            # with SqliteSaver.from_conn_string(":memory:") as saver:
            #     checkpointer = saver
            
            checkpointer = MemorySaver()
            print("使用内存中的SQLite保存器")
        elif directory_path.endswith(".db") or directory_path.endswith(".sqlite"):
            # 使用 with 语句正确获取 SqliteSaver 实例
            with SqliteSaver.from_conn_string(f"sqlite:///{directory_path}") as saver:
                checkpointer = saver
            print(f"使用SQLite保存器: {directory_path}")
        else:
            # 使用文件路径
            db_path = f"{directory_path}/conversations.db"
            # 使用 with 语句正确获取 SqliteSaver 实例
            with SqliteSaver.from_conn_string(f"sqlite:///{db_path}") as saver:
                checkpointer = saver
            print(f"使用SQLite保存器: {db_path}")
        
        print(f"已设置对话状态持久化: {directory_path}")
    except Exception as e:
        print(f"设置持久化保存器失败: {e}")
        checkpointer = None


# 测试检查点机制 TODO:持久化历史
# def test_checkpointer():
#     """测试检查点机制是否正常工作"""
#     if not checkpointer:
#         print("检查点机制未设置")
#         return
    
#     test_key = "test_conversation_id"
#     test_data = {"test": "data", "timestamp": str(datetime.now())}
    
#     try:
#         # 保存测试数据
#         checkpointer.put(test_key, test_data)
#         print(f"已保存测试数据: {test_key}")
        
#         # 读取测试数据
#         retrieved_data = checkpointer.get(test_key)
#         print(f"已读取测试数据: {retrieved_data}")
        
#         # 验证数据
#         if retrieved_data == test_data:
#             print("检查点机制工作正常")
#         else:
#             print("检查点机制数据不匹配")
            
#         # 清理测试数据
#         checkpointer.delete(test_key)
#         print("已清理测试数据")
        
#     except Exception as e:
#         print(f"测试检查点机制失败: {e}")

# 在设置完检查点后调用测试

# 设置状态持久化  TODO: 这里可以使用SQLite持久化
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", ":memory:")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
setup_checkpointer(CHECKPOINT_DIR)
# test_checkpointer()

