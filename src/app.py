"""
LangGraph应用构建和运行逻辑
duolunduihua
"""

from typing import Dict, Any, Optional
import os
import uuid
from src.langgraph.graph import create_graph
from src.utils.logger import get_logger
from src.langgraph.nodes.context_manager_node import setup_checkpointer, checkpointer
import json
from src.langgraph.states import KubeSphereAgentState, ContextManagementState, WorkflowControlState

logger = get_logger(__name__)

# 设置状态持久化  TODO: 这里可以使用SQLite持久化
# CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", ":memory:")
# os.makedirs(CHECKPOINT_DIR, exist_ok=True)
# setup_checkpointer(CHECKPOINT_DIR)

# 启用 LangSmith 跟踪（需要有效的 API 密钥）


def create_app():
    """创建并编译LangGraph应用
    
    Returns:
        编译好的LangGraph应用
    """
    # 创建状态图
    graph = create_graph()
    
    logger.info("KubeSphere Agent Graph compiled.")

    # 编译图
    app = graph.compile(checkpointer=checkpointer)
    
    return app

def process_query(
    query: str, 
    conversation_id: Optional[str] = None
) -> Dict[str, Any]:
    """处理单个查询
    
    Args:
        query: 用户查询文本
        conversation_id: 可选的对话ID，用于多轮对话
        
    Returns:
        处理结果，包含答案和中间状态
    """
    app = create_app()
    
    # # 准备初始状态
    # initial_state = {
    #     "original_query": query,
    #     "context": {"conversation_id": conversation_id} if conversation_id else {}
    # }

    # 如果有会话ID，尝试加载之前的状态
    previous_state = None
    if conversation_id and checkpointer:
        try:
            previous_state = checkpointer.get(conversation_id)
            logger.info(f"已加载会话 {conversation_id} 的历史状态")
        except Exception as e:
            logger.warning(f"无法加载会话历史: {e}")
    
    # 准备初始状态
    if previous_state:
        # 更新现有状态
        initial_state = previous_state.copy()
        initial_state["original_query"] = query
        initial_state["context"]["current_query"] = query
    else:
        # 创建新状态
        conversation_id = conversation_id or str(uuid.uuid4())
        initial_state: KubeSphereAgentState = {
            "original_query": query,
            "context": ContextManagementState(
                conversation_id=conversation_id, 
                conversation_history=[],
                current_query=query,
                current_answer=""
            ),
            "workflow": WorkflowControlState(
                iteration_count=0, 
                query_rewrite_count=0, 
                needs_web_search=False
            )
        }
    
    # 配置 LangGraph 的 thread_id
    config = {"configurable": {"thread_id": conversation_id}}


    # 执行图
    result = None
    try:
        # 使用stream模式获取更新
        for event in app.stream(initial_state, config=config, stream_mode="updates"): #"values"
            # 记录事件，可用于调试
            # logger.info(f"事件: {event}")
            
            # 如果有END事件，获取最终状态
            if "end_node" in event:
                logger.debug(f"获取到结束事件")
                result = event["end_node"]
                break
        
        # 如果没有获取到结果，使用invoke方式再试一次
        if not result:
            result = "app.invoke(initial_state, config=config)"
            
    except Exception as e:
        logger.error(f"处理查询时出错: {e}")
        return {
            "query": query,
            "answer": "抱歉，处理您的问题时出现错误。请稍后再试。",
            "conversation_id": conversation_id,
            "state": {}
        }
    
    # 提取答案和对话ID
    answer = result.get("generation", {}).get("answer", "")
    if not answer:
        answer = result.get("answer", "抱歉，我无法回答这个问题。")
    
    # 确保返回对话ID
    conversation_id = result.get("context", {}).get("conversation_id", conversation_id)
    
    return {
        "query": query,
        "answer": answer,
        "conversation_id": conversation_id,
        "state": result
    }


# 简单的交互式聊天测试功能
def run_interactive_chat():
    """运行交互式聊天"""
    print("欢迎使用KubeSphere AI助手！输入'退出'或'exit'结束对话。")
    print("输入'新对话'或'new'开始新的对话。")
    
    conversation_id = None
    
    while True:
        # 获取用户输入
        user_input = input("\n请输入您的问题: ")
        
        # 检查退出条件
        if user_input.lower() in ["退出", "exit", "quit", "q"]:
            print("感谢使用KubeSphere AI助手，再见！")
            break
            
        # 检查是否需要新对话
        if user_input.lower() in ["新对话", "new", "新会话", "clear"]:
            conversation_id = None
            print("已开始新对话！")
            continue
        
        # 处理查询
        try:
            result = process_query(user_input, conversation_id)
            conversation_id = result["conversation_id"]  # 更新对话ID
            print("\n回答:", result["answer"])
            
            # 可选：显示对话ID
            print(f"\n[对话ID: {conversation_id[:8]}...]")
        except Exception as e:
            logger.error(f"处理查询时出错: {e}")
            print("\n抱歉，处理您的问题时出现错误。请稍后再试。")

if __name__ == "__main__":
    run_interactive_chat()
