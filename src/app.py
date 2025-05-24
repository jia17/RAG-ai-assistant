"""
LangGraph应用构建和运行逻辑
支持数据库持久化的多轮对话
"""

import os
import uuid
import json
import traceback
from typing import Dict, Any, Optional, List
from datetime import datetime

from src.langgraph.graph import create_graph
from src.utils.logger import get_logger
from src.langgraph.nodes.context_manager_node import checkpointer
from src.storage import get_database_manager
from src.langgraph.states import KubeSphereAgentState, ContextManagementState, WorkflowControlState

logger = get_logger(__name__)


# 设置状态持久化  TODO: 这里可以使用SQLite持久化
# setup_checkpointer(CHECKPOINT_DIR)

# 启用 LangSmith 跟踪（需要有效的 API 密钥）



# 应用单例
_app_instance = None

def create_app():
    """创建并编译LangGraph应用
    
    Returns:
        编译好的LangGraph应用
    """
    global _app_instance
    
    # 使用单例模式避免重复创建
    if _app_instance is not None:
        return _app_instance
    
    # 创建状态图
    graph = create_graph()
    
    logger.info("KubeSphere Agent Graph compiled.")

    # 编译图，使用数据库检查点保存器
    _app_instance = graph.compile(checkpointer=checkpointer)
    
    return _app_instance

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
    
    logger.info(f"处理查询，conversation_id: {conversation_id[:8]}...")
    
    # 尝试从数据库加载之前的状态
    db_manager = get_database_manager()
    previous_state = None
    
    try:
        previous_state = db_manager.load_conversation_state(conversation_id)
        if previous_state:
            logger.info(f"已加载会话 {conversation_id[:8]}... 的历史状态")
            # 验证加载的状态数据格式
            if not isinstance(previous_state, dict):
                logger.warning(f"加载的状态数据格式异常: {type(previous_state)}")
                previous_state = None
    except Exception as e:
        logger.warning(f"无法加载会话历史: {e}")
        previous_state = None
    
    initial_state = None
    # 准备初始状态
    if previous_state and isinstance(previous_state, dict):
        # 检查是否是有效的对话状态
        has_valid_state = any(key in previous_state for key in ["original_query", "context", "workflow"])
        
        if has_valid_state:
            # 更新现有状态
            try:
                initial_state = previous_state.copy()
                initial_state["original_query"] = query
                
                # 更新上下文
                if "context" in initial_state and isinstance(initial_state["context"], dict):
                    initial_state["context"]["current_query"] = query
                else:
                    initial_state["context"] = {
                        "conversation_id": conversation_id, 
                        "conversation_history": [],
                        "current_query": query,
                        "current_answer": ""
                    }
                
                # 确保工作流状态存在
                if "workflow" not in initial_state or not isinstance(initial_state["workflow"], dict):
                    initial_state["workflow"] = {
                        "iteration_count": 0, 
                        "query_rewrite_count": 0, 
                        "needs_web_search": False
                    }
                
                logger.debug(f"使用现有状态，更新查询: {query}")
            except Exception as e:
                logger.error(f"处理历史状态时出错: {e}")
                # 回退到创建新状态
                initial_state = None
    
    # 如果没有有效的历史状态，创建新状态
    if not initial_state:
        initial_state = {
            "original_query": query,
            "context": {
                "conversation_id": conversation_id, 
                "conversation_history": [],
                "current_query": query,
                "current_answer": ""
            },
            "workflow": {
                "iteration_count": 0, 
                "query_rewrite_count": 0, 
                "needs_web_search": False
            }
        }
        logger.debug(f"创建新状态，conversation_id: {conversation_id[:8]}...")
    
    # 配置 LangGraph 的 thread_id
    config = {"configurable": {"thread_id": conversation_id}}

    # 执行图
    result = None
    try:
        # 使用stream模式获取更新
        for event in app.stream(initial_state, config=config, stream_mode="updates"):
            # 记录事件，可用于调试
            logger.debug(f"事件: {list(event.keys())}")
            
            # 如果有END事件，获取最终状态
            if "end_node" in event:
                logger.debug(f"获取到结束事件")
                result = event["end_node"]
                break
        
        # 如果没有获取到结果，使用invoke方式再试一次
        if not result:
            logger.debug("使用invoke方式获取结果")
            result = app.invoke(initial_state, config=config)
            
    except Exception as e:
        logger.error(f"处理查询时出错: {e}")
        logger.error(f"详细错误信息: {traceback.format_exc()}")
        return {
            "query": query,
            "answer": "抱歉，处理您的问题时出现错误。请稍后再试。",
            "conversation_id": conversation_id,
            "state": {},
            "error": str(e)
        }
    
    # 提取答案和对话ID
    answer = ""
    if result:
        answer = result.get("answer", "")
        
        # 确保返回对话ID
        context = result.get("context", {})
        if isinstance(context, dict):
            result_conversation_id = context.get("conversation_id")
            if result_conversation_id and result_conversation_id != conversation_id:
                logger.warning(f"对话ID不一致: 预期 {conversation_id}, 得到 {result_conversation_id}")
                conversation_id = result_conversation_id
    
    if not answer:
        answer = "抱歉，我无法回答这个问题。"
    
    # 保存最终状态到数据库（作为备份）
    # try:
    #     if result and isinstance(result, dict):
    #         # 保存简化的状态信息作为索引
    #         success = db_manager.save_conversation_state(
    #             conversation_id=conversation_id,
    #             state_data={"app_backup": True, "simplified_state": {
    #                 "last_query": query,
    #                 "last_answer": answer,
    #                 "timestamp": datetime.now().isoformat()
    #             }},
    #             current_query=query,
    #             latest_answer=answer
    #         )
    #         if success:
    #             logger.debug(f"保存状态索引成功: {conversation_id[:8]}...")
    # except Exception as e:
    #     logger.debug(f"保存状态索引失败（不影响主流程）: {e}")
    
    return {
        "query": query,
        "answer": answer,
        "conversation_id": conversation_id,
        "state": result or {}
    }


def run_interactive_chat():
    """运行交互式聊天"""
    print("欢迎使用KubeSphere AI助手！输入'退出'或'exit'结束对话。")
    print("输入'新对话'或'new'开始新的对话。")
    
    conversation_id = None
    
    while True:
        # 获取用户输入
        user_input = input("\n请输入您的问题: ").strip()
        
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
        if not user_input:
            print("请输入有效的问题")
            continue
            
        try:
            result = process_query(user_input, conversation_id)
            
            # 重要：更新conversation_id，确保后续对话能够关联
            if result.get("conversation_id"):
                conversation_id = result["conversation_id"]
            
            print("\n回答:", result["answer"])
            
            # 显示对话ID（仅在新对话时显示）
            if conversation_id and not result.get("state", {}).get("context", {}).get("conversation_history"):
                print(f"\n[新对话已开始，ID: {conversation_id[:8]}...]")
            
            # 如果有错误，显示错误信息
            if "error" in result:
                print(f"[错误信息: {result['error']}]")
                
        except Exception as e:
            logger.error(f"处理查询时出错: {e}")
            print("\n抱歉，处理您的问题时出现错误。请稍后再试。")




if __name__ == "__main__":
    run_interactive_chat()
