"""
生成答案节点
"""
import json
from typing import Dict, Any
from typing_extensions import TypedDict, NotRequired
from langgraph.graph import StateGraph
from datetime import datetime
from src.models import default_llm
from src.prompts import ANSWER_GENERATION_PROMPT
from src.utils.logger import get_logger
from ..states import KubeSphereAgentState
from src.prompts import SYSTEM_PROMPT

logger = get_logger(__name__)

def generate_answer(state: KubeSphereAgentState) -> Dict[str, Any]:
    """
    生成答案节点
    
    Args:
        state: 当前状态
        
    Returns:
        要更新到状态的字典
    """
    logger.info("生成答案...")
    
    # 获取必要的输入
    original_query = state["original_query"]
    analyzed_query = state.get("analyzed_query", original_query)
    
    # 获取过滤后的文档块
    filtered_chunks = state.get("filtering", {}).get("filtered_chunks", [])
    
    # 获取查询分析结果
    query_analysis = state.get("analysis", {})
    
    # 获取对话历史
    chat_history = state.get("context", {}).get("conversation_history", [])
    
    # 格式化对话历史为文本
    chat_history_text = ""
    if chat_history:
        chat_history_text = "\n".join([
            f"用户: {turn.get('user', '')}\n助手: {turn.get('assistant', '')}"
            for turn in chat_history
        ])
    
    # 格式化上下文块为文本
    context_chunks_text = ""
    if filtered_chunks:
        context_chunks_text = "\n\n".join([
            f"[来源: {chunk.get('source_url', '未知来源')}]\n{chunk.get('chunk_text', '')}"
            for chunk in filtered_chunks
        ])
    
    # 准备生成提示
    prompt = ANSWER_GENERATION_PROMPT.format(
        original_query=original_query,
        context=context_chunks_text,
        chat_history=chat_history_text,
        context_chunks_text=context_chunks_text,
        query_analysis=json.dumps(query_analysis, ensure_ascii=False, indent=2)
    )
    
    try:
        # 调用LLM生成答案
        system_prompt = SYSTEM_PROMPT
        
        response = default_llm.generate(
            system_prompt=system_prompt,
            prompt=prompt,
            # context_chunks=filtered_chunks  # 传递上下文块，以便LLM可以引用
        )
        
        # 提取生成的答案
        generated_answer = response
        if isinstance(response, dict) and "text" in response:
            generated_answer = response["text"]
        
        logger.info(f"生成答案完成: {generated_answer[:100]}...")
        
        # 构建要更新的生成状态
        generation_update = {
            "query": original_query,
            "filtered_chunks": filtered_chunks,
            "answer": generated_answer,
            "sources": [
                {"url": chunk.get("source_url", ""), "title": chunk.get("title", "")}
                for chunk in filtered_chunks
            ],
            # "timestamp": datetime.now().isoformat()
        }
        
    
        # 获取工作流状态
        workflow = state.get("workflow", {})
        # 增加重写计数
        current_rewrite_count = workflow.get("query_rewrite_count", 0) + 1
        current_iteration = workflow.get("iteration_count", 0) + 1
        needs_web_search = workflow.get("needs_web_search", False)
        
        # 更新工作流状态
        updated_workflow = {
            "iteration_count": current_iteration,
            "query_rewrite_count": current_rewrite_count,
            "needs_web_search": needs_web_search,
            "last_agent_action": "query_rewrite"
        }

        # 返回要更新的状态部分
        return {
            # 更新工作流状态
            "workflow": updated_workflow,

            "generation": generation_update,
            "answer": generated_answer,  # 同时更新顶层状态中的答案
        }
    except Exception as e:
        logger.error(f"生成答案失败: {str(e)}")
        error_message = f"生成答案失败: {str(e)}"
        
        # 返回错误信息
        return {
            "generation": {
                "query": original_query,
                # "filtered_chunks": filtered_chunks,
                "answer": "抱歉，在生成答案时遇到了技术问题。请稍后再试。",
                "error": error_message
            },
            "answer": "抱歉，在生成答案时遇到了技术问题。请稍后再试。",
            "error_message": error_message,
            "last_agent_action": "generation_error"
        }
