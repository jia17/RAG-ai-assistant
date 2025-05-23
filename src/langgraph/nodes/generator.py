"""
生成答案节点
"""
import json
import traceback
from typing import Dict, Any
from typing_extensions import TypedDict, NotRequired
from langgraph.graph import StateGraph
from datetime import datetime
from src.models import default_llm
from src.prompts import ANSWER_GENERATION_PROMPT
from src.utils.logger import get_logger
from src.storage import get_database_manager
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

        # 保存生成的答案到数据库
        conversation_id = state.get("context", {}).get("conversation_id")
        if conversation_id and generated_answer and generated_answer.strip():
            db_manager = get_database_manager()
            
            try:
                # 提取增强信息
                generation_result = state.get("generation", {})
                retrieval_result = state.get("retrieval_result", {})
                filtering_result = state.get("filtering", {})
                evaluation_result = state.get("evaluation", {})
                
                # 计算置信度（从多个来源）
                confidence_score = None
                if evaluation_result.get("evaluation_scores"):
                    eval_scores = evaluation_result["evaluation_scores"]
                    if isinstance(eval_scores, dict) and "score" in eval_scores:
                        confidence_score = eval_scores["score"]
                
                # 检索方法
                retrieval_method = retrieval_result.get("retrieval_method")
                
                # 文档块数量
                filtered_chunks_count = 0
                if filtering_result.get("filtered_chunks"):
                    filtered_chunks_count = len(filtering_result["filtered_chunks"])
                
                # 迭代次数
                iteration_count = state.get("workflow", {}).get("iteration_count", 0)
                
                # 保存助手消息
                success = db_manager.save_conversation_message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=generated_answer,
                    confidence_score=confidence_score,
                    retrieval_method=retrieval_method,
                    filtered_chunks_count=filtered_chunks_count,
                    iteration_count=iteration_count,
                    metadata={
                        "timestamp": datetime.now().isoformat(),
                        "sources": generation_result.get("sources", []),
                        "generation_metadata": {
                            "retrieval_method": retrieval_method,
                            "filtered_chunks_count": filtered_chunks_count,
                            "confidence_score": confidence_score,
                            "evaluation_scores": evaluation_result.get("evaluation_scores"),
                            "chunks_info": [
                                {
                                    "source": chunk.get("metadata", {}).get("source", "未知"),
                                    "score": chunk.get("score", 0)
                                }
                                for chunk in filtered_chunks[:3]  # 只保存前3个块的信息
                            ] if filtered_chunks else []
                        }
                    }
                )
                
                if success:
                    logger.info(f"保存助手消息成功: 置信度={confidence_score}, 检索方法={retrieval_method}, 文档块={filtered_chunks_count}")
                else:
                    logger.warning("保存助手消息失败")
                    
            except Exception as e:
                logger.warning(f"保存助手消息时出错: {e}")
                logger.debug(f"保存助手消息详细错误: {traceback.format_exc()}")

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
