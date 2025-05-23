"""
查询分析节点
"""
import re
import os
import json
import traceback
from datetime import datetime
from typing import Dict, Any

from src.models import default_llm
from src.prompts import QUERY_ANALYSIS_PROMPT
from src.utils.logger import get_logger
from src.storage import get_database_manager
from ..states import KubeSphereAgentState
from src.prompts import SYSTEM_PROMPT

logger = get_logger(__name__)

def analyze_query(state: KubeSphereAgentState) -> Dict[str, Any]: 
    """
    分析用户查询，提取意图和实体，决定是否需要澄清
    
    Args:
        state: 包含原始查询的状态
        
    Returns:
        更新后的状态，包含分析结果
    """
    original_query = state["original_query"]
    conversation_history = state.get("context", {}).get("conversation_history", [])
    
    logger.info(f"分析查询: {original_query}")
    # logger.info(f"对话历史: {conversation_history}")
    
    # 构建增强的对话历史字符串，包含分析信息
    if conversation_history:
        enhanced_history = []
        for msg in conversation_history:
            if msg["role"] == "user":
                user_content = f"用户: {msg['content']}"
                # 如果有分析后的查询，也包含进去
                if msg.get("analyzed_query") and msg["analyzed_query"] != msg["content"]:
                    user_content += f"\n（分析重写为: {msg['analyzed_query']}）"
                # 如果有意图分析，也包含
                if msg.get("analysis", {}).get("intent"):
                    user_content += f"\n（意图: {msg['analysis']['intent']}）"
                enhanced_history.append(user_content)
            elif msg["role"] == "assistant":
                assistant_content = f"助手: {msg['content']}"
                # 如果有来源信息，简要说明
                if msg.get("sources"):
                    source_count = len(msg["sources"]) if isinstance(msg["sources"], list) else 1
                    assistant_content += f"\n（基于{source_count}个知识来源）"
                enhanced_history.append(assistant_content)
        
        formatted_history = "\n".join(enhanced_history[-6:])  # 最近3轮对话
        logger.info(f"增强对话历史: {formatted_history}")
    else:
        formatted_history = "无对话历史"
    
    # 准备提示
    prompt = QUERY_ANALYSIS_PROMPT.format(
        original_query=original_query,
        chat_history=formatted_history
    )
    
    try:
        # 调用LLM进行查询分析
        response = default_llm.generate(
            system_prompt=SYSTEM_PROMPT,
            prompt=prompt
        )
        
        # 解析JSON响应
        analysis_result = extract_json_from_response(response["text"])
        
        # 验证必需字段
        if not analysis_result or not all(key in analysis_result for key in ["intent", "entities", "rewritten_query_for_retrieval"]):
            raise ValueError("分析结果缺少必需字段")
        
        logger.info(f"查询分析结果: {analysis_result}")
        
        # 保存分析结果到数据库（重要：在这里保存用户消息的分析信息）
        conversation_id = state.get("context", {}).get("conversation_id")
        if conversation_id:
            db_manager = get_database_manager()
            
            analyzed_query = analysis_result.get("rewritten_query_for_retrieval", original_query)
            intent = analysis_result.get("intent")
            
            try:
                # 直接保存用户消息及其分析结果（不检查重复）
                success = db_manager.save_conversation_message(
                    conversation_id=conversation_id,
                    role="user",
                    content=original_query,
                    original_query=original_query,
                    analyzed_query=analyzed_query if analyzed_query != original_query else None,
                    intent=intent,
                    metadata={
                        "timestamp": datetime.now().isoformat(),
                        "analysis": analysis_result,
                        "analyzed_query": analyzed_query,
                    }
                )
                
                if success:
                    logger.info(f"保存用户消息分析成功: intent={intent}, analyzed_query={analyzed_query[:50]}...")
                else:
                    logger.warning("保存用户消息分析失败")
                    
            except Exception as e:
                logger.warning(f"保存用户消息分析时出错: {e}")
                logger.debug(f"保存用户消息详细错误: {traceback.format_exc()}")
        else:
            logger.warning("没有conversation_id，无法保存用户消息")
        
        # 返回状态更新
        return {
            "analysis": analysis_result,
            "analyzed_query": analysis_result.get("rewritten_query_for_retrieval", original_query),
            "query_entities": analysis_result.get("entities", {}),
            "needs_clarification": analysis_result.get("needs_clarification", False),
            "clarification_question": analysis_result.get("clarification_question")
        }
        
    except Exception as e:
        logger.error(f"查询分析失败: {e}")
        logger.error(f"查询分析详细错误: {traceback.format_exc()}")
        
        # 返回默认分析结果
        return {
            "analysis": {
                "intent": "信息查询",
                "entities": {},
                "rewritten_query_for_retrieval": original_query,
                "needs_clarification": False
            },
            "analyzed_query": original_query,
            "query_entities": {},
            "needs_clarification": False
        }

def extract_json_from_response(response: str) -> Dict[str, Any]:
    """
    从LLM响应中提取JSON数据
    
    Args:
        response: LLM返回的响应文本
        
    Returns:
        解析后的JSON数据
    """
    # 尝试查找JSON代码块
    json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    matches = re.findall(json_pattern, response)
    
    if matches:
        # 使用找到的第一个JSON块
        try:
            return json.loads(matches[0])
        except json.JSONDecodeError:
            pass
    
    # 尝试直接解析整个响应
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # 如果所有尝试都失败，返回空字典
        logger.error(f"无法从响应中提取JSON: {response}")
        return {}