
"""
查询分析节点
"""
import re
import os
import json

from typing import Dict, Any
from src.models import default_llm
from src.prompts import QUERY_ANALYSIS_PROMPT
from src.utils.logger import get_logger
from ..states import KubeSphereAgentState
from src.prompts import SYSTEM_PROMPT

logger = get_logger(__name__)

def analyze_query(state: KubeSphereAgentState) -> Dict[str, Any]: 
    """
    分析用户查询，确定意图和类型
    
    Args:
        state: 当前状态
        
    Returns:
        更新后的状态
    """
    original_query = state["original_query"]
    chat_history = state["context"].get("conversation_history", "")
    logger.info(f"分析查询: {original_query}")
    logger.info(f"对话历史: {chat_history}")
    
    # 准备提示
        # 准备提示
    prompt = QUERY_ANALYSIS_PROMPT.format(
        original_query=original_query,
        chat_history=chat_history
    )
    
    try:
        # 调用LLM进行分析
        response = default_llm.generate(
            system_prompt=SYSTEM_PROMPT,
            prompt=prompt
        )
        
         # 解析JSON响应
        analysis_result = extract_json_from_response(response["text"])
        logger.info(f"查询分析结果: {analysis_result}")
        
        # 更新状态
        return {
            "analyzed_query": analysis_result.get("rewritten_query_for_retrieval", original_query),
            "query_entities": analysis_result.get("entities", {}),
            "analysis": {
                "query": original_query,
                "intent": analysis_result.get("intent", ""),
                "entities": analysis_result.get("entities", {}),
                "sub_queries": analysis_result.get("sub_queries", []),
                "clarification_question": analysis_result.get("clarification_question", ""),
                "rewritten_query_for_retrieval": analysis_result.get("rewritten_query_for_retrieval", original_query),
                "needs_clarification": analysis_result.get("needs_clarification", False)
            }
        }
    except Exception as e:
        logger.error(f"查询分析失败: {e}")
        # 返回默认分析结果
        return {
            "analysis": {
                "query": original_query,
                "intent": "未确定",
                "entities": {},
                "sub_queries": [],
                "clarification_question": "",
                "rewritten_query_for_retrieval": original_query,
                "needs_clarification": False
            }
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