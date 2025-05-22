"""
评估query是否需要重写的节点
"""
import json
from typing import Dict, Any
from typing_extensions import TypedDict, NotRequired
from langgraph.graph import StateGraph
from datetime import datetime
from src.models import default_llm
from src.utils.logger import get_logger
from ..states import KubeSphereAgentState
from src.prompts import SYSTEM_PROMPT
from src.prompts import QUERY_REWRITING_PROMPT

def query_rewriter(state: KubeSphereAgentState) -> KubeSphereAgentState:
    print("\n>>> 进入节点: query_rewriter_node")
    original_query = state["original_query"]
    # 实际会基于 critique_result 和 chat_history 来重写查询
    system_prompt = QUERY_REWRITING_PROMPT


    # 模拟查询重写，考虑效率高 开销小的方法，eg 小型LLM
    # rewritten_query = default_llm.generate(system_prompt, f"原始查询：{original_query}")
    rewritten_query = original_query

    
    print(f"重写后查询: {rewritten_query}")

    # 获取工作流状态
    workflow = state.get("workflow", {})
    
    # # 增加重写计数
    # current_rewrite_count = workflow.get("query_rewrite_count", 0) + 1
    # current_iteration = workflow.get("iteration_count", 0) + 1
    # needs_web_search = workflow.get("needs_web_search", False)
    
    # # 更新工作流状态
    # updated_workflow = {
    #     "iteration_count": current_iteration,
    #     "query_rewrite_count": current_rewrite_count,
    #     "needs_web_search": needs_web_search,
    #     "last_agent_action": "query_rewrite"
    # }


    # 清空之前的检索结果
    return {
        # 更新查询
        "analyzed_query": rewritten_query,
        
        # 清空之前的查询结果
        "retrieval_result": {
            "query": original_query,
            "analyzed_query" : rewritten_query,
            "retrieved_chunks": [],
            "retrieval_method": "",
            "top_k": 0
        },
        "filtering": {
            "query": rewritten_query,
            "retrieved_chunks": [],
            "filtered_chunks": [],
            "filter_scores": []
        },
        

        "generation": None,  # 清空生成结果
        "evaluation": None,  # 清空评估结果
        
    }