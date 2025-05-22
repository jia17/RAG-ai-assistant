"""
LangGraph状态图定义
"""

from typing import Dict, Any, TypedDict, Annotated, List, Literal, Union, cast
from langgraph.graph import StateGraph, END
from src.utils.logger import get_logger
from .states import KubeSphereAgentState
from .nodes.query_analyser import analyze_query
from .nodes.adaptive_retriever import retrieve_documents
from .nodes.filter_validator import filter_chunks
from .nodes.generator import generate_answer
from .nodes.answer_critique import evaluate_answer
from .nodes.query_rewriter import query_rewriter
from .nodes.web_search_node import web_search
from ..config import MAX_ITERATIONS

logger = get_logger(__name__)


from .nodes.context_manager_node import manage_context

def create_graph() -> StateGraph:
    """
    创建KubeSphere AI助手的状态图
    
    Returns:
        配置好的StateGraph实例
    """
    # 创建状态图
    workflow = StateGraph(KubeSphereAgentState)
    
    # 添加节点
    workflow.add_node("context_manager_node", manage_context)# 新增上下文管理节点
    workflow.add_node("query_analysis_node", analyze_query)
    workflow.add_node("retrieval_node", retrieve_documents)
    workflow.add_node("filtering_node", filter_chunks)  # 注意这里也修改了名称
    workflow.add_node("generation_node", generate_answer)
    workflow.add_node("evaluation_node", evaluate_answer)
    workflow.add_node("query_rewriter_node", query_rewriter)# 重用查询分析节点作为查询重写
    workflow.add_node("web_search_node", web_search)
    workflow.add_node("end_node", lambda state: state)  # 结束节点 TODO: 这里可以是一个实际的结束处理函数, 用户可以连续提问


    # 设置边
    # 首先处理上下文
    workflow.add_edge("context_manager_node", "query_analysis_node")
    
    # 查询分析 -> 检索
    workflow.add_edge("query_analysis_node", "retrieval_node")
    
    # 检索 -> 过滤
    workflow.add_edge("retrieval_node", "filtering_node")
    
    # 过滤 -> 生成
    workflow.add_edge("filtering_node", "generation_node")
    
    # 生成 -> 评估
    workflow.add_edge("generation_node", "evaluation_node")
    
    # # 评估 -> 上下文管理（更新对话历史）
    # workflow.add_edge("evaluation_node", "context_manager_node")

    # 从查询重写回到检索 - 使用新的节点名称
    workflow.add_edge("query_rewriter_node", "retrieval_node")
    

    workflow.add_edge("end_node", END)  # 结束节点指向终止状态
    # 评估 -> 决策点
    # 添加评估后的条件边
    workflow.add_conditional_edges(
        "evaluation_node",
        route_based_on_evaluation,
        {
            "query_rewriter_node": "query_rewriter_node",
            "retrieval_node": "retrieval_node",
            "context_manager_node": "context_manager_node",
            "web_search_node": "web_search_node",
            "end_node": "end_node"
        }
    )
    
    # # TODO:
    # # 添加从过滤到查询重写的条件边
    # def decide_after_filtering(state: KubeSphereAgentState) -> str:
    #     """根据过滤结果决定是否需要重写查询"""
    #     filtered_chunks = state.get("filtering", {}).get("filtered_chunks", [])
        
    #     if not filtered_chunks or len(filtered_chunks) < 2:
    #         # 如果没有找到足够相关的文档，尝试重写查询
    #         if state.get("workflow", {}).get("iteration_count", 0) < MAX_ITERATIONS:
    #             return "query_rewriter_node"
        
    #     # 默认继续生成答案
    #     return "generation_node"

    # # 添加条件边
    # workflow.add_conditional_edges(
    #     "filtering_node",
    #     decide_after_filtering,
    #     {
    #         "query_rewriter_node": "query_rewriter_node",
    #         "generation_node": "generation_node"
    #     }
    # )

    # # 从查询重写回到检索 - 使用新的节点名称
    # workflow.add_edge("query_rewriter_node", "retrieval_node")

    # 设置入口点
    workflow.set_entry_point("context_manager_node")
    
    return workflow


# 评估后的决策路由
def route_based_on_evaluation(state: KubeSphereAgentState) -> str:
    """根据评估结果决定下一步"""
    evaluation = state.get("evaluation", {})
    needs_improvement = evaluation.get("needs_improvement", "False")
    decision = evaluation.get("decision", "Accept")
    
    # 获取工作流状态
    workflow = state.get("workflow", {})
    iteration_count = workflow.get("iteration_count", 0)
    query_rewrite_count = workflow.get("query_rewrite_count", 0)
    
    logger.info(f"评估决策: {decision}, 当前迭代: {iteration_count}/{MAX_ITERATIONS}")
    
    if decision == "Accept":
        # 答案被接受，更新上下文后结束
        logger.info("答案被接受，流程结束")
        logger.info("答案： %s", evaluation.get("answer", ""))
        return "end_node"  # 修改为新节点名称
    
    elif decision == "Reject_RewriteQuery" and query_rewrite_count < 2:
        return "end_node"  # 修改为新节点名称
    
        # 需要重写查询，且未超过最大重写次数
        logger.info("答案被拒绝，需要重写查询")
        return "query_rewriter_node"  # 修改为新节点名称
    
    elif iteration_count < MAX_ITERATIONS:
        return "end_node"  # 修改为新节点名称
    
        # 其他拒绝情况，但未达到最大迭代次数
        # 这里不直接回到generation，而是回到retrieval
        # 
        logger.info(f"答案被拒绝，尝试迭代 )")
        return "retrieval_node"  # 修改为新节点名称
    
    else:
        return "end_node"  # 修改为新节点名称
    
        # 达到最大迭代次数，更新上下文后结束
        # 这里可以选择进行Web搜索
        workflow["needs_web_search"] = True
        logger.info(f"达到最大迭代次数 ({MAX_ITERATIONS})，流程结束")
        return "web_search_node"  # 修改为新节点名称