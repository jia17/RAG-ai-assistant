from typing import TypedDict, List, Optional, Dict, Any

# class KubeSphereAgentState(TypedDict):
#     # 原始输入与查询处理
#     original_query: str                 # 用户最初的查询
#     analyzed_query: str                 # LLM分解或重写后的查询，用于检索
#     query_entities: Optional[Dict[str, Any]] # 从查询中识别的实体 (如版本、组件、错误码)
    
#     # 检索与内容处理
#     retrieved_chunks: Optional[List[Dict[str, Any]]] # 检索到的原始文本块列表
#     filtered_chunks: Optional[List[Dict[str, Any]]] # 经过相关性验证和过滤后的文本块
#     web_search_results: Optional[str]   # 外部网页搜索得到的结果 (如果执行)
    
#     # 生成与评估
#     generation: Optional[str]           # LLM生成的答案
#     critique_result: Optional[Dict[str, Any]] # 答案评估结果 (例如：{score: float, decision: str, reasoning: str})
#                                         # decision可以是: "Accept", "Reject_RewriteQuery", "Reject_WebSearch", "CannotAnswer"
    
#     # 工作流控制与历史
#     chat_history: List[Dict[str,str]]   # 对话历史 (例如: [{"user": "...", "assistant": "..."}, ...])
#     iteration_count: int                # 当前主要RAG循环的迭代次数
#     query_rewrite_count: int            # 查询重写的次数
#     error_message: Optional[str]        # 流程中发生的错误信息
#     last_agent_action: Optional[str]    # 记录Agent的上一个主要动作，用于调试和流转判断
#     needs_web_search: bool              # 标志是否需要进行Web搜索



"""
LangGraph状态定义
"""

from typing import Dict, List, Any, Optional, TypedDict, Annotated, Sequence, Union
from typing_extensions import NotRequired

# 定义状态类型
class QueryAnalysisState(TypedDict):
    """查询分析状态"""
    query: str
    intent: NotRequired[str]
    entities: NotRequired[List[str]]
    sub_queries: NotRequired[List[str]]
    clarification_question: NotRequired[List[str]]
    rewritten_query_for_retrieval: NotRequired[List[str]]
    needs_clarification: NotRequired[bool]

class RetrievalState(TypedDict):
    """检索状态"""
    query: str
    analyzed_query: NotRequired[str]
    retrieved_chunks: NotRequired[List[Dict[str, Any]]]
    retrieval_method: NotRequired[str]
    top_k: NotRequired[int]

class FilteringState(TypedDict):
    """过滤状态"""
    query: str
    retrieved_chunks: List[Dict[str, Any]]
    filtered_chunks: NotRequired[List[Dict[str, Any]]]
    filter_scores: NotRequired[List[float]]

class GenerationState(TypedDict):
    """生成状态"""
    query: str
    filtered_chunks: List[Dict[str, Any]]
    answer: NotRequired[str]
    sources: NotRequired[List[Dict[str, Any]]]

class EvaluationState(TypedDict):
    """评估状态"""
    query: str
    answer: str
    filtered_chunks: List[Dict[str, Any]]
    evaluation_scores: NotRequired[Dict[str, float]]
    needs_improvement: NotRequired[bool]
    decision: NotRequired[List[str]]

class WebSearchState(TypedDict):
    """Web搜索状态"""
    query: str
    web_results: NotRequired[List[Dict[str, Any]]]
    web_search_success: NotRequired[bool]
    search_params: NotRequired[Dict[str, Any]]  # 动态搜索参数
    api_usage: NotRequired[Dict[str, int]]  # API使用统计
    search_quality_score: NotRequired[float]  # 搜索质量评分
    fallback_attempted: NotRequired[bool]  # 是否尝试了降级方案
    cached_results: NotRequired[List[Dict[str, Any]]]  # 缓存的搜索结果
    search_metadata: NotRequired[Dict[str, Any]]  # 搜索元数据（响应时间、来源等）
    error_message: NotRequired[str]  # 搜索过程中的错误信息
    search_type: NotRequired[str]  # 搜索类型 (simple, complex, urgent)
    bilingual_search_used: NotRequired[bool]  # 是否使用了双语搜索

class ContextManagementState(TypedDict):
    """上下文管理状态"""
    conversation_id: NotRequired[str]
    conversation_history: NotRequired[List[Dict[str, str]]]
    current_query: str
    current_answer: NotRequired[str]
    # turn_count: NotRequired[int]
    
class WorkflowControlState(TypedDict):
    """工作流控制与历史"""
    iteration_count: int                # 当前主要RAG循环的迭代次数
    query_rewrite_count: int            # 查询重写的次数
    # error_message: Optional[str]        # 流程中发生的错误信息
    # last_agent_action: Optional[str]    # 记录Agent的上一个主要动作
    needs_web_search: bool              # 标志是否需要进行Web搜索


# 组合状态
class KubeSphereAgentState(TypedDict):
    """完整状态"""

    # 原始输入与查询处理
    original_query: str                 # 用户最初的查询
    analyzed_query: str                 # LLM分解或重写后的查询，用于检索
    query_entities: Optional[Dict[str, Any]] # 从查询中识别的实体 (如版本、组件、错误码)
    
    answer: NotRequired[str]

    error_message: NotRequired[str]        # 流程中发生的错误信息

    # 分析状态
    analysis: NotRequired[QueryAnalysisState]
    
    # 检索状态
    retrieval_result: NotRequired[RetrievalState]
    
    # 过滤状态
    filtering: NotRequired[FilteringState]
    
    # 生成状态
    generation: NotRequired[GenerationState]
    
    # 评估状态
    evaluation: NotRequired[EvaluationState]
    
    # Web搜索状态
    web_search: NotRequired[WebSearchState]
    
    # 上下文管理
    context: NotRequired[ContextManagementState]
    
    workflow: NotRequired[WorkflowControlState]
    
    # 工作流控制与历史
    # iteration_count: int                # 当前主要RAG循环的迭代次数
    # query_rewrite_count: int            # 查询重写的次数
    # needs_web_search: bool              # 标志是否需要进行Web搜索