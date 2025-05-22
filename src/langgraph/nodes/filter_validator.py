from typing import Dict, Any, List
from typing_extensions import TypedDict, NotRequired
from langgraph.graph import StateGraph

from ..states import KubeSphereAgentState, FilteringState
from src.utils.logger import get_logger # 假设您有这个logger工具

logger = get_logger(__name__)


def filter_chunks(state: KubeSphereAgentState) -> Dict[str, Any]:
    """
    过滤和验证检索到的文档块
    
    Args:
        state: 当前状态
        
    Returns:
        要更新到状态的字典
    """
    # 获取检索结果
    retrieved_chunks = state.get("retrieval", {}).get("retrieved_chunks", [])
    query = state.get("analyzed_query", state.get("original_query", ""))
    
    logger.info(f"节点 'filter_chunks'：开始处理 {len(retrieved_chunks)} 个检索到的块。查询：'{query}'")

    # 模拟过滤和验证逻辑
    # TODO:这里会使用LLM、交叉编码器或其他相关性评估模型。
    # 以后的项目，可以扩展此逻辑：
    # - 检查块内容是否为空
    # - 使用关键词或简单规则进行初步筛选
    # - 调用模型进行相关性打分并根据阈值过滤
    
    final_filtered_chunks: List[Dict[str, Any]] = []
    filter_scores: List[float] = [] 

    if retrieved_chunks:
        final_filtered_chunks = retrieved_chunks
        # 模拟为所有通过的块打一个完美的分数
        filter_scores = [1.0] * len(final_filtered_chunks)
        logger.info(f"过滤验证（模拟）：通过，保留 {len(final_filtered_chunks)} 个文档块。")
    else:
        logger.warning("过滤验证：没有检索到任何文档块进行过滤。")

    # 3. 构建 FilteringState 的更新部分
    # 注意：FilteringState 定义中包含 query, retrieved_chunks, filtered_chunks, filter_scores
    filtering_update: FilteringState = {
        "query": query,
        "retrieved_chunks": retrieved_chunks, # 记录输入到过滤器的原始块
        "filtered_chunks": final_filtered_chunks,
        "filter_scores": filter_scores # 模拟的score
    }

    updates_to_state: Dict[str, Any] = {
        "filtering": filtering_update,
    }

    return updates_to_state
