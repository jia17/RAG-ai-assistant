"""
自适应检索节点
"""

from typing import Dict, Any, List, Optional
from src.models import default_embedding, VectorStoreService # 假设这些已正确定义和导入
from src.config import MILVUS_COLLECTION, MILVUS_HOST, MILVUS_PORT # 假设这些已正确定义和导入
from src.utils.logger import get_logger

from ..states import KubeSphereAgentState, QueryAnalysisState, RetrievalState # 确保正确导入

logger = get_logger(__name__)

class AdaptiveRetriever:
    """自适应检索器，根据查询类型选择不同的检索策略"""

    def __init__(self):
        """初始化检索器"""
        self.vector_store = VectorStoreService(
            collection_name=MILVUS_COLLECTION,
            host=MILVUS_HOST,
            port=MILVUS_PORT
        )
        # 确保 embedding 模型已初始化，如果 default_embedding 是一个模块，
        # 可能需要实例化一个具体的 embedding 类
        self.embedding_model = default_embedding # 或者实例化: default_embedding.YourEmbeddingClass()

    def retrieve(
        self,
        query: str,
        query_type: str = "一般查询",
        components: Optional[List[str]] = None,
        version: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """执行检索

        Args:
            query: 查询文本
            query_type: 查询类型 (从 QueryAnalysisState 获取)
            components: 相关组件列表 (从 QueryAnalysisState 获取)
            version: KubeSphere版本 (从 QueryAnalysisState 获取)
            top_k: 返回结果数量

        Returns:
            检索到的文档列表，每个文档是一个字典，通常包含 'content' 和 'metadata'
        """
        logger.info(f"开始检索: query='{query}', query_type='{query_type}', components={components}, version='{version}', top_k={top_k}")

        # 生成查询嵌入
        # 假设 self.embedding_model 有 embed_query 方法
        try:
            query_embedding = self.embedding_model.embed_query(query)
            logger.debug(f"查询嵌入生成成功，维度: {len(query_embedding)}")
        except Exception as e:
            logger.error(f"生成查询嵌入失败: {e}", exc_info=True)
            return []


        # 构建过滤表达式
        filter_expr = self._build_filter_expression(components, version)
        logger.info(f"构建的Milvus过滤表达式: {filter_expr if filter_expr else '无'}")

        # 根据查询类型调整检索策略 (top_k)
        # TODO: 未来可以考虑更动态的策略
        if query_type: # 确保 query_type 不是 None
            if query_type.lower() in ["故障排除", "错误", "问题", "error", "issue", "troubleshooting"]:
                effective_top_k = max(top_k, 8)
                logger.info(f"查询类型 '{query_type}'，调整 top_k 从 {top_k} 到 {effective_top_k}")
                top_k = effective_top_k
            elif query_type.lower() in ["概念", "解释", "concept", "explanation"]:
                effective_top_k = min(top_k, 3) # 对于概念解释，可能更少的精准结果更好
                logger.info(f"查询类型 '{query_type}'，调整 top_k 从 {top_k} 到 {effective_top_k}")
                top_k = effective_top_k
        else:
            logger.warning("query_type 未提供，使用默认 top_k")


        # 执行向量搜索
        try:
            results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                filter_expr=filter_expr
                # 可以在这里添加 output_fields=['field1', 'field2', 'metadata'] 来指定返回字段
            )
            logger.info(f"向量数据库检索完成，获得 {len(results)} 个结果。")
            # Log a sample of results for debugging if needed
            # if results:
            #     logger.debug(f"检索结果示例: {results[0]}")
            return results
        except Exception as e:
            logger.error(f"向量数据库检索失败: {e}", exc_info=True)
            return []

    def _build_filter_expression(
        self,
        components: Optional[List[str]] = None,
        version: Optional[str] = None
    ) -> Optional[str]:
        """构建Milvus过滤表达式
        
        Args:
            components: 组件列表
            version: KubeSphere版本
            
        Returns:
            过滤表达式字符串或None
        """
        filters = []

        # --- 推荐的基于独立标量字段的过滤 ---
        # 假设Schema中有字段: `ks_version` (VARCHAR), `components` (ARRAY of VARCHAR)
        # 或者 `component` (VARCHAR) 如果一个文档只关联一个组件

        if version:
            # 精确匹配版本号
            filters.append(f"ks_version == '{version}'")
            # 或者范围/前缀匹配:
            # filters.append(f"starts_with(ks_version, '{version_prefix}')")

        if components and len(components) > 0:
            # 如果 'components' 是 ARRAY 类型的字段
            # component_conditions = [f"'{comp}' in components_field_name" for comp in components]
            # filters.append("(" + " or ".join(component_conditions) + ")")

            # 如果 'component' 是 VARCHAR 类型的字段 (一个文档一个组件)
            # 并且希望匹配任何一个指定组件
            component_conditions = [f"component_name == '{comp}'" for comp in components]
            if component_conditions:
                filters.append("(" + " OR ".join(component_conditions) + ")")


        # --- 旧的基于 metadata LIKE 的过滤 (如果 metadata 是一个JSON字符串) ---
        # if components and len(components) > 0:
        #     # 注意: 在JSON字符串中用LIKE搜索效率较低且容易出错
        #     # 确保 component 名称在JSON中不会与其他词部分匹配
        #     component_conditions = []
        #     for component in components:
        #         # 示例: 假设 metadata 是 {"component": "logging", "version": "v3.3"}
        #         # 更安全的做法是针对特定键进行匹配，如果Milvus支持JSON路径查询
        #         # 或者在预处理时将组件和版本提取到独立的、可精确过滤的元数据字段中
        #         component_conditions.append(f"metadata_json_field['component'] == '{component}'") # 假设Milvus支持这种JSON查询
        #         # 或者如果 metadata 是纯字符串:
        #         # component_conditions.append(f"metadata_string_field LIKE '%\"component\":\"{component}\"%'")
        #     if component_conditions:
        #         filters.append("(" + " OR ".join(component_conditions) + ")")
        #
        # if version:
        #     # 示例: metadata_json_field['version'] == '{version}'
        #     # 或者 metadata_string_field LIKE '%\"version\":\"{version}\"%'
        #     filters.append(f"metadata_json_field['version'] == '{version}'")


        if not filters:
            return None

        return " AND ".join(filters)

# 创建全局检索器实例
# 考虑是否应该在函数/类内部按需创建，或者如果应用是单例的，全局也可以
retriever_instance = AdaptiveRetriever()

def retrieve_documents(state: KubeSphereAgentState) -> Dict[str, Any]: 
    """
    检索相关文档

    从状态中获取分析后的查询、实体等信息，调用AdaptiveRetriever进行检索，
    并将检索结果更新状态
    """
    original_query = state["original_query"]
    # 优先使用分析/重写后的查询进行检索，如果存在的话
    query_to_use = state.get("analyzed_query", original_query)

    analysis_state: QueryAnalysisState = state.get("analysis", {}) # 提供默认空字典

    query_type = analysis_state.get("query_type") # 默认会是 None 如果不存在
    components = analysis_state.get("components")
    version = analysis_state.get("version")

    # logger.info(
    #     f"节点 'retrieve_documents' 执行检索: "
    #     f"原始查询='{original_query}', "
    #     f"用于检索的查询='{query_to_use}', "
    #     f"类型='{query_type if query_type else '未指定'}', "
    #     f"组件={components if components else '未指定'}, "
    #     f"版本='{version if version else '未指定'}'"
    # )

    # # 使用检索器获取文档
    # retrieved_chunks = retriever_instance.retrieve(
    #     query=query_to_use,
    #     query_type=query_type if query_type else "一般查询", # 传递默认值给retrieve方法
    #     components=components,
    #     version=version,
    #     top_k=state.get("retrieval", {}).get("top_k", 5) # 从状态中获取top_k或使用默认值
    # )

    # # 构建要更新到状态的 retrieval 部分
    # retrieval_update: RetrievalState = {
    #     "query": query_to_use, # 记录实际用于检索的查询
    #     "rewritten_query": query_to_use if query_to_use != original_query else None, # 如果查询被重写了
    #     "retrieved_chunks": retrieved_chunks,
    #     "retrieval_method": "vector_search_with_filters", 
    #     "top_k": len(retrieved_chunks) # 实际返回的数量
    # }
    
    # 构建要更新到状态的 retrieval_result 部分
    retrieval_update: RetrievalState = {
        "query": query_to_use, # 记录实际用于检索的查询
        "rewritten_query": query_to_use if query_to_use != original_query else None, # 如果查询被重写了
        "retrieved_chunks": "retrieved_chunks",
        "retrieval_method": "vector_search_with_filters", 
        "top_k": 1 # 实际返回的数量
    }
    # 更新状态并返回
    # LangGraph 期望节点返回一个字典，其中包含要更新到状态图状态的键值对
    return {"retrieval_result": retrieval_update}
