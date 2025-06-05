"""
向量数据库服务
"""

import time
from typing import List, Dict, Any, Optional, Union
import json
from pymilvus import (
    connections, 
    Collection, 
    utility,
    FieldSchema, 
    CollectionSchema, 
    DataType
)
from src.config import (
    MILVUS_HOST, MILVUS_PORT, MILVUS_COLLECTION,
    MILVUS_MAX_RETRIES, MILVUS_INITIAL_DELAY, MILVUS_MAX_DELAY, MILVUS_BACKOFF_FACTOR,
    MILVUS_DEFAULT_METRIC_TYPE, MILVUS_CONSISTENCY_LEVEL, MILVUS_ENABLE_SEARCH_OPTIMIZATION
)
from src.models.connection_manager import get_connection_manager, ensure_connection
from src.models.search_optimizer import SearchOptimizer
from src.utils.logger import get_logger

logger = get_logger(__name__)

class VectorStoreService:
    """Milvus向量数据库服务 - 优化版本"""
    
    def __init__(
        self, 
        collection_name: str = MILVUS_COLLECTION,
        host: str = MILVUS_HOST,
        port: int = MILVUS_PORT,
        embedding_dim: int = 384
    ):
        """初始化向量存储服务
        
        Args:
            collection_name: Milvus集合名称
            host: Milvus服务器主机
            port: Milvus服务器端口
            embedding_dim: 嵌入向量维度
        """
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.embedding_dim = embedding_dim
        self.collection = None
        
        # 初始化连接管理器和搜索优化器
        self.connection_manager = get_connection_manager()
        self.search_optimizer = SearchOptimizer() if MILVUS_ENABLE_SEARCH_OPTIMIZATION else None
        
        logger.info(f"VectorStoreService初始化: {collection_name}, 搜索优化: {MILVUS_ENABLE_SEARCH_OPTIMIZATION}")
    
    def _ensure_connection(self) -> bool:
        """确保Milvus连接可用
        
        Returns:
            bool: 连接是否成功
        """
        return ensure_connection()
    
    def _get_collection(self) -> Collection:
        """获取或创建集合
        
        Returns:
            Collection: Milvus集合对象
        """
        if self.collection is not None:
            return self.collection
            
        if not self._ensure_connection():
            raise ConnectionError("无法连接到Milvus服务器")
            
        try:
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                logger.info(f"加载现有集合: {self.collection_name}")
            else:
                logger.info(f"集合 {self.collection_name} 不存在，将创建新集合")
                raise ValueError(f"集合 {self.collection_name} 不存在，请先使用setup_milvus.py创建集合")
                
            return self.collection
            
        except Exception as e:
            logger.error(f"获取集合失败: {e}")
            raise
    
    def add_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]) -> List[str]:
        """添加文档到向量存储
        
        Args:
            documents: 文档列表，每个文档是一个字典，包含id、content和metadata
            embeddings: 文档嵌入向量列表
            
        Returns:
            添加的文档ID列表
        """
        try:
            collection = self._get_collection()
            
            insert_data = []
            doc_ids = []
            
            for doc, embedding in zip(documents, embeddings):
                doc_id = doc.get('id', f"doc_{len(doc_ids)}")
                doc_ids.append(doc_id)
                
                insert_record = {
                    "id": doc_id,
                    "content": doc.get('content', ''),
                    "embedding": embedding,
                    "source": doc.get('source', ''),
                    "title": doc.get('title', ''),
                    "chunk_index": doc.get('chunk_index', 0),
                    "metadata": doc.get('metadata', {})
                }
                insert_data.append(insert_record)
            
            collection.insert(insert_data)
            collection.flush()
            
            logger.info(f"成功添加 {len(doc_ids)} 个文档到集合 {self.collection_name}")
            return doc_ids
            
        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            return []
    
    def search(
        self, 
        query_embedding: List[float], 
        top_k: int = 5,
        filter_expr: Optional[str] = None,
        metric_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """搜索最相似的文档 - 优化版本
        
        Args:
            query_embedding: 查询嵌入向量
            top_k: 返回的最相似文档数量
            filter_expr: 过滤表达式
            metric_type: 距离度量类型，如果为None则使用默认配置
            
        Returns:
            最相似文档列表
        """
        start_time = time.time()
        
        try:
            collection = self._get_collection()
            collection.load()
            
            # 获取优化的搜索参数
            if self.search_optimizer:
                search_params = self.search_optimizer.get_optimal_search_params(
                    collection=collection,
                    metric_type=metric_type or MILVUS_DEFAULT_METRIC_TYPE,
                    consistency_level=MILVUS_CONSISTENCY_LEVEL
                )
            else:
                # 使用默认搜索参数
                search_params = {
                    "metric_type": metric_type or MILVUS_DEFAULT_METRIC_TYPE,
                    "params": {"nprobe": 64},
                    "consistency_level": MILVUS_CONSISTENCY_LEVEL
                }
            
            logger.debug(f"使用搜索参数: {search_params}")
            
            # 执行搜索
            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=filter_expr,
                output_fields=["content", "source", "title", "chunk_index", "metadata"],
                consistency_level=search_params.get("consistency_level", "Strong")
            )
            
            # 处理搜索结果
            documents = []
            for hit in results[0]:
                doc = {
                    "id": hit.id,
                    "content": hit.entity.get("content"),
                    "source": hit.entity.get("source"),
                    "title": hit.entity.get("title"),
                    "chunk_index": hit.entity.get("chunk_index"),
                    "metadata": hit.entity.get("metadata"),
                    "score": hit.score
                }
                documents.append(doc)
            
            # 性能监控
            search_time = time.time() - start_time
            logger.info(f"搜索完成，返回 {len(documents)} 个结果，耗时: {search_time:.3f}秒")
            
            # 记录性能分析（如果启用了搜索优化）
            if self.search_optimizer:
                performance_analysis = self.search_optimizer.analyze_search_performance(
                    search_time=search_time,
                    result_count=len(documents),
                    query_limit=top_k
                )
                logger.debug(f"搜索性能分析: {performance_analysis}")
                
                # 生成优化建议
                suggestions = self.search_optimizer.suggest_optimization(
                    performance_analysis=performance_analysis,
                    current_params=search_params
                )
                if suggestions:
                    logger.info(f"搜索优化建议: {suggestions}")
            
            return documents
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []
    
    def get_connection_status(self) -> Dict[str, Any]:
        """获取连接状态信息
        
        Returns:
            Dict[str, Any]: 连接状态信息
        """
        try:
            connection_info = self.connection_manager.get_connection_info()
            connection_info.update({
                "collection_name": self.collection_name,
                "collection_exists": utility.has_collection(self.collection_name) if connection_info["connected"] else False,
                "search_optimization_enabled": self.search_optimizer is not None
            })
            return connection_info
        except Exception as e:
            logger.error(f"获取连接状态失败: {e}")
            return {"error": str(e)}
    
    def optimize_collection_index(self, force_rebuild: bool = False) -> bool:
        """优化集合索引
        
        Args:
            force_rebuild: 是否强制重建索引
            
        Returns:
            bool: 优化是否成功
        """
        if not self.search_optimizer:
            logger.warning("搜索优化器未启用，跳过索引优化")
            return False
            
        try:
            collection = self._get_collection()
            
            # 获取当前数据量
            data_count = self.search_optimizer._get_collection_count(collection)
            
            # 获取优化的索引参数
            optimal_index_params = self.search_optimizer.get_optimal_index_params(
                data_count=data_count,
                metric_type=MILVUS_DEFAULT_METRIC_TYPE
            )
            
            if force_rebuild:
                # 删除现有索引
                try:
                    collection.drop_index()
                    logger.info("已删除现有索引")
                except:
                    pass  # 索引可能不存在
                
                # 创建新索引
                collection.create_index(
                    field_name="embedding",
                    index_params=optimal_index_params
                )
                logger.info(f"已使用优化参数重建索引: {optimal_index_params}")
                
            return True
            
        except Exception as e:
            logger.error(f"优化索引失败: {e}")
            return False