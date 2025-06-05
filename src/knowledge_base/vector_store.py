"""
向量存储接口 - 提供统一的向量数据库抽象接口
"""

import time
from typing import List, Dict, Any, Optional, Union
import uuid
from pymilvus import (
    connections, 
    Collection, 
    utility,
    FieldSchema, 
    CollectionSchema, 
    DataType,
    IndexType,
    MetricType
)
from src.config import (
    MILVUS_HOST, MILVUS_PORT, MILVUS_COLLECTION,
    MILVUS_DEFAULT_METRIC_TYPE, MILVUS_CONSISTENCY_LEVEL, MILVUS_ENABLE_SEARCH_OPTIMIZATION
)
from src.models.connection_manager import get_connection_manager, ensure_connection
from src.models.search_optimizer import SearchOptimizer
from src.utils.logger import get_logger

logger = get_logger(__name__)

class VectorStore:
    """向量存储类，封装Milvus向量数据库操作 - 优化版本"""
    
    def __init__(self, collection_name: str = None, embedding_dim: int = 384):
        """初始化向量存储
        
        Args:
            collection_name: 集合名称，默认使用配置中的名称
            embedding_dim: 嵌入向量维度，默认384
        """
        self.collection_name = collection_name or MILVUS_COLLECTION
        self.embedding_dim = embedding_dim
        self.collection = None
        
        # 初始化连接管理器和搜索优化器
        self.connection_manager = get_connection_manager()
        self.search_optimizer = SearchOptimizer() if MILVUS_ENABLE_SEARCH_OPTIMIZATION else None
        
        logger.info(f"VectorStore 初始化，集合: {self.collection_name}, 向量维度: {self.embedding_dim}, 搜索优化: {MILVUS_ENABLE_SEARCH_OPTIMIZATION}")
    
    def _ensure_connection(self) -> bool:
        """确保Milvus连接建立
        
        Returns:
            bool: 连接是否成功
        """
        return ensure_connection()
    
    def create_collection(self) -> bool:
        """创建Milvus集合
        
        Returns:
            bool: 创建是否成功
        """
        if not self._ensure_connection():
            logger.error("无法连接到Milvus服务器")
            return False
            
        try:
            if utility.has_collection(self.collection_name):
                logger.info(f"集合 '{self.collection_name}' 已存在")
                self.collection = Collection(self.collection_name)
                return True
            
            # 定义字段Schema
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True, auto_id=False),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=8192),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
                FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="chunk_index", dtype=DataType.INT64),
                FieldSchema(name="metadata", dtype=DataType.JSON)
            ]
            
            # 创建集合Schema
            schema = CollectionSchema(
                fields=fields,
                description=f"KubeSphere知识库向量存储集合，向量维度: {self.embedding_dim}"
            )
            
            # 创建集合
            self.collection = Collection(
                name=self.collection_name,
                schema=schema,
                using='default'
            )
            
            logger.info(f"集合 '{self.collection_name}' 创建成功")
            
            # 创建索引
            self._create_indexes()
            
            return True
            
        except Exception as e:
            logger.error(f"创建集合失败: {e}")
            return False
    
    def _create_indexes(self):
        """创建索引 - 优化版本"""
        try:
            # 获取优化的索引参数
            if self.search_optimizer:
                index_params = self.search_optimizer.get_optimal_index_params(
                    data_count=None,  # 新集合，使用默认参数
                    metric_type=MILVUS_DEFAULT_METRIC_TYPE
                )
                
                # 转换为Milvus的MetricType枚举
                metric_type_map = {
                    "COSINE": MetricType.COSINE,
                    "L2": MetricType.L2,
                    "IP": MetricType.IP
                }
                
                # 创建向量索引
                self.collection.create_index(
                    field_name="embedding",
                    index_params={
                        "metric_type": metric_type_map.get(index_params["metric_type"], MetricType.COSINE),
                        "index_type": IndexType.IVF_FLAT,
                        "params": index_params["params"]
                    }
                )
                logger.info(f"使用优化参数创建向量索引: {index_params}")
            else:
                # 使用默认索引参数
                index_params = {
                    "metric_type": MetricType.COSINE,
                    "index_type": IndexType.IVF_FLAT,
                    "params": {"nlist": 1024}
                }
                
                self.collection.create_index(
                    field_name="embedding",
                    index_params=index_params
                )
                logger.info(f"使用默认参数创建向量索引")
            
            # 创建标量字段索引
            self.collection.create_index(field_name="source")
            self.collection.create_index(field_name="title")
            
            logger.info(f"集合 '{self.collection_name}' 索引创建成功")
            
        except Exception as e:
            logger.error(f"创建索引失败: {e}")
    
    def add_documents(self, documents: List[Dict], embeddings: List[List[float]]) -> List[str]:
        """批量添加文档到向量存储
        
        Args:
            documents: 文档列表，每个文档包含id、content、source、title、chunk_index、metadata字段
            embeddings: 对应的嵌入向量列表
            
        Returns:
            List[str]: 成功添加的文档ID列表
        """
        if not self._ensure_connection():
            logger.error("无法连接到Milvus服务器")
            return []
            
        if not self.collection:
            if not self.create_collection():
                return []
        
        try:
            insert_data = {
                "id": [],
                "content": [],
                "embedding": [],
                "source": [],
                "title": [],
                "chunk_index": [],
                "metadata": []
            }
            
            added_ids = []
            
            for doc, embedding in zip(documents, embeddings):
                # 生成ID（如果文档没有提供）
                doc_id = doc.get('id', str(uuid.uuid4()))
                
                insert_data["id"].append(doc_id)
                insert_data["content"].append(doc.get('content', ''))
                insert_data["embedding"].append(embedding)
                insert_data["source"].append(doc.get('source', ''))
                insert_data["title"].append(doc.get('title', ''))
                insert_data["chunk_index"].append(doc.get('chunk_index', 0))
                insert_data["metadata"].append(doc.get('metadata', {}))
                
                added_ids.append(doc_id)
            
            self.collection.insert(insert_data)
            self.collection.flush()
            
            logger.info(f"成功添加 {len(added_ids)} 个文档到集合 '{self.collection_name}'")
            return added_ids
            
        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            return []
    
    def search_similar(
        self, 
        query_embedding: List[float], 
        top_k: int = 5,
        filter_expr: str = None,
        metric_type: str = None
    ) -> List[Dict[str, Any]]:
        """搜索最相似的文档 - 优化版本
        
        Args:
            query_embedding: 查询向量
            top_k: 返回结果数量
            filter_expr: 过滤表达式
            metric_type: 距离度量类型
            
        Returns:
            List[Dict]: 搜索结果列表
        """
        start_time = time.time()
        
        if not self._ensure_connection():
            logger.error("无法连接到Milvus服务器")
            return []
            
        if not self.collection:
            logger.error("集合未初始化")
            return []
        
        try:
            self.collection.load()
            
            # 获取优化的搜索参数
            if self.search_optimizer:
                search_params = self.search_optimizer.get_optimal_search_params(
                    collection=self.collection,
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
            
            # 转换度量类型为Milvus枚举
            metric_type_map = {
                "COSINE": MetricType.COSINE,
                "L2": MetricType.L2,
                "IP": MetricType.IP
            }
            
            # 执行搜索
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param={
                    "metric_type": metric_type_map.get(search_params["metric_type"], MetricType.COSINE),
                    "params": search_params["params"]
                },
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
    
    def delete_documents(self, ids: List[str]) -> bool:
        """删除指定ID的文档
        
        Args:
            ids: 要删除的文档ID列表
            
        Returns:
            bool: 删除是否成功
        """
        if not self._ensure_connection():
            return False
            
        if not self.collection:
            logger.error("集合未初始化")
            return False
        
        try:
            # 构建删除表达式
            id_list = "', '".join(ids)
            expr = f"id in ['{id_list}']"
            
            # 执行删除
            self.collection.delete(expr)
            self.collection.flush()
            
            logger.info(f"成功删除 {len(ids)} 个文档")
            return True
            
        except Exception as e:
            logger.error(f"删除文档失败: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """获取集合信息
        
        Returns:
            Dict: 集合统计信息
        """
        if not self._ensure_connection():
            return {}
            
        try:
            if not utility.has_collection(self.collection_name):
                return {"exists": False}
            
            collection = Collection(self.collection_name)
            collection.load()
            
            info = {
                "exists": True,
                "name": self.collection_name,
                "num_entities": collection.num_entities,
                "description": collection.description,
                "schema": {
                    "fields": [
                        {
                            "name": field.name,
                            "type": str(field.dtype),
                            "params": field.params
                        }
                        for field in collection.schema.fields
                    ]
                }
            }
            
            return info
            
        except Exception as e:
            logger.error(f"获取集合信息失败: {e}")
            return {"exists": False, "error": str(e)}
