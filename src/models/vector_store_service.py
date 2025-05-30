"""
向量数据库服务
"""

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
from src.config import MILVUS_HOST, MILVUS_PORT, MILVUS_COLLECTION
from src.utils.logger import get_logger

logger = get_logger(__name__)

class VectorStoreService:
    """Milvus向量数据库服务"""
    
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
        
        try:
            connections.connect(host=host, port=port)
            logger.info(f"已连接到Milvus服务器: {host}:{port}")
        except Exception as e:
            logger.error(f"连接Milvus服务器失败: {e}")
            raise
    
    def _get_collection(self) -> Collection:
        """获取或创建集合
        
        Returns:
            Collection: Milvus集合对象
        """
        if self.collection is not None:
            return self.collection
            
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
        filter_expr: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """搜索最相似的文档
        
        Args:
            query_embedding: 查询嵌入向量
            top_k: 返回的最相似文档数量
            filter_expr: 过滤表达式
            
        Returns:
            最相似文档列表
        """
        try:
            collection = self._get_collection()
            collection.load()
            
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 64}
            }
            
            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=filter_expr,
                output_fields=["content", "source", "title", "chunk_index", "metadata"],
                consistency_level="Strong"
            )
            
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
            
            logger.info(f"搜索完成，返回 {len(documents)} 个结果")
            return documents
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []