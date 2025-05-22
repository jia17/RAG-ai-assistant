# tests/unit/models/test_embedding_service.py

import unittest
import sys
import os
import logging
from unittest.mock import patch, MagicMock

# 添加项目根目录到 Python 路径，以便能够导入 src 中的模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# 导入要测试的类
from src.models.embedding_service import EmbeddingService

class TestEmbeddingService(unittest.TestCase):
    """测试 EmbeddingService 类的功能"""
    
    def setUp(self):
        """测试前的设置"""
        # 设置日志
        logging.basicConfig(level=logging.INFO, 
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def test_jina_embedding_model(self):
        """测试 Jina 嵌入模型的基本功能"""
        # 使用离线的 Jina 模型
        embedding_service = EmbeddingService(model_name="jinaai/jina-embeddings-v2-base-zh")
        
        # 获取模型维度
        dimension = embedding_service.get_dimension()
        self.logger.info(f"模型嵌入维度: {dimension}")
        self.assertEqual(dimension, 768)  # Jina v2 base 模型维度应为 768
        
        # 嵌入单个查询
        query = "什么是Kubernetes?"
        query_embedding = embedding_service.embed_query(query)
        self.logger.info(f"查询嵌入向量长度: {len(query_embedding)}")
        self.assertEqual(len(query_embedding), dimension)
        
        # 嵌入多个文档
        docs = [
            "Kubernetes是一个开源的容器编排平台，用于自动化部署、扩展和管理容器化应用程序。",
            "KubeSphere是基于Kubernetes构建的企业级分布式多租户容器平台。",
            "Docker是一个开源的应用容器引擎，让开发者可以打包他们的应用以及依赖包到一个可移植的容器中。"
        ]
        doc_embeddings = embedding_service.embed_documents(docs)
        self.logger.info(f"嵌入了 {len(doc_embeddings)} 个文档块")
        self.assertEqual(len(doc_embeddings), len(docs))
        
        # 计算相似度
        similarity = embedding_service.similarity(query, docs[0])
        self.logger.info(f"查询与第一个文档的相似度: {similarity:.4f}")
        self.assertTrue(0 <= similarity <= 1)
        
        # 计算向量相似度
        vector_similarity = embedding_service.similarity_vector(query_embedding, doc_embeddings[0])
        self.logger.info(f"查询向量与第一个文档向量的相似度: {vector_similarity:.4f}")
        self.assertAlmostEqual(similarity, vector_similarity, places=4)
    
    # @patch('src.models.embedding_service.OpenAIEmbeddings')
    # def test_openai_embedding_api(self, mock_openai):
    #     """测试 OpenAI API 嵌入功能（使用 mock）"""
    
    
    def test_empty_input_handling(self):
        """测试空输入处理"""
        embedding_service = EmbeddingService(model_name="jinaai/jina-embeddings-v2-base-zh")
        
        # 测试空查询
        with self.assertRaises(ValueError):
            embedding_service.embed_query("")
        
        # 测试空文档列表
        with self.assertRaises(ValueError):
            embedding_service.embed_documents([])


if __name__ == '__main__':
    unittest.main()
