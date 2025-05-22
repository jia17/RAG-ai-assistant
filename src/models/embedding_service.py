"""
嵌入模型服务
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
from src.utils.logger import get_logger
from pathlib import Path
from transformers import AutoModel
import re
import torch

logger = get_logger(__name__)

class EmbeddingService:
    """
    嵌入模型服务，支持多种嵌入模型和API服务。
    可以自动选择使用离线模型或API服务，取决于可用性和配置。
    """

    def __init__(self,
                 model_name: str = "jinaai/jina-embeddings-v2-base-zh",
                 device: Optional[str] = None,
                 normalize_embeddings: bool = True,
                 use_api: bool = False):
        """
        初始化嵌入服务。

        Args:
            model_name: 模型名称或路径
            device: 设备名称 ('cuda', 'cpu' 等)
            normalize_embeddings: 是否归一化嵌入向量
            use_api: 是否强制使用API (即使离线模型可用)
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.normalize_embeddings = normalize_embeddings
        self.use_api = use_api
        self._model = None
        self._api_client = None
        self._model_type = self._determine_model_type()
        
        logger.info(f"EmbeddingService 初始化，模型: {self.model_name}, 类型: {self._model_type}, 设备: {self.device}")

    def _determine_model_type(self) -> str:
        """
        确定要使用的模型类型。
        
        Returns:
            str: 模型类型 ('huggingface', 'openai', 'zhipu', 'jina')
        """
        if self.use_api:
            if "openai" in self.model_name.lower():
                return "openai"
            elif "zhipu" in self.model_name.lower():
                return "zhipu"
            elif "jina" in self.model_name.lower() and self.use_api:
                return "jina_api"
            else:
                return "openai"  # 默认API
        else:
            # 尝试判断是否可以离线使用
            if "jina" in self.model_name.lower():
                return "jina"
            else:
                return "huggingface"  # 默认使用HuggingFace

    def _load_model(self):
        """
        根据模型类型加载相应的模型或API客户端。
        """
        if self._model is not None or self._api_client is not None:
            return
            
        try:
            if self._model_type == "huggingface":
                from langchain.embeddings.huggingface import HuggingFaceEmbeddings
                logger.info(f"加载HuggingFace嵌入模型: {self.model_name}")
                self._model = HuggingFaceEmbeddings(model_name=self.model_name, model_kwargs={'device': self.device})
                
            elif self._model_type == "jina":
                from transformers import AutoModel, AutoTokenizer
                logger.info(f"加载Jina嵌入模型: {self.model_name}")
                self._model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
                self._model.to(self.device)
                self._model.eval()
                
            elif self._model_type == "openai":
                from langchain.embeddings.openai import OpenAIEmbeddings
                logger.info(f"初始化OpenAI嵌入API")
                self._api_client = OpenAIEmbeddings()
                
            elif self._model_type == "zhipu":
                from zhipuai import ZhipuAI
                logger.info(f"初始化ZhipuAI嵌入API")
                api_key = os.getenv("ZHIPUAI_API_KEY")
                if not api_key:
                    raise ValueError("未设置ZHIPUAI_API_KEY环境变量")
                self._api_client = ZhipuAI(api_key=api_key)
                
            elif self._model_type == "jina_api":
                import requests
                logger.info(f"初始化Jina API客户端")
                self._api_client = requests
                # 这里需要API密钥，可以从环境变量获取
                self._api_key = os.getenv("JINA_API_KEY")
                if not self._api_key:
                    logger.warning("未设置JINA_API_KEY环境变量，API调用可能会失败")
                
            logger.info(f"嵌入模型/API初始化成功: {self.model_name}")
            
        except ImportError as e:
            logger.error(f"导入错误: {e}. 请安装相关依赖。")
            raise
        except Exception as e:
            logger.error(f"加载嵌入模型/API失败: {e}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """
        将单个查询文本转换为向量。

        Args:
            text: 输入的查询文本

        Returns:
            文本的向量表示
        """
        self._load_model()
        
        if not text or not isinstance(text, str):
            logger.warning("embed_query 接收到空或无效的文本输入")
            return []
            
        try:
            if self._model_type == "huggingface":
                embedding = self._model.embed_query(text)
                
            elif self._model_type == "jina":
                # 使用Jina模型直接编码
                with torch.no_grad():
                    embedding = self._model.encode([text])[0]
                if isinstance(embedding, torch.Tensor):
                    embedding = embedding.cpu().numpy()
                    
            elif self._model_type == "openai":
                text = text.replace("\n", " ")  # OpenAI API建议
                embedding = self._api_client.embed_query(text)
                
            elif self._model_type == "zhipu":
                response = self._api_client.embeddings.create(
                    model="embedding-2",
                    input=text
                )
                embedding = response.data[0].embedding
                
            elif self._model_type == "jina_api":
                url = "https://api.jina.ai/v1/embeddings"
                headers = {
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "input": text,
                    "model": self.model_name
                }
                response = self._api_client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                embedding = response.json()["data"][0]["embedding"]
                
            # 归一化向量（如果需要）
            if self.normalize_embeddings and isinstance(embedding, np.ndarray):
                embedding = embedding / np.linalg.norm(embedding)
                
            return embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
            
        except Exception as e:
            logger.error(f"查询文本嵌入失败: {e}")
            raise

    def embed_documents(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        将多个文档文本转换为向量。

        Args:
            texts: 输入的文本块列表
            batch_size: 用于编码的批处理大小

        Returns:
            每个文本块的向量表示列表
        """
        self._load_model()
        
        if not texts or not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            logger.warning("embed_documents 接收到空或无效的文本列表输入")
            return []
            
        try:
            all_embeddings = []
            
            # 根据不同模型类型处理批量嵌入
            if self._model_type == "huggingface":
                # HuggingFace模型通常支持批量处理
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]
                    batch_embeddings = [self._model.embed_query(text) for text in batch_texts]
                    all_embeddings.extend(batch_embeddings)
                    
            elif self._model_type == "jina":
                # Jina模型直接支持批量编码
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]
                    with torch.no_grad():
                        batch_embeddings = self._model.encode(batch_texts)
                    if isinstance(batch_embeddings, torch.Tensor):
                        batch_embeddings = batch_embeddings.cpu().numpy()
                    all_embeddings.extend(batch_embeddings.tolist())
                    
            elif self._model_type == "openai":
                # 使用OpenAI API批量处理
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]
                    # 替换换行符，OpenAI API建议
                    batch_texts = [text.replace("\n", " ") for text in batch_texts]
                    batch_embeddings = self._api_client.embed_documents(batch_texts)
                    all_embeddings.extend(batch_embeddings)
                    
            elif self._model_type == "zhipu":
                # 使用智谱API逐个处理（可能不支持批量）
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]
                    batch_embeddings = []
                    for text in batch_texts:
                        response = self._api_client.embeddings.create(
                            model="embedding-2",
                            input=text
                        )
                        batch_embeddings.append(response.data[0].embedding)
                    all_embeddings.extend(batch_embeddings)
                    
            elif self._model_type == "jina_api":
                # 使用Jina API批量处理
                url = "https://api.jina.ai/v1/embeddings"
                headers = {
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json"
                }
                
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]
                    payload = {
                        "input": batch_texts,
                        "model": self.model_name
                    }
                    response = self._api_client.post(url, json=payload, headers=headers)
                    response.raise_for_status()
                    batch_embeddings = [item["embedding"] for item in response.json()["data"]]
                    all_embeddings.extend(batch_embeddings)
                
            # 归一化向量（如果需要）
            if self.normalize_embeddings:
                all_embeddings = [
                    (np.array(emb) / np.linalg.norm(emb)).tolist() 
                    if np.linalg.norm(emb) > 0 else emb 
                    for emb in all_embeddings
                ]
                
            return all_embeddings
            
        except Exception as e:
            logger.error(f"文档列表嵌入失败: {e}")
            raise

    def get_dimension(self) -> Optional[int]:
        """
        返回嵌入模型的输出维度。

        Returns:
            嵌入向量的维度，如果无法确定则返回None
        """
        self._load_model()
        
        try:
            # 尝试通过嵌入一个简单文本来确定维度
            sample_embedding = self.embed_query("测试维度")
            return len(sample_embedding)
        except Exception as e:
            logger.error(f"获取嵌入维度失败: {e}")
            return None

    def similarity(self, text1: str, text2: str) -> float:
        """
        比较两个文本的相似度。

        Args:
            text1: 第一个文本
            text2: 第二个文本

        Returns:
            两个文本的余弦相似度（0到1之间）
        """
        embedding1 = self.embed_query(text1)
        embedding2 = self.embed_query(text2)
        
        if not embedding1 or not embedding2:
            return 0.0
            
        return self.similarity_vector(embedding1, embedding2)

    def similarity_vector(self, vector1: List[float], vector2: List[float]) -> float:
        """
        比较两个向量的相似度。

        Args:
            vector1: 第一个向量
            vector2: 第二个向量

        Returns:
            两个向量的余弦相似度（0到1之间）
        """
        if not vector1 or not vector2:
            return 0.0
            
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        
        if not magnitude:
            return 0.0
            
        return float(dot_product / magnitude)