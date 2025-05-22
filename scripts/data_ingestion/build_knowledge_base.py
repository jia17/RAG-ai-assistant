"""
构建知识库脚本 - 从文件夹读取MD文件，进行切分、嵌入并写入数据文件
"""

import argparse
import logging
import os
from datetime import datetime

from src.knowledge_base.document_splitter import DocumentSplitter
from src.models.embedding_service import EmbeddingService
from src.knowledge_base.data_processor import DataProcessor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"knowledge_base_build_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="构建KubeSphere知识库")
    parser.add_argument("--docs_dir", type=str, required=True, help="Markdown文档目录路径")
    parser.add_argument("--output_dir", type=str, default="./data", help="输出目录")
    parser.add_argument("--base_url", type=str, default="https://kubesphere.io/docs", help="文档基础URL")
    parser.add_argument("--output_name", type=str, default="kubesphere_kb", help="输出文件名前缀")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="嵌入模型名称")
    parser.add_argument("--chunk_size", type=int, default=1000, help="文档切分大小")
    parser.add_argument("--chunk_overlap", type=int, default=200, help="文档切分重叠大小")
    
    args = parser.parse_args()
    
    try:
        logger.info("开始构建知识库")
        logger.info(f"文档目录: {args.docs_dir}")
        logger.info(f"输出目录: {args.output_dir}")
        logger.info(f"使用嵌入模型: {args.model}")
        
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 初始化组件
        splitter = DocumentSplitter(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        
        embedding_service = EmbeddingService(
            model_name=args.model
        )
        
        processor = DataProcessor(
            splitter=splitter,
            embedding_service=embedding_service,
            output_dir=args.output_dir
        )
        
        # 处理文档目录
        logger.info("开始处理文档目录")
        chunks = processor.process_directory(
            directory_path=args.docs_dir,
            base_url=args.base_url
        )
        
        # 创建向量存储
        logger.info("开始创建向量存储")
        processor.create_vector_store(
            chunks=chunks,
            output_name=args.output_name
        )
        
        logger.info("知识库构建完成")
        
    except Exception as e:
        logger.error(f"知识库构建失败: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
