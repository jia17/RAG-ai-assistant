# tests/unit/knowladge/data_process.py

import unittest
import sys
import os
import logging
import tempfile
import json
from unittest.mock import patch, MagicMock

# 添加项目根目录到 Python 路径，以便能够导入 src 中的模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# 导入要测试的类
from src.knowledge_base.data_processor import DataProcessor

from src.knowledge_base.markdown_splitter import MarkdownSplitter 
from src.models.embedding_service import EmbeddingService # 假设 EmbeddingService 已定义


# 禁用日志输出，避免测试时输出过多日志
# logging.disable(logging.CRITICAL)


class TestDataProcessorDirectory(unittest.TestCase):
    """测试 DataProcessor 处理目录的功能"""

    def setUp(self):
        """测试前的准备工作"""
        # 导入 DataProcessor 类
        from src.knowledge_base.data_processor import DataProcessor
        
        # 创建临时目录
        self.temp_dir = "../../../data/raw/docs"
        self.input_dir = "../../../data/raw/docs"
        self.output_dir = "../../../data/processed/chunks"
        
        # 创建输入目录结构
        # os.makedirs(self.input_dir)
        # os.makedirs(os.path.join(self.input_dir, "zh-cn", "v3.4"))
        
        if not os.path.exists( self.input_dir ):
            print(f"输入目录 {self.input_dir} 不存在，创建中...")

        # 创建测试 Markdown 文件
        self.create_test_markdown_files()
        
        # 初始化模拟服务
        self.embedding_service = EmbeddingService(model_name="jinaai/jina-embeddings-v2-base-zh")
        self.splitter = MarkdownSplitter(chunk_size=200, chunk_overlap=20)
        
        # 初始化 DataProcessor
        self.processor = DataProcessor(
            splitter=self.splitter,
            embedding_service=self.embedding_service,
            output_dir=self.output_dir
        )
        
        # 基础 URL
        self.base_url = "https://test.example.com/docs"

    def tearDown(self):
        """测试后的清理工作"""
        # 删除临时目录
        # shutil.rmtree(self.temp_dir)

    def create_test_markdown_files(self):
        """创建测试用的 Markdown 文件"""
        # 主目录中的文件
        # with open(os.path.join(self.input_dir, "skywalking-kubesphere.md"), "w", encoding="utf-8") as f:
        #     f.write("# 测试文档简介\n\n这是一个测试文档。\n\n## 主要功能\n\n- 功能一\n- 功能二")
        
        # # 子目录中的文件
        # with open(os.path.join(self.input_dir, "zh-cn", "v3.4", "guide.md"), "w", encoding="utf-8") as f:
        #     f.write("# 使用指南\n\n请按照以下步骤操作。\n\n### 第一步\n\n初始化环境。")

    def test_process_directory(self):
        """测试处理目录功能"""
        # 处理目录
        chunks = self.processor.process_directory(
            input_directory=self.input_dir,
            base_url_for_docs=self.base_url
        )
        
        # 验证是否成功处理了文件
        self.assertTrue(self.splitter.split_markdown, "应该调用了 split_markdown 方法")
        self.assertGreater(len(chunks), 0, "应该至少处理了一些文本块")
        
        # 验证块的基本结构
        for chunk in chunks:
            self.assertIn("chunk_text", chunk, "每个块应该包含文本内容")
            self.assertIn("metadata", chunk, "每个块应该包含元数据")
            self.assertIn("global_chunk_id", chunk["metadata"], "元数据应该包含全局块ID")
            self.assertIn("source_url", chunk["metadata"], "元数据应该包含源URL")
            self.assertIn("local_file_path", chunk["metadata"], "元数据应该包含文件路径")
            self.assertIn("relative_file_path", chunk["metadata"], "元数据应该包含相对文件路径")
        
        # 验证是否处理了所有文件
        file_paths = set(chunk["metadata"]["local_file_path"] for chunk in chunks)
        relative_paths = set(chunk["metadata"]["relative_file_path"] for chunk in chunks)

        # 获取目录下所有文件的路径
        expected_files = []
        for root, dirs, files in os.walk(self.input_dir):
            for file in files:
                expected_files.append(file)
        expected_files = set(expected_files)
        self.assertEqual(relative_paths, expected_files, "处理的文件应该与预期的文件列表匹配")
        
        # 验证相对路径和URL构建
        for chunk in chunks:
            relative_path = chunk["metadata"]["relative_file_path"]
            source_url = chunk["metadata"]["source_url"]
            
            # URL应该是基础URL加相对路径
            expected_url = f"{self.base_url}/{relative_path}"
            self.assertEqual(source_url, expected_url, "源URL应该正确构建")

    def test_save_data_with_embeddings(self):
        """测试保存带嵌入的数据功能"""
        # 先处理目录获取块
        chunks = self.processor.process_directory(
            input_directory=self.input_dir,
            base_url_for_docs=self.base_url
        )
        
        # 保存带嵌入的数据
        output_prefix = "test_output"
        self.processor.save_data_with_embeddings(
            chunks_with_metadata=chunks,
            output_filename_prefix=output_prefix,
            embedding_batch_size=5
        )
        
        # 验证是否调用了嵌入服务
        self.assertTrue(self.embedding_service.embed_documents, "应该调用了嵌入服务")
        
        # 验证输出文件是否创建
        expected_output_file = os.path.join(self.output_dir, f"{output_prefix}_embeddings.jsonl")
        self.assertTrue(os.path.exists(expected_output_file), f"应该创建输出文件 {expected_output_file}")
        
        # 验证输出文件内容
        with open(expected_output_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), len(chunks), "输出文件应该包含与块数量相同的行")
            
            # 检查第一行的内容结构
            first_record = json.loads(lines[0])
            self.assertIn("id", first_record, "每条记录应该包含ID")
            self.assertIn("text", first_record, "每条记录应该包含文本")
            self.assertIn("metadata", first_record, "每条记录应该包含元数据")
            self.assertIn("embedding", first_record, "每条记录应该包含嵌入向量")
            self.assertEqual(len(first_record["embedding"]), self.embedding_service.get_dimension(), 
                            "嵌入向量维度应该与服务提供的一致")

if __name__ == "__main__":
    unittest.main()