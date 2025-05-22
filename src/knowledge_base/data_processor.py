from typing import List, Dict, Any, Union # Union 用于类型提示 Path or str
import json
import pickle # 仍然保留，但主要输出改为 JSONL
from pathlib import Path
from src.utils.logger import get_logger
from tqdm import tqdm
import hashlib

from .markdown_splitter import MarkdownSplitter 
from src.models.embedding_service import EmbeddingService # 假设 EmbeddingService 已定义

# logger = get_logger(__name__)

from src.utils.logger import setup_logger

logger = setup_logger()

class DataProcessor:
    """数据预处理器，负责读取、切分、嵌入Markdown文档，并将处理结果保存为JSON Lines文件。"""
    
    def __init__(self, 
                 splitter: 'MarkdownSplitter', # 使用你实际的切分器类名
                 embedding_service: EmbeddingService,
                 output_dir: str = "./processed_data"): # 更改了默认输出目录名
        self.splitter = splitter
        self.embedding_service = embedding_service
        self.output_dir = Path(output_dir) # 使用 Path 对象
        
        self.output_dir.mkdir(parents=True, exist_ok=True) # 确保输出目录存在
        logger.info(f"DataProcessor 初始化完成。输出目录: {self.output_dir.resolve()}")
    
    def _get_file_hash(self, file_path: Union[str, Path]) -> str:
        """计算文件的MD5哈希值"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk_content in iter(lambda: f.read(4096), b""): # 迭代读取文件块
                    hash_md5.update(chunk_content)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"计算文件哈希失败 {file_path}: {e}")
            return "error_hashing_file" # 返回一个错误标识或抛出异常

    def process_md_file(self, 
                        file_path: Union[str, Path], 
                        input_dir_root: Path, # 新增参数：扫描的根目录
                        base_url_for_docs: str = "") -> List[Dict[str, Any]]:
        """
        处理单个Markdown文件。

        Args:
            file_path: Markdown文件的绝对路径。
            input_dir_root: 处理的Markdown文件所在根目录的Path对象。
            base_url_for_docs: 用于构建文档在线URL的基础URL (例如 "https://kubesphere.io/docs")。

        Returns:
            包含文本块及其元数据的字典列表。
        """
        file_path_obj = Path(file_path)
        try:
            with open(file_path_obj, 'r', encoding='utf-8') as f:
                content = f.read()
            
            doc_title = file_path_obj.stem # 文件名（不含扩展名）作为标题
            
            # 计算文件相对于 input_dir_root 的路径
            relative_path_from_root = file_path_obj.relative_to(input_dir_root)

            # 构建 source_url (在线文档的URL)
            # 例如: base_url_for_docs = "https://kubesphere.io/docs"
            #       relative_path_from_root = "zh-cn/v3.3/quick-start/installation.md"
            # Result: "https://kubesphere.io/docs/zh-cn/v3.3/quick-start/installation.md"
            source_url = ""
            if base_url_for_docs:
                # 确保路径分隔符是 /
                url_parts = [base_url_for_docs.rstrip('/')] + [part for part in relative_path_from_root.parts]
                source_url = "/".join(url_parts)

            # 构建基础元数据，将传递给 splitter
            base_doc_metadata = {
                "document_title": doc_title,
                "source_url": source_url, # 指向在线文档的完整URL (如果提供了base_url)
                "local_file_path": str(file_path_obj.resolve()), # 本地绝对路径
                "relative_file_path": str(relative_path_from_root), # 相对于扫描根目录的路径
                "source_type": "markdown",
                "file_hash": self._get_file_hash(file_path_obj)
            }
            
            # 使用 splitter 切分文档，传入基础元数据
            chunks_from_doc = self.splitter.split_markdown(content, base_doc_metadata)
            
            # 为每个块添加全局唯一的ID (基于文件哈希和文档内块ID)
            for chunk in chunks_from_doc:
                chunk['metadata']['global_chunk_id'] = f"{base_doc_metadata['file_hash']}_{chunk['metadata'].get('chunk_id_within_doc', 'N/A')}"

            logger.debug(f"处理Markdown文件 {file_path_obj.name} 完成，生成 {len(chunks_from_doc)} 个块。")
            return chunks_from_doc
        except Exception as e:
            logger.error(f"处理Markdown文件 {file_path_obj} 失败: {e}", exc_info=True)
            raise # 重新抛出异常，由 process_directory 处理

    #TODO: 当处理大量文件时，all_processed_chunks 和 all_embeddings 可能会占用大量内存 
    #允许在处理大量文件时分批保存结果，而不是将所有内容保存在内存中。
    def process_directory(self, 
                          input_directory: str, 
                          base_url_for_docs: str = "") -> List[Dict[str, Any]]:
        """
        处理指定目录中的所有Markdown文件。

        Args:
            input_directory: 包含Markdown文件的根目录路径。
            base_url_for_docs: 用于构建文档在线URL的基础URL。

        Returns:
            所有成功处理的文件的文本块和元数据列表。
        """
        all_processed_chunks = []
        input_dir_path = Path(input_directory).resolve() # 获取绝对路径

        if not input_dir_path.is_dir():
            logger.error(f"输入目录 {input_directory} 不存在或不是一个有效的目录。")
            return []

        # rglob 会递归查找所有匹配 *.md 的文件
        md_files_to_process = list(input_dir_path.rglob("*.md"))

        if not md_files_to_process:
            logger.warning(f"在目录 {input_directory} 中未找到任何Markdown文件。")
            return []

        logger.info(f"在 {input_directory} 中发现 {len(md_files_to_process)} 个Markdown文件。")

        for file_abs_path in tqdm(md_files_to_process, desc="处理Markdown文件"):
            try:
                # 调用 process_md_file，传入文件的绝对路径和根目录路径
                chunks_from_file = self.process_md_file(file_abs_path, input_dir_path, base_url_for_docs)
                all_processed_chunks.extend(chunks_from_file)
            except Exception as e:
                # 记录错误但继续处理其他文件
                logger.error(f"跳过文件 {file_abs_path}，处理过程中发生错误: {e}", exc_info=False) 

        logger.info(f"目录处理完成。共生成 {len(all_processed_chunks)} 个文本块。")
        return all_processed_chunks
    
    def save_data_with_embeddings(self, 
                                  chunks_with_metadata: List[Dict[str, Any]], 
                                  output_filename_prefix: str = "kubesphere_kb",
                                  embedding_batch_size: int = 32):
        """
        为文本块生成嵌入向量，并将所有数据（文本、元数据、嵌入）保存到JSON Lines文件中。

        Args:
            chunks_with_metadata: 包含 "chunk_text" 和 "metadata" 的字典列表。
            output_filename_prefix: 输出的JSONL文件名的前缀。
            embedding_batch_size: 嵌入服务处理文本时的批次大小。
        """
        if not chunks_with_metadata:
            logger.warning("没有提供文本块用于生成嵌入和保存。")
            return

        texts_to_embed = [chunk_data["chunk_text"] for chunk_data in chunks_with_metadata]
        
        logger.info(f"开始为 {len(texts_to_embed)} 个文本块生成嵌入向量...")
        
        all_embeddings = []
        try:
            # 批量生成嵌入
            for i in tqdm(range(0, len(texts_to_embed), embedding_batch_size), desc="生成嵌入向量"):
                batch_texts = texts_to_embed[i : i + embedding_batch_size]
                batch_embeddings = self.embedding_service.embed_documents(batch_texts)
                all_embeddings.extend(batch_embeddings)
        except Exception as e:
            logger.error(f"嵌入向量生成失败: {e}", exc_info=True)
            logger.error("由于嵌入失败，将不会保存数据文件。请检查EmbeddingService。")
            return # 如果嵌入失败，则不继续保存

        if len(all_embeddings) != len(chunks_with_metadata):
            logger.error(f"生成的嵌入向量数量 ({len(all_embeddings)}) 与文本块数量 ({len(chunks_with_metadata)}) 不匹配。数据可能已损坏。")
            return

        # 准备输出文件路径
        output_jsonl_file_path = self.output_dir / f"{output_filename_prefix}_embeddings.jsonl"
        embedding_dimension = self.embedding_service.get_dimension()
        embedding_model_name = self.embedding_service.model_name # 假设 EmbeddingService 有 model_name 属性

        logger.info(f"准备将数据写入到: {output_jsonl_file_path}")

        try:
            with open(output_jsonl_file_path, 'w', encoding='utf-8') as f_out:
                for i, chunk_data in enumerate(chunks_with_metadata):
                    # 构建要写入 JSONL 文件的记录
                    record = {
                        "id": chunk_data["metadata"].get("global_chunk_id", f"chunk_{i}"), # 使用全局ID
                        "text": chunk_data["chunk_text"],
                        "metadata": chunk_data["metadata"], # 包含所有收集到的元数据
                        "embedding": all_embeddings[i],    # 对应的嵌入向量
                        "embedding_model": embedding_model_name,
                        "embedding_dimension": embedding_dimension
                    }
                    # ensure_ascii=False 保证中文字符正确写入
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            
            logger.info(f"数据成功保存到 {output_jsonl_file_path}。共处理 {len(chunks_with_metadata)} 条记录。")

        except IOError as e:
            logger.error(f"写入输出文件 {output_jsonl_file_path} 失败: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"保存数据时发生未知错误: {e}", exc_info=True)


# --- 示例：如何使用 DataProcessor (通常在主脚本或调用模块中) ---
if __name__ == '__main__':
    # 0. 配置日志 (简单示例)
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 1. 初始化依赖的服务
    # 假设 EmbeddingService 和 MarkdownSplitter (或 DocumentSplitter) 已定义且可导入
    # from .embedding_service import EmbeddingService # 假设在同级
    # from .document_splitter import MarkdownSplitter # 假设在同级

    # Mocking services for demonstration if they are not fully implemented yet
    class MockEmbeddingService:
        def __init__(self, model_name="mock-model"):
            self.model_name = model_name
            self._dimension = 10 # Mock dimension
            logger.info(f"MockEmbeddingService initialized with model: {self.model_name}")

        def embed_documents(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
            logger.info(f"MockEmbeddingService: Embedding {len(texts)} documents in batches of {batch_size}...")
            # 返回与文本数量相同的随机向量列表
            import random
            return [[random.random() for _ in range(self._dimension)] for _ in texts]

        def get_dimension(self) -> int:
            return self._dimension

    # 确保 MarkdownSplitter 类定义可用 (从上面复制过来或导入)
    # class MarkdownSplitter: ... (如上修改后的版本)

    try:
        # 替换为你的实际 EmbeddingService 和 Splitter 实现
        # embedding_srv = EmbeddingService(model_name="bge-base-en-v1.5") # 例如
        embedding_srv = MockEmbeddingService(model_name="bge-base-zh-v1.5") # 使用 Mock 服务进行测试
        
        # 使用修改后的 MarkdownSplitter
        splitter_instance = MarkdownSplitter(chunk_size=500, chunk_overlap=50) 

        # 2. 初始化 DataProcessor
        # 输出目录将是项目根目录下的 'project_root/data/processed_output/'
        # (假设此脚本在 project_root/src/knowledge_base/data_processor.py)
        project_root_dir = Path(__file__).resolve().parent.parent.parent 
        output_data_dir = project_root_dir / "data" / "processed_output"
        
        processor = DataProcessor(
            splitter=splitter_instance,
            embedding_service=embedding_srv,
            output_dir=str(output_data_dir) 
        )

        # 3. 定义输入Markdown文件的目录和可选的文档基础URL
        # 假设你的Markdown文件存放在 'project_root/data/raw_markdown_docs/'
        markdown_input_dir = project_root_dir / "data" / "raw_markdown_docs"
        
        # (可选) 为在线文档构建URL的基础部分
        # 例如，如果你的文档在 https://kubesphere.io/zh-cn/docs/v3.3/xxx.md
        # 则 base_url_for_docs 可以是 "https://kubesphere.io/zh-cn/docs/v3.3" (如果你的相对路径不包含版本)
        # 或者 "https://kubesphere.io" (如果你的相对路径包含 /zh-cn/docs/v3.3/)
        # 这里我们假设 markdown_input_dir 下的结构直接映射到 base_url 之后的部分
        docs_base_url = "https://kubesphere.io/docs" 

        # 为了测试，创建示例输入目录和文件 (如果不存在)
        markdown_input_dir.mkdir(parents=True, exist_ok=True)
        sample_md_file1 = markdown_input_dir / "intro.md"
        if not sample_md_file1.exists():
            with open(sample_md_file1, "w", encoding="utf-8") as f:
                f.write("# KubeSphere简介\n\nKubeSphere是一个分布式操作系统。\n\n## 核心功能\n\n- 多集群管理\n- DevOps\n```yaml\napiVersion: v1\nkind: Pod\nmetadata:\n  name: mypod\nspec:\n  containers:\n  - name: mycontainer\n    image: nginx\n```\n这是代码块后的文本。")
            logger.info(f"创建了示例文件: {sample_md_file1}")

        sample_md_sub_dir = markdown_input_dir / "zh-cn" / "v3.4"
        sample_md_sub_dir.mkdir(parents=True, exist_ok=True)
        sample_md_file2 = sample_md_sub_dir / "installation.md"
        if not sample_md_file2.exists():
             with open(sample_md_file2, "w", encoding="utf-8") as f:
                f.write("# 安装指南\n\n请参考官方文档进行安装。\n\n### 步骤一\n\n准备环境。\n\n```bash\necho 'Hello KubeSphere'\n```")
             logger.info(f"创建了示例文件: {sample_md_file2}")
        
        # 4. 执行处理流程
        logger.info(f"开始处理目录: {markdown_input_dir}")
        all_chunks = processor.process_directory(
            input_directory=str(markdown_input_dir),
            base_url_for_docs=docs_base_url
        )

        if all_chunks:
            logger.info(f"处理完成，共获得 {len(all_chunks)} 个文本块。现在开始生成嵌入并保存。")
            # 5. 生成嵌入并保存数据
            processor.save_data_with_embeddings(
                chunks_with_metadata=all_chunks,
                output_filename_prefix="ks_kb_data", # 输出文件会是 ks_kb_data_embeddings.jsonl
                embedding_batch_size=16 # 可以根据实际情况调整
            )
            logger.info(f"所有处理步骤完成。检查输出目录: {output_data_dir}")
        else:
            logger.warning("未能从目录中处理得到任何文本块。")

    except ImportError as e:
        logger.error(f"导入模块失败，请确保 EmbeddingService 和 MarkdownSplitter/DocumentSplitter 类已定义或正确安装: {e}")
    except Exception as e:
        logger.error(f"执行 DataProcessor 示例时发生错误: {e}", exc_info=True)

    finally:
        logger.info("脚本执行完毕。")