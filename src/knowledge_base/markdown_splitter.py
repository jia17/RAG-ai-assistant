"""
文档切分模块 - 负责将文档切分为适合嵌入的文本块
"""

from typing import List, Dict, Any, Optional
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
import re
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class MarkdownSplitter:
    """文档切分器，专注于处理Markdown文档"""
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """初始化文档切分器
        
        Args:
            chunk_size: 文档切分大小
            chunk_overlap: 文档切分重叠大小
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 预定义Markdown标题级别
        self.md_headers_to_split_on = [
            ("#", "header1"),
            ("##", "header2"),
            ("###", "header3"),
            ("####", "header4"),
        ]
        

                # LangChain splitters
        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.md_headers_to_split_on,
            return_each_line=False
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""], # Common separators
            length_function=len
        )
        
        # 代码语言正则表达式
        # self.code_lang_pattern = re.compile(r"```(\w+)")
        self.code_block_start_pattern = re.compile(r"^\s*```(\w*)\s*$") # Matches ```lang
        logger.info(f"DocumentSplitter initialized with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    def _analyze_chunk_for_code(self, text_chunk: str) -> tuple[str, bool]:
        """
        分析单个文本块，判断其主要代码语言以及是否包含大量代码。
        这是一个简化的分析逻辑。

        Args:
            text_chunk (str): 需要分析的文本块。

        Returns:
            tuple: (主要代码语言: str, 是否代码密集: bool)
                如果未明确识别出语言，则语言为 "none"。
        """
        lines = text_chunk.split('\n')
        detected_code_language = "none"
        is_code_heavy_chunk = False
        in_fenced_code_block = False # 标记当前是否在 ``` ... ``` 块内部

        for line in lines:
            stripped_line = line.strip()
            # 检查是否是代码块的开始或结束标记
            if stripped_line.startswith("```"):
                is_code_heavy_chunk = True # 只要出现 ``` 就认为是代码相关的块
                if not in_fenced_code_block: # 如果当前不在代码块内，则这是开始标记
                    in_fenced_code_block = True
                    match = self.code_block_start_pattern.match(stripped_line) # 尝试从行首匹配 ```lang
                    if match:
                        lang = match.group(1)
                        if lang: # 如果 ```lang 中的 lang 不为空
                            detected_code_language = lang.lower() # 统一小写
                            # 简单策略：一旦检测到语言，就用第一个。更复杂可以统计频率。
                else: # 如果当前在代码块内，则这是结束标记
                    in_fenced_code_block = False
            elif in_fenced_code_block: # 当前行在代码块内部
                is_code_heavy_chunk = True # 再次确认是代码密集型
        
        # 如果整个块都在一个未闭合的```内开始，也认为是代码
        if in_fenced_code_block and detected_code_language == "none" and is_code_heavy_chunk:
            pass


        return detected_code_language, is_code_heavy_chunk

    def split_markdown(self, markdown_text: str, base_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        切分Markdown文档。

        Args:
            markdown_text (str): Markdown文本内容。
            base_metadata (Dict[str, Any], optional): 附加到每个块的基础元数据。默认为 None。

        Returns:
            List[Dict[str, Any]]: 包含 "chunk_text" 和 "metadata" 的字典列表。
        """
        if base_metadata is None:
            base_metadata = {}
        
        final_chunks = []
        # 用于生成在本文档内唯一的块ID后缀
        chunk_index_in_doc_counter = 0 
        
        try:
            # 1. 基于Markdown标题进行初步切分
            header_splits = self.md_splitter.split_text(markdown_text)
            
            for header_split_doc in header_splits:
                text_from_header_split = header_split_doc.page_content
                # MarkdownHeaderTextSplitter 提取的元数据 (例如 {'header1': '标题内容'})
                metadata_from_header = header_split_doc.metadata 
                
                # 2. 如果基于标题切分后的块仍然过大，则使用RecursiveCharacterTextSplitter进行二次切分
                if len(text_from_header_split) > self.chunk_size:
                    sub_chunks_texts = self.text_splitter.split_text(text_from_header_split)
                else:
                    # 如果块不大，则将其视为单个子块处理
                    sub_chunks_texts = [text_from_header_split]
                
                for current_chunk_text in sub_chunks_texts:
                    if not current_chunk_text.strip(): # 跳过完全是空白的块
                        continue
                        
                    # 3. 分析当前子块的代码信息
                    code_language, is_code_heavy = self._analyze_chunk_for_code(current_chunk_text)
                    
                    # 4. 构建当前块的元数据
                    current_chunk_metadata = {
                        **base_metadata,             # 合并基础元数据
                        **metadata_from_header,      # 合并标题元数据
                        "chunk_id_within_doc": chunk_index_in_doc_counter, # 文档内块的索引
                        "code_language": code_language,
                        "is_code_heavy": is_code_heavy
                    }
                    
                    final_chunks.append({
                        "chunk_text": current_chunk_text,
                        "metadata": current_chunk_metadata
                    })
                    chunk_index_in_doc_counter += 1
            
            logger.debug(f"Markdown文档 (源: {base_metadata.get('source_url', '未知')}) 被切分为 {len(final_chunks)} 个块。")
            return final_chunks
            
        except Exception as e:
            logger.error(f"切分Markdown文档失败 (源: {base_metadata.get('source_url', '未知')}): {e}", exc_info=True)
            raise 
    

