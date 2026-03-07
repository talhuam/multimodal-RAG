import base64
import io
from typing import List
import hashlib
import re

from PIL import Image

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from utils.os_utils import get_sorted_md_files

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


class MarkdownDirSplitter:
    """
    切割markdown文件夹 -> chunk
    抽取出markdown中的base64格式图片保存为本地图片，同时移除markdown中的base64字符串
    """
    def __init__(self, images_output_dir, text_chunk_size=1000):
        self.images_output_dir = images_output_dir
        self.text_chunk_size = text_chunk_size
        os.makedirs(images_output_dir, exist_ok=True)

        # 按三级标题来切割markdown
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        self.text_splitter = MarkdownHeaderTextSplitter(self.headers_to_split_on)

        # 语义相似性切割，原理：计算两两句子的相似性，如果相似性低于某个阈值，则拆分
        self.embeddings = OpenAIEmbeddings(
            model="Qwen3-Embedding",
            base_url="http://localhost:8000/v1",
            api_key="empty"
        )
        self.semantic_splitter = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type="percentile"
        )

    def save_base64_to_image(self, base64_str, output_path):
        """
        将base64字符串解码为图像并保存
        base64格式形如：![](data:image/png;base64,xxxxxxx)
        Args:
            base64_str (str): base64字符串
            output_path (str): 图片保存目录
        """
        if base64_str.startswith("data:image"):
            base64_str = base64_str.split(",", 1)[1]
        img_data = base64.b64decode(base64_str)
        with Image.open(io.BytesIO(img_data)) as img:
            img.save(output_path)

    def process_images(self, content, source) -> List[Document]:
        """
        处理Markdown中的base64图片
        Args:
            content (str): text split后的文本内容
            source (str): 所属文件路径
        """
        image_docs = []
        pattern = r'data:image/(.*?);base64,(.*?)\)'  # 正则匹配base64图片

        def replace_image(match):
            img_type = match.group(1).split(';')[0]
            base64_data = match.group(2)

            # 生成唯一文件名
            hash_key = hashlib.md5(base64_data.encode()).hexdigest()
            filename = f"{hash_key}.{img_type if img_type in ['png', 'jpg', 'jpeg'] else 'png'}"
            img_path = os.path.join(self.images_output_dir, filename)

            # 保存图片
            self.save_base64_to_image(base64_data, img_path)

            # 创建图片Document
            image_docs.append(Document(
                page_content=str(img_path),
                metadata={
                    "source": source,
                    "alt_text": "图片",
                    "embedding_type": "image"
                }
            ))

            return "[图片]"

        # 替换所有base64图片
        content = re.sub(pattern, replace_image, content, flags=re.DOTALL)
        return image_docs

    def process_md_file(self, md_file: str) -> List[Document]:
        """
        单独处理一个md文件
        Args:
            md_file (str): markdown文件路径
        """
        with open(md_file, "r", encoding="utf-8") as file:
            content = file.read()

        # 分割Markdown内容
        split_documents: List[Document] = self.text_splitter.split_text(content)
        documents = []
        for doc in split_documents:
            # 处理图片
            if '![](data:image/png;base64' in doc.page_content:
                image_docs = self.process_images(doc.page_content, md_file)
                # 移除图片之后，还有剩下的文本内容
                cleaned_content = self.remove_base64_images(doc.page_content)
                if cleaned_content.strip():
                    doc.metadata['embedding_type'] = 'text'
                    documents.append(Document(page_content=cleaned_content, metadata=doc.metadata))
                documents.extend(image_docs)

            else:
                doc.metadata['embedding_type'] = 'text'
                documents.append(doc)

        # 语义分割
        final_docs = []
        for d in documents:
            if len(d.page_content) > self.text_chunk_size:
                final_docs.extend(self.semantic_splitter.split_documents([d]))
            else:
                final_docs.append(d)

        # 添加标题层级
        return final_docs

    def remove_base64_images(self, text: str) -> str:
        """
        移除所有Base64图片标记
        """
        pattern = r'!\[\]\(data:image/(.*?);base64,(.*?)\)'
        return re.sub(pattern, '', text)

    def add_title_hierarchy(self, documents: List[Document], source_filename: str) -> List[Document]:
        """
        为文档添加标题层级结构，实现段落即使跨页也嫩打上相同的层级
        """
        current_titles = {1: "", 2: "", 3: ""}
        processed_docs = []

        for doc in documents:
            new_metadata = doc.metadata.copy()
            new_metadata['source'] = source_filename

            # 更新标题状态
            for level in range(1, 4):
                header_key = f'Header {level}'
                if header_key in new_metadata:
                    current_titles[level] = new_metadata[header_key]
                    for lower_level in range(level + 1, 4):
                        current_titles[lower_level] = ""

            # 补充缺失的标题
            for level in range(1, 4):
                header_key = f'Header {level}'
                if header_key not in new_metadata:
                    new_metadata[header_key] = current_titles[level]
                elif current_titles[level] != new_metadata[header_key]:
                    new_metadata[header_key] = current_titles[level]

            processed_docs.append(
                Document(
                    page_content=doc.page_content,
                    metadata=new_metadata
                )
            )

        return processed_docs

    def process_md_dir(self, md_dir: str, source_filename: str) -> List[Document]:
        """
        指定一个md文件目录，切割里面的所有数据
        Args:
            md_dir: ocr解析后的文件夹
            source_filename: 原始pdf文件名
        """
        md_files = get_sorted_md_files(md_dir)
        all_documents = []
        for md_file in md_files:
            all_documents.extend(self.process_md_file(md_file))
        # 添加标题层级
        return self.add_title_hierarchy(all_documents, source_filename)


if __name__ == '__main__':
    splitter = MarkdownDirSplitter(images_output_dir="../data/images")
    result = splitter.process_md_dir(md_dir="../data/Attention Is All You Need", source_filename="Attention Is All You Need.pdf")
    print(result)