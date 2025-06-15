import os
import json
import hashlib
import uuid
from datetime import datetime
from typing import List, Dict
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


class FileManager:
    """文件管理器 - 集成文档处理、索引管理和向量数据库"""

    def __init__(self, persist_directory="./chroma_db", chunk_size=512, chunk_overlap=50, folder_path=None):
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 初始化嵌入模型
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        # 初始化向量数据库
        self.docsearch = self._initialize_vectorstore()

        # 映射文件路径
        self.mapping_file = os.path.join(persist_directory, "doc_vector_mapping.json")
        self.index_file = os.path.join(persist_directory, "document_index.json")

        # 文本分割器
        self.text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # 加载映射和索引
        self.doc_vector_mapping = self._load_mapping()
        self.document_index = self._load_document_index()

        # 初始化文件夹（如果提供）
        if folder_path:
            file_list = []
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_list.append(file_path)
            self.add_documents(file_list)

    def _initialize_vectorstore(self):
        """初始化向量数据库"""
        if os.path.exists(self.persist_directory):
            return Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        else:
            os.makedirs(self.persist_directory, exist_ok=True)
            return Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )

    def has_documents(self) -> bool:
        """检查是否有文档"""
        try:
            return len(self.docsearch.get()['ids']) > 0
        except:
            return False

    def get_retriever(self, k: int = 5):
        """获取检索器"""
        return self.docsearch.as_retriever(search_kwargs={"k": k})

    def _load_mapping(self) -> Dict:
        """加载文档到向量ID映射"""
        try:
            if os.path.exists(self.mapping_file):
                with open(self.mapping_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading mapping file: {e}")
        return {}

    def _save_mapping(self):
        """保存映射关系"""
        try:
            os.makedirs(os.path.dirname(self.mapping_file), exist_ok=True)
            with open(self.mapping_file, 'w', encoding='utf-8') as f:
                json.dump(self.doc_vector_mapping, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving mapping file: {e}")

    def _load_document_index(self) -> Dict:
        """加载文档索引"""
        try:
            if os.path.exists(self.index_file):
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading index file: {e}")
        return {}

    def _save_document_index(self):
        """保存文档索引"""
        try:
            os.makedirs(os.path.dirname(self.index_file), exist_ok=True)
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(self.document_index, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving index file: {e}")

    def _get_file_hash(self, file_path: str) -> str:
        """获取文件哈希"""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def _load_document(self, file_path: str):
        """根据文件类型加载文档"""
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
        return loader.load()

    def add_documents(self, file_paths: List[str]) -> Dict:
        """添加文档"""
        results = {
            "added": [],
            "skipped": [],
            "errors": []
        }

        for file_path in file_paths:
            try:
                if not os.path.exists(file_path):
                    results["errors"].append(f"文件未找到: {file_path}")
                    continue

                file_name = os.path.basename(file_path)
                file_hash = self._get_file_hash(file_path)

                # 检查是否需要更新
                if file_name in self.document_index:
                    if self.document_index[file_name]["hash"] == file_hash:
                        results["skipped"].append(f"文件未改变: {file_name}")
                        continue
                    else:
                        # 文件已修改，删除旧版本
                        self.remove_document(file_name)

                # 加载和分割文档
                documents = self._load_document(file_path)

                # 生成唯一ID并添加元数据
                document_ids = []
                texts_with_ids = []

                for doc in documents:
                    doc.metadata["source_file"] = file_name

                texts = self.text_splitter.split_documents(documents)

                for text in texts:
                    doc_id = str(uuid.uuid4())
                    text.metadata["doc_id"] = doc_id
                    document_ids.append(doc_id)
                    texts_with_ids.append(text)

                if texts_with_ids:
                    # 直接添加到向量数据库
                    self.docsearch.add_documents(texts_with_ids, ids=document_ids)

                    # 记录映射关系
                    self.doc_vector_mapping[file_name] = document_ids

                    # 更新文档索引
                    self.document_index[file_name] = {
                        "path": file_path,
                        "hash": file_hash,
                        "chunks": len(texts_with_ids),
                        "vector_ids": document_ids,
                        "added_time": datetime.now().isoformat()
                    }

                    results["added"].append(f"成功添加: {file_name} ({len(texts_with_ids)} 个片段)")

            except UnicodeDecodeError as e:
                results["errors"].append(f"编码错误 - 文件 {file_path}: {str(e)}")
            except Exception as e:
                results["errors"].append(f"处理文件错误 {file_path}: {str(e)}")

        # 保存状态
        self._save_mapping()
        self._save_document_index()

        return results

    def remove_document(self, file_name: str) -> bool:
        """删除文档"""
        try:
            if file_name not in self.doc_vector_mapping:
                return False

            # 获取该文档的所有向量ID
            vector_ids = self.doc_vector_mapping[file_name]

            # 从向量库删除
            if vector_ids:
                self.docsearch.delete(ids=vector_ids)

            # 清理映射和索引
            del self.doc_vector_mapping[file_name]
            if file_name in self.document_index:
                del self.document_index[file_name]

            # 保存状态
            self._save_mapping()
            self._save_document_index()

            return True

        except Exception as e:
            print(f"删除文档错误: {str(e)}")
            return False

    def list_documents(self) -> Dict:
        """列出所有文档"""
        return self.document_index

    def get_stats(self) -> Dict:
        """获取统计信息"""
        total_documents = len(self.document_index)
        total_chunks = sum(doc.get("chunks", 0) for doc in self.document_index.values())

        return {
            "total_documents": total_documents,
            "total_chunks": total_chunks,
            "has_documents": total_chunks > 0
        }