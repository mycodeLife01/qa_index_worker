from config import SystemConfig
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from utils import load_file
import chromadb
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
import time
from datetime import datetime, timezone
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.parsers import RapidOCRBlobParser


class IndexWorker:
    def __init__(self, config: SystemConfig):
        self.config = config
        self.embeddings = self._init_embeddings()
        self.chroma_client = self._init_chroma_client()
        self.vector_store = self._init_vector_store()

    def _init_embeddings(self):
        return HuggingFaceEndpointEmbeddings(
            model=self.config.model_config.embeddings.model,
            provider=self.config.model_config.embeddings.provider,
            huggingfacehub_api_token=self.config.secret_config.huggingfacehub_api_token,
        )

    def _init_chroma_client(self):
        """创建 ChromaDB HttpClient"""
        return chromadb.HttpClient(
            host=self.config.vdb_config.chroma_host,
            port=self.config.vdb_config.chroma_port,
        )

    def _init_vector_store(self):
        """使用 ChromaDB 服务器模式（HttpClient）"""
        return Chroma(
            collection_name=self.config.vdb_config.collection_name,
            embedding_function=self.embeddings,
            client=self.chroma_client,
        )

    async def _process_job(self, job_id: str, job_data: dict):
        # 从任务中拿到必要信息
        content_hash = job_data["content_hash"]
        file_url = job_data["file_url"]
        file_type = job_data["file_type"]

        print(f"[Job {job_id}] Processing content_hash: {content_hash}")

        # 创建loader并加载
        if file_type == "pdf":
            loader = PyPDFLoader(
                file_path=file_url,
                extract_images=True,  # 提取图片
                images_parser=RapidOCRBlobParser(),  # 使用 OCR 识别图片文字
                extraction_mode="layout",  # 保持布局（可选）
                images_inner_format="text",  # 文本格式输出
            )
        else:
            # 加载文件字节流到内存
            file_content_bytes = await load_file(file_url)
            loader = UnstructuredLoader(
                file=file_content_bytes, metadata_filename=content_hash
            )
        docs = await loader.aload()

        # 将文档分割为语意块
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50, add_start_index=True
        )
        all_splits = text_splitter.split_documents(docs)

        # 获取当前时间戳，用于标注
        print(f"[Job {job_id}] Annotating {len(all_splits)} chunks...")
        indexed_at_iso = datetime.now(timezone.utc).isoformat()
        for i, chunk in enumerate(all_splits):
            chunk.metadata["content_hash"] = content_hash
            chunk.metadata["indexed_at"] = indexed_at_iso
            chunk.metadata["original_url"] = file_url
            chunk.metadata["chunk_index"] = i  # 标注这是第几块

        filtered_splits = filter_complex_metadata(all_splits)

        # 添加数据到向量数据库
        print(f"[Job {job_id}] Adding {len(filtered_splits)} chunks to Vector Store...")
        self.vector_store.add_documents(filtered_splits)

    async def run(self, job_id: str, job_data: dict):
        try:
            await self._process_job(job_id, job_data)
        except Exception as e:
            print(f"[Run {job_id}] Error: {e}")
            raise e
