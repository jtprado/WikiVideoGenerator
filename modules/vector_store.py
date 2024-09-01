from typing import List
from qdrant_client import QdrantClient
from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex
import config

class VectorStore:
    def __init__(self):
        self.client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=config.QDRANT_COLLECTION_NAME
        )
        self.pipeline = self._create_ingestion_pipeline()
        self.index = None

    def _create_ingestion_pipeline(self):
        embed_model = HuggingFaceEmbedding(model_name=config.HUGGINGFACE_MODEL_NAME)
        return IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=512, chunk_overlap=20),
                TitleExtractor(),
                embed_model,
            ],
            vector_store=self.vector_store,
        )

    def add_documents(self, documents: List[Document]):
        self.pipeline.run(documents=documents)
        self.index = VectorStoreIndex.from_vector_store(self.vector_store)

    def search(self, query: str, top_k: int = 5):
        if not self.index:
            raise ValueError("Index not created. Please add documents first.")
        return self.index.as_query_engine().query(query)

    def clear(self):
        self.client.delete_collection(config.QDRANT_COLLECTION_NAME)
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=config.QDRANT_COLLECTION_NAME
        )
        self.pipeline = self._create_ingestion_pipeline()
        self.index = None