import logging
from typing import List, Dict
from llama_index.core import Document
from modules.vector_store import VectorStore
from modules.token_counter import TokenCounter

logger = logging.getLogger(__name__)

class ContentIndexer:
    def __init__(self, token_counter: TokenCounter):
        self.vector_store = VectorStore()
        self.token_counter = token_counter

    def create_index(self, documents: List[Dict]) -> None:
        logger.info(f'Creating index from {len(documents)} documents')
        llama_docs = [Document(text=doc['content'], metadata={'title': doc['title'], 'url': doc['url']}) for doc in documents]
        self.vector_store.add_documents(llama_docs)
        for doc in documents:
            self.token_counter.add_embedding_tokens(self.token_counter.count_tokens(doc['content']))

    def add_document_to_index(self, document: Dict) -> None:
        logger.info('Adding single document to index')
        llama_doc = Document(text=document['content'], metadata={'title': document['title'], 'url': document['url']})
        self.vector_store.add_documents([llama_doc])
        self.token_counter.add_embedding_tokens(self.token_counter.count_tokens(document['content']))

    def search_index(self, query: str, top_k: int = 5):
        return self.vector_store.search(query, top_k)

    def clear_index(self) -> None:
        logger.info('Clearing index')
        self.vector_store.clear()

def create_index(documents: List[Dict], token_counter: TokenCounter) -> ContentIndexer:
    indexer = ContentIndexer(token_counter)
    indexer.create_index(documents)
    return indexer

def load_index(token_counter: TokenCounter) -> ContentIndexer:
    logger.info('Loading existing index')
    return ContentIndexer(token_counter)