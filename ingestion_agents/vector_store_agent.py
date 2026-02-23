import logging
from typing import List, Dict
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document

from infrastructure.qdrant_client import QdrantClientProvider
from infrastructure.embedding_client import EmbeddingClientProvider

logger = logging.getLogger("VectorStoreAgent")

class VectorStoreAgent:
    """
    Stores embeddings in Qdrant.
    """

    def __init__(self, collection_name: str = "enterprise_docs"):
        self.collection_name = collection_name
        self.client = QdrantClientProvider.get_client()
        self.embeddings = EmbeddingClientProvider.get_embeddings()
        self._ensure_collection()

        self.store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings
        )

    def _ensure_collection(self):
        try:
            existing = [c.name for c in self.client.get_collections().collections]
            if self.collection_name not in existing:
                logger.info(f"Creating Qdrant collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
                )
        except Exception as e:
            logger.exception("Failed to initialize Qdrant collection")
            raise

    async def run(self, chunks: List[Dict]):
        logger.info("Storing chunks in Qdrant")

        if not chunks:
            raise ValueError("No chunks provided to VectorStoreAgent")

        try:
            docs = [
                Document(
                    page_content=c["text"],
                    metadata={"source": c["source"], "chunk_id": c["chunk_id"]}
                )
                for c in chunks
            ]

            self.store.add_documents(docs)
            logger.info("Chunks successfully stored in vector DB")

        except Exception as e:
            logger.exception("Failed to store vectors in Qdrant")
            raise
