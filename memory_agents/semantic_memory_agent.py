import logging
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document

from infrastructure.qdrant_client import QdrantClientProvider
from infrastructure.embedding_client import EmbeddingClientProvider

logger = logging.getLogger("SemanticMemoryAgent")


class SemanticMemoryAgent:
    """
    Tier-1 semantic memory.
    Stores and retrieves past Q&A pairs using vector similarity.
    """

    def __init__(self, collection_name: str = "chat_history_cache"):
        # Use shared infrastructure (NO direct instantiation)
        self.collection_name = collection_name
        self.qdrant_client = QdrantClientProvider.get_client()
        self.embeddings = EmbeddingClientProvider.get_embeddings()

        # Ensure collection exists (important!)
        self._ensure_collection()

        # Vector store for semantic cache
        self.memory_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            embedding=self.embeddings
        )
    def _ensure_collection(self):
        try:
            existing = [c.name for c in self.qdrant_client.get_collections().collections]

            if self.collection_name not in existing:
                logger.info(f"Creating Qdrant collection for Semantic Memory: {self.collection_name}")
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
                )
        except Exception:
            logger.exception("Failed to initialize Semantic Memory collection")
            raise
    async def lookup(self, state: dict) -> dict:
        request_id = state.get("request_id", "NA")
        query = state["query"]

        logger.info(f"[{request_id}] SemanticMemory lookup started for: '{query}'")

        try:
            results = self.memory_store.similarity_search_with_score(query, k=1)

            if results:
                doc, score = results[0]
                logger.info(f"[{request_id}] Semantic candidate found with score {score:.4f}")

                if score > 0.75:  # similarity threshold
                    logger.info(f"[{request_id}] Semantic memory HIT")
                    state["semantic_hit"] = True
                    state["semantic_score"] = score
                    state["final_answer"] = doc.metadata.get("answer")
                    state["mode"] = "Semantic Memory (Tier-1)"
                    return state

            logger.info(f"[{request_id}] Semantic memory MISS")
            state["semantic_hit"] = False
            return state

        except Exception:
            logger.exception(f"[{request_id}] Semantic memory lookup failed")
            state["semantic_hit"] = False
            return state

    async def store(self, state: dict) -> dict:
        request_id = state.get("request_id", "NA")

        try:
            query = state["query"]
            answer = state.get("final_answer")

            if not answer:
                return state

            doc = Document(
                page_content=query,  # we vectorize the QUESTION
                metadata={
                    "answer": answer,
                    "mode": state.get("mode", "Super RAG")
                }
            )

            self.memory_store.add_documents([doc])

            logger.info(f"[{request_id}] Stored Q&A in Semantic Memory")

        except Exception:
            logger.exception(f"[{request_id}] Failed to store semantic memory")

        return state
