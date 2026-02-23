import logging
import os
from langchain_qdrant import QdrantVectorStore
from infrastructure.embedding_client import EmbeddingClientProvider
from infrastructure.qdrant_client import QdrantClientProvider

logger = logging.getLogger("DocumentHunterAgent")


class DocumentHunterAgent:
    """
    Retrieves full documents relevant to the planner output
    using semantic vector search + metadata filtering.
    """

    def __init__(self):
        self.embeddings = EmbeddingClientProvider.get_embeddings()
        self.qdrant_client = QdrantClientProvider.get_client()

        # Reuse the same collection as RAG
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name="enterprise_docs",
            embedding=self.embeddings
        )

        self.data_dir = "data"  # where original PDFs/TXTs exist

    async def run(self, state: dict) -> dict:
        request_id = state.get("request_id", "NA")

        logger.info(f"[{request_id}] DocumentHunterAgent started")

        query = state["query"]
        planner_hints = state.get("document_hints", [])
        entities = state.get("entities", [])

        logger.info(f"[{request_id}] Planner document hints: {planner_hints}")
        logger.info(f"[{request_id}] Planner entities: {entities}")

        try:
            # 1. Semantic search
            results = self.vector_store.similarity_search(query, k=10)

            # 2. Group by document source
            doc_names = set()
            for doc in results:
                if "source" in doc.metadata:
                    doc_names.add(doc.metadata["source"])

            logger.info(f"[{request_id}] Candidate documents from vector search: {list(doc_names)}")

            # 3. Load full documents from disk
            full_docs = []
            for doc_name in doc_names:
                full_path = os.path.join(self.data_dir, doc_name)
                if os.path.exists(full_path):
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()

                    full_docs.append({
                        "doc_name": doc_name,
                        "metadata": {
                            "source": doc_name
                        },
                        "full_text": text
                    })
                else:
                    logger.warning(f"[{request_id}] Document file not found on disk: {doc_name}")

            state["relevant_documents"] = full_docs

            logger.info(f"[{request_id}] Retrieved {len(full_docs)} full documents for deep reasoning")

            return state

        except Exception as e:
            logger.exception(f"[{request_id}] DocumentHunterAgent failed")
            state["error"] = "Document retrieval failed" + str(e)
            return state
