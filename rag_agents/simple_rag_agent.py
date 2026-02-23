import logging
import time
from typing import List

from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document

from infrastructure.qdrant_client import QdrantClientProvider
from infrastructure.embedding_client import EmbeddingClientProvider
from infrastructure.llm_client import LLMClientProvider
from google.genai import types

logger = logging.getLogger("SimpleRAGAgent")


class SimpleRAGAgent:
    """
    Tier-2 RAG Agent.
    Handles straightforward factual queries using vector retrieval + LLM.
    """

    def __init__(self, collection_name: str = "enterprise_docs"):
        self.client = QdrantClientProvider.get_client()
        self.embeddings = EmbeddingClientProvider.get_embeddings()
        self.llm = LLMClientProvider.get_client()
        self.model = LLMClientProvider.get_formatter_model()

        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=self.embeddings
        )

    async def run(self, state: dict) -> dict:
        request_id = state.get("request_id", "NA")
        query = state["query"]

        logger.info(f"[{request_id}] SimpleRAGAgent started")
        start_time = time.time()

        try:
            # 1. Retrieve relevant chunks
            docs_with_scores = self.vector_store.similarity_search_with_score(query, k=5)

            if not docs_with_scores:
                logger.warning(f"[{request_id}] No relevant documents found in vector store")
                state["final_answer"] = "No relevant information found in the knowledge base."
                state["sources"] = []
                state["mode"] = "Simple RAG (No Hit)"
                return state

            # 2. Prepare context and sources
            context_parts: List[str] = []
            sources = set()

            for doc, score in docs_with_scores:
                context_parts.append(doc.page_content)
                if "source" in doc.metadata:
                    sources.add(doc.metadata["source"])

            context = "\n\n".join(context_parts)

            prompt = f"""
You are a factual enterprise assistant.
Answer strictly based on the provided context.

Context:
{context}

Question:
{query}

Instructions:
- Be concise and accurate.
- Do not hallucinate.
- If the answer is not in the context, say so.
"""

            # 3. Call LLM
            response = self.llm.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.1)
            )

            answer = response.text.strip()

            latency = round(time.time() - start_time, 2)

            # 4. Populate state
            state["final_answer"] = answer
            state["sources"] = list(sources)
            state["mode"] = "Simple RAG (Vector Search)"
            state["latency_seconds"] = latency

            logger.info(f"[{request_id}] SimpleRAGAgent completed in {latency}s")
            logger.info(f"[{request_id}] Sources: {state['sources']}")

            return state

        except Exception as e:
            logger.exception(f"[{request_id}] SimpleRAGAgent failed")
            state["error"] = str(e)
            state["mode"] = "Simple RAG (Error)"
            return state
