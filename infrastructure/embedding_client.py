import os
import logging
from langchain_google_genai import GoogleGenerativeAIEmbeddings

logger = logging.getLogger("EmbeddingClient")

class EmbeddingClientProvider:
    """
    Centralized embedding model access.
    Used by:
    - Ingestion (to embed chunks)
    - Retrieval (to embed queries)
    - Semantic Memory
    """

    _embeddings = None

    @classmethod
    def get_embeddings(cls) -> GoogleGenerativeAIEmbeddings:
        if cls._embeddings is None:
            api_key = os.getenv("API_KEY")
            model = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")

            logger.info(f"Initializing Embedding Model: {model} with api_key={api_key[:10]}***")

            cls._embeddings = GoogleGenerativeAIEmbeddings(
                model=model,
                google_api_key=api_key
            )

        return cls._embeddings
