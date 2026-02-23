import logging
from typing import List, Dict
from infrastructure.embedding_client import EmbeddingClientProvider

logger = logging.getLogger("EmbeddingAgent")

class EmbeddingAgent:
    """
    Converts chunks into vector embeddings.
    """

    async def run(self, chunks: List[Dict]) -> List[Dict]:
        logger.info("Generating embeddings")

        if not chunks:
            raise ValueError("No chunks provided to EmbeddingAgent")

        try:
            embeddings_model = EmbeddingClientProvider.get_embeddings()
            texts = [c["text"] for c in chunks]

            vectors = embeddings_model.embed_documents(texts)

            if len(vectors) != len(chunks):
                raise RuntimeError("Embedding count mismatch")

            for i, vec in enumerate(vectors):
                chunks[i]["vector"] = vec

            logger.info("Embeddings generated successfully")
            return chunks

        except Exception as e:
            logger.exception("Embedding generation failed")
            raise
