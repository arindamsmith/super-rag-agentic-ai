import os
import logging
from qdrant_client import QdrantClient

logger = logging.getLogger("QdrantClient")

class QdrantClientProvider:
    """
    Centralized Qdrant client.
    All agents (ingestion, retrieval, memory) will reuse this.
    """

    _client = None

    @classmethod
    def get_client(cls) -> QdrantClient:
        if cls._client is None:
            qdrant_path = os.getenv("QDRANT_PATH", "./qdrant_storage")
            logger.info(f"Initializing Qdrant at path: {qdrant_path}")

            cls._client = QdrantClient(path=qdrant_path)

        return cls._client
