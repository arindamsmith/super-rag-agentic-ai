import logging
from ingestion_agents.document_loader_agent import DocumentLoaderAgent
from ingestion_agents.chunker_agent import ChunkerAgent
from ingestion_agents.embedding_agent import EmbeddingAgent
from ingestion_agents.vector_store_agent import VectorStoreAgent

logger = logging.getLogger("IngestionOrchestrator")

class IngestionOrchestrator:

    def __init__(self):
        self.loader = DocumentLoaderAgent()
        self.chunker = ChunkerAgent()
        self.embedder = EmbeddingAgent()
        self.vector_store = VectorStoreAgent()

    async def ingest(self, data_dir: str):
        logger.info("Starting ingestion pipeline")

        try:
            # Ingestion pipeline: Load, Chunk, Embed, Store
            documents = await self.loader.run(data_dir)
            chunks = await self.chunker.run(documents)
            embedded_chunks = await self.embedder.run(chunks)
            await self.vector_store.run(embedded_chunks)

            logger.info("Ingestion pipeline completed successfully")

        except Exception as e:
            logger.exception("Ingestion pipeline failed")
            raise
