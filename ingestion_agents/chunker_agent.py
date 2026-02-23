import logging
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger("ChunkerAgent")

class ChunkerAgent:
    """
    Splits documents into overlapping semantic chunks.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    async def run(self, documents: List[Dict]) -> List[Dict]:
        logger.info("Chunking documents")

        if not documents:
            raise ValueError("No documents provided to ChunkerAgent")

        try:
            chunks = []
            for doc in documents:
                splits = self.splitter.split_text(doc["text"])
                for idx, chunk in enumerate(splits):
                    chunks.append({
                        "source": doc["source"],
                        "chunk_id": idx,
                        "text": chunk
                    })

            logger.info(f"Created {len(chunks)} chunks")
            return chunks

        except Exception as e:
            logger.exception("Chunking failed")
            raise
