import os
import logging
from typing import List, Dict
from PyPDF2 import PdfReader

logger = logging.getLogger("DocumentLoaderAgent")

class DocumentLoaderAgent:
    """
    Loads raw text from PDF and TXT files.
    """

    async def run(self, data_dir: str) -> List[Dict]:
        logger.info(f"Loading documents from directory: {data_dir}")

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        documents = []

        try:
            for filename in os.listdir(data_dir):
                file_path = os.path.join(data_dir, filename)

                if filename.lower().endswith(".txt"):
                    try:
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            text = f.read()
                            documents.append({"source": filename, "text": text})
                    except Exception as e:
                        logger.exception(f"Failed to read TXT file: {filename}")
                        raise

                elif filename.lower().endswith(".pdf"):
                    try:
                        reader = PdfReader(file_path)
                        pages = [page.extract_text() for page in reader.pages if page.extract_text()]
                        full_text = "\n".join(pages)
                        documents.append({"source": filename, "text": full_text})
                    except Exception as e:
                        logger.exception(f"Failed to read PDF file: {filename}")
                        raise

            logger.info(f"Loaded {len(documents)} documents successfully")
            return documents

        except Exception as e:
            logger.exception("Document loading failed")
            raise
