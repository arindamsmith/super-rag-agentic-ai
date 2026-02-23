import logging
from google.genai import types
from infrastructure.llm_client import LLMClientProvider

logger = logging.getLogger("LongContextLoaderAgent")

class LongContextLoaderAgent:
    """
    Loads full relevant documents into Gemini Cached Content (long context).
    Falls back to inline context if caching fails.
    """

    def __init__(self):
        self.client = LLMClientProvider.get_client()
        self.model = LLMClientProvider.get_analyst_model()

    async def run(self, state: dict) -> dict:
        request_id = state.get("request_id", "NA")
        documents = state.get("relevant_documents", [])

        logger.info(f"[{request_id}] LongContextLoaderAgent started")
        logger.info(f"[{request_id}] Number of documents to load: {len(documents)}")

        if not documents:
            logger.warning(f"[{request_id}] No documents provided for long-context loading")
            return state

        # Combine all document texts with separators
        combined_text = ""
        for doc in documents:
            combined_text += f"\n\n--- Document: {doc['doc_name']} ---\n"
            combined_text += doc["full_text"]

        try:
            logger.info(f"[{request_id}] Attempting Gemini Cached Content creation")

            cache_result = self.client.caches.create(
                model=self.model,
                config=types.CreateCachedContentConfig(
                    display_name=f"super_rag_cache_{request_id}",
                    system_instruction=(
                        "You are an enterprise reasoning engine. "
                        "Answer strictly based on the provided documents. "
                        "Perform deep cross-document analysis and return structured JSON."
                    ),
                    contents=[combined_text],
                    ttl="3600s"
                )
            )

            state["cache_id"] = cache_result.name
            state["big_context_fallback"] = None

            logger.info(f"[{request_id}] Long context cached successfully. Cache ID: {cache_result.name}")

        except Exception as e:
            # Fallback: inline long context
            logger.error(f"[{request_id}] Cache creation failed. Falling back to inline context.")
            logger.exception(e)

            state["cache_id"] = None
            state["big_context_fallback"] = combined_text

            logger.info(f"[{request_id}] Stored combined documents in big_context_fallback for direct prompting")

        return state
