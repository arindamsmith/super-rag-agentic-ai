import logging
from google.genai import types
import json
from infrastructure.llm_client import LLMClientProvider

logger = logging.getLogger("ResponseFormatterAgent")

class ResponseFormatterAgent:
    """
    Converts structured analysis + citations into
    a user-friendly, well-written, cited final answer.
    """

    def __init__(self):
        self.client = LLMClientProvider.get_client()
        self.model = LLMClientProvider.get_formatter_model()

    async def run(self, state: dict) -> dict:
        request_id = state.get("request_id", "NA")
        analysis = state.get("analysis_json")
        citations = state.get("citations")

        if not analysis:
            logger.warning(f"[{request_id}] No analysis_json found for formatting")
            return state

        logger.info(f"[{request_id}] ResponseFormatterAgent started using model {self.model}")

        prompt = f"""
You are an Enterprise Answer Generator.

Given:
1. Structured reasoning JSON
2. Evidence citations

Generate a clear, professional final answer for the user.
Include evidence references in brackets.

Structured Reasoning:
{json.dumps(analysis, indent=2)}

Citations:
{json.dumps(citations, indent=2)}

Produce:
- Final natural language answer
- With inline citations
"""

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2
                )
            )

            final_text = response.text.strip()
            logger.info(f"[{request_id}] Final answer: {final_text}")
            
            state["final_answer"] = final_text
            state["mode"] = "Super RAG (Agentic Long-Context Reasoning)"
            
            logger.info(f"[{request_id}] Final answer formatted successfully")
            return state

        except Exception as e:
            logger.exception(f"[{request_id}] ResponseFormatterAgent failed")
            state["error"] = str(e)
            return state
