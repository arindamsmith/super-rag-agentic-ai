import logging
import os
import json
from google import genai
from google.genai import types
from infrastructure.llm_client import LLMClientProvider

logger = logging.getLogger("CitationAgent")


class CitationAgent:
    """
    Grounds each derived fact and the final conclusion
    in exact document sources (name + section/paragraph).
    """

    def __init__(self):
        self.client = LLMClientProvider.get_client()
        self.model = LLMClientProvider.get_analyst_model()

    async def run(self, state: dict) -> dict:
        request_id = state.get("request_id", "NA")

        analysis = state.get("analysis_json")
        cache_id = state.get("cache_id")
        big_context = state.get("big_context_fallback")

        if not analysis:
            logger.warning(f"[{request_id}] No analysis_json found. Skipping citation.")
            return state

        logger.info(f"[{request_id}] CitationAgent started using model {self.model}")

        citation_prompt = f"""
You are an Evidence Grounding Agent.

You are given:
1. A structured analysis JSON produced by another AI.
2. Full enterprise documents (in long context).

Your task:
For each key fact and for the final conclusion, identify:
- Document name
- Section / clause / paragraph reference
- Exact text span if possible

Return ONLY valid JSON in this format:

{{
  "citations": {{
     "derived_facts": {{
         "<fact_key>": {{
             "document": "...",
             "section": "...",
             "evidence": "..."
         }}
     }},
     "final_conclusion": {{
         "document": "...",
         "section": "...",
         "evidence": "..."
     }}
  }}
}}

Structured Analysis:
{json.dumps(analysis, indent=2)}
"""

        try:
            if cache_id:
                logger.info(f"[{request_id}] Using cached long-context for citation grounding")

                response = self.client.models.generate_content(
                    model=self.model,
                    contents=citation_prompt,
                    config=types.GenerateContentConfig(
                        cached_content=cache_id,
                        temperature=0.1
                    )
                )
            else:
                logger.info(f"[{request_id}] Using inline long-context fallback for citation grounding")

                full_prompt = f"""
Context Documents:
{big_context}

{citation_prompt}
"""

                response = self.client.models.generate_content(
                    model=self.model,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1
                    )
                )

            raw_text = response.text.strip()
            logger.info(f"[{request_id}] Citation raw response received {raw_text}")

            # Remove Markdown code fences if present
            if raw_text.startswith("```"):
                raw_text: str = raw_text.replace("```json", "").replace("```", "").strip()

            citation_json = json.loads(raw_text)
            state["citations"] = citation_json.get("citations", {})

            logger.info(f"[{request_id}] CitationAgent successfully grounded the answer")

            return state

        except Exception as e:
            logger.exception(f"[{request_id}] CitationAgent failed")
            state["error"] = str(e)
            return state
