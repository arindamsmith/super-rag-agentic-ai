import logging
import json
from google.genai import types

from infrastructure.llm_client import LLMClientProvider

logger = logging.getLogger("RouterAgent")


class RouterAgent:
    """
    Decides whether a query requires:
    - SIMPLE_LOOKUP  → SimpleRAGAgent
    - COMPLEX_REASONING → Planner + SuperRAG pipeline
    """

    def __init__(self):
        self.llm = LLMClientProvider.get_client()
        self.model = LLMClientProvider.get_router_model()

    async def run(self, state: dict) -> dict:
        request_id = state.get("request_id", "NA")
        query = state["query"]

        logger.info(f"[{request_id}] RouterAgent evaluating query: {query}")

        prompt = f"""
You are a routing classifier for an enterprise AI system.

Classify the user query into one of two categories:

1. SIMPLE_LOOKUP:
   - Fact retrieval
   - Single document answers
   - Definitions
   - Straightforward questions

2. COMPLEX_REASONING:
   - Requires combining multiple documents
   - Requires comparison, policy interpretation, joins
   - Requires multi-step reasoning

Return ONLY valid JSON:
{{
  "intent": "SIMPLE_LOOKUP" or "COMPLEX_REASONING",
  "reason": "short explanation"
}}

User Query:
{query}
"""

        try:
            response = self.llm.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0
                )
            )

            raw = response.text.strip()
            logger.info(f"[{request_id}] Router raw response: {raw}")

            # Remove Markdown code fences if present
            if raw.startswith("```"):
                raw: str = raw.replace("```json", "").replace("```", "").strip()

            result = json.loads(raw)
            intent = result.get("intent", "SIMPLE_LOOKUP")
            reason = result.get("reason", "")

            state["intent"] = intent
            state["routing_reason"] = reason

            logger.info(f"[{request_id}] Router decision: {intent} ({reason})")

            return state

        except Exception as e:
            # Fallback: rule-based heuristic
            logger.exception(f"[{request_id}] Router LLM failed, using fallback rules")

            complex_keywords = [
                "compare", "difference", "policy", "eligibility", "can i",
                "allowed", "not allowed", "across", "between", "combine",
                "rule", "regulation", "clause", "contract", "implication"
            ]

            if any(k in query.lower() for k in complex_keywords):
                state["intent"] = "COMPLEX_REASONING"
                state["routing_reason"] = "Keyword-based fallback detection"
            else:
                state["intent"] = "SIMPLE_LOOKUP"
                state["routing_reason"] = "Keyword-based fallback detection"

            logger.info(f"[{request_id}] Router fallback decision: {state['intent']}")
            return state
