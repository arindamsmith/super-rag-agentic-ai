import logging
import json
from google.genai import types
from infrastructure.llm_client import LLMClientProvider

logger = logging.getLogger("AnalystAgent")

class AnalystAgent:
    """
    Performs deep multi-document reasoning using Gemini-Pro.
    Consumes:
      - reasoning plan
      - entities
      - required attributes
      - long-context (cache or fallback)
    Produces:
      - structured JSON analysis
    """

    def __init__(self):
        self.client = LLMClientProvider.get_client()
        self.model = LLMClientProvider.get_analyst_model()

    async def run(self, state: dict) -> dict:
        request_id = state.get("request_id", "NA")

        logger.info(f"[{request_id}] AnalystAgent started using model {self.model}")

        entities = state.get("entities", [])
        attributes = state.get("required_attributes", [])
        plan_steps = state.get("plan_steps", [])
        cache_id = state.get("cache_id")
        big_context = state.get("big_context_fallback")

        analysis_prompt = f"""
You are a senior enterprise analyst AI.

You have access to full internal documents.
You must reason strictly from them.

User Entities:
{entities}

Information to Derive:
{attributes}

Reasoning Plan:
{plan_steps}

Instructions:
1. Execute the reasoning plan step by step.
2. Join information across documents if needed.
3. Resolve conflicts using policy precedence if any.
4. Do not hallucinate. Use only the provided documents.
5. Return ONLY valid JSON in the following format:

{{
  "entities": {entities},
  "derived_facts": {{ }},
  "analysis": "step-by-step reasoning",
  "final_conclusion": "clear answer to the user",
  "confidence": 0.0
}}
"""

        try:
            if cache_id:
                logger.info(f"[{request_id}] Using Gemini Cached Content: {cache_id}")

                response = self.client.models.generate_content(
                    model=self.model,
                    contents=analysis_prompt,
                    config=types.GenerateContentConfig(
                        cached_content=cache_id,
                        temperature=0.1
                    )
                )
            else:
                logger.info(f"[{request_id}] Using inline long-context fallback")

                full_prompt = f"""
                Context Documents:
                {big_context}

                {analysis_prompt}
                """

                response = self.client.models.generate_content(
                    model=self.model,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1
                    )
                )

            raw_text = response.text.strip()
            logger.info(f"[{request_id}] Analyst raw response received: {raw_text}")

            # Remove Markdown code fences if present
            if raw_text.startswith("```"):
                raw_text: str = raw_text.replace("```json", "").replace("```", "").strip()

            # Parse structured JSON
            analysis_json = json.loads(raw_text)

            state["analysis_json"] = analysis_json
            state["final_answer"] = analysis_json.get("final_conclusion")
            state["mode"] = "Super RAG (Long-Context Agentic Reasoning)"

            logger.info(f"[{request_id}] Analyst final conclusion: {analysis_json.get('final_conclusion')}")

            logger.info(f"[{request_id}] Analyst completed reasoning successfully")

            return state

        except Exception as e:
            logger.exception(f"[{request_id}] AnalystAgent failed \n with error: {str(e)}")
            state["error"] = str(e)
            return state
