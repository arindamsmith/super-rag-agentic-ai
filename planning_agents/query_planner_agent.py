import logging

from google.genai import types
import json
from infrastructure.llm_client import LLMClientProvider

logger = logging.getLogger("QueryPlannerAgent")

class QueryPlannerAgent:
    """
    LLM-based planner that converts any user query into:
    - Entities
    - Required information
    - Reasoning steps
    - Retrieval hints
    """

    def __init__(self):
        self.client = LLMClientProvider.get_client()
        self.model = LLMClientProvider.get_planner_model()

    async def run(self, state: dict) -> dict:
        request_id = state.get("request_id", "NA")
        query = state["query"]

        logger.info(f"[{request_id}] Planner (LLM) analyzing query")

        prompt = f"""
You are a Query Planning Agent for an Enterprise Super RAG system.

Your job:
Given a user question, produce a structured reasoning plan that will later be
executed by retrieval and analysis agents.

Return ONLY valid JSON with the following fields:

{{
  "entities": [list of important entities or concepts],
  "required_attributes": [what needs to be derived or looked up],
  "document_hints": [types or domains of documents likely needed],
  "reasoning_steps": [ordered steps to answer the question]
}}

User Question:
{query}
"""

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2
                )
            )

            plan_json = response.text
            logger.info(f"[{request_id}] Planner raw output: {plan_json}")

            # Remove Markdown code fences if present
            if plan_json.startswith("```"):
                plan_json: str = plan_json.replace("```json", "").replace("```", "").strip()

            # Parse JSON safely
            plan = json.loads(plan_json)

            state["entities"] = plan.get("entities", [])
            state["required_attributes"] = plan.get("required_attributes", [])
            state["plan_steps"] = plan.get("reasoning_steps", [])
            state["document_hints"] = plan.get("document_hints", [])

            logger.info(f"[{request_id}] Planner entities: {state['entities']}")
            logger.info(f"[{request_id}] Planner attributes: {state['required_attributes']}")
            logger.info(f"[{request_id}] Planner document hints: {state['document_hints']}")
            logger.info(f"[{request_id}] Planner steps: {state['plan_steps']}")

            return state

        except Exception as e:
            logger.exception(f"[{request_id}] Planner failed")
            state["error"] = "Planner failed to generate structured plan" + str(e)
            return state
