
import logging

from routing_agents.router_agent import RouterAgent
from rag_agents.simple_rag_agent import SimpleRAGAgent
from planning_agents.query_planner_agent import QueryPlannerAgent
from retrieval_agents.document_hunter_agent import DocumentHunterAgent
from reasoning_agents.long_context_loader_agent import LongContextLoaderAgent
from reasoning_agents.analyst_agent import AnalystAgent
from grounding_agents.citation_agent import CitationAgent
from reasoning_agents.response_formatter_agent import ResponseFormatterAgent
from memory_agents.semantic_memory_agent import SemanticMemoryAgent

logger = logging.getLogger("OrchestratorAgent")


class OrchestratorAgent:
    """
    Central brain of Super RAG.
    Routes queries through:
    - Tier 1: Semantic Memory
    - Tier 2: Simple RAG
    - Tier 3: Agentic Super RAG
    """

    def __init__(self):
        self.semantic_memory = SemanticMemoryAgent()
        self.router = RouterAgent()
        self.simple_rag = SimpleRAGAgent()

        # Super RAG pipeline
        self.planner = QueryPlannerAgent()
        self.hunter = DocumentHunterAgent()
        self.context_loader = LongContextLoaderAgent()
        self.analyst = AnalystAgent()
        self.citation = CitationAgent()
        self.formatter = ResponseFormatterAgent()

    async def run(self, state: dict) -> dict:
        
        request_id = state.get("request_id", "NA")

        logger.info(f"[{request_id}] Orchestration started for query: {state['query']}")

        try:
            # ---------- Tier 1: Semantic Memory ----------
            state = await self.semantic_memory.lookup(state)
            if state.get("semantic_hit"):
                logger.info(f"[{request_id}] Served from Semantic Memory (Tier-1)")
                return state

            # ---------- Routing ----------
            state = await self.router.run(state)
            intent = state.get("intent")

            # ---------- Tier 2: Simple RAG ----------
            if intent == "SIMPLE_LOOKUP":
                logger.info(f"[{request_id}] Routing to SimpleRAGAgent (Tier-2)")
                state = await self.simple_rag.run(state)

            # ---------- Tier 3: Super RAG ----------
            else:
                logger.info(f"[{request_id}] Routing to Super RAG Agentic Pipeline (Tier-3)")

                state = await self.planner.run(state)
                state = await self.hunter.run(state)
                state = await self.context_loader.run(state)
                state = await self.analyst.run(state)
                state = await self.citation.run(state)
                state = await self.formatter.run(state)

            # ---------- Store in Semantic Memory ----------
            state = await self.semantic_memory.store(state)

            logger.info(f"[{request_id}] Orchestration completed successfully")
            return state

        except Exception as e:
            logger.exception(f"[{request_id}] Orchestration failed")
            state["error"] = str(e)
            return state
