from typing import Dict, Any, List, Optional
from pydantic import BaseModel

from typing import List, Optional, Dict, Any
from pydantic import BaseModel


class AgentState(BaseModel):
    # Observability
    request_id: Optional[str] = None

    # User input
    query: str

    # Router output
    intent: Optional[str] = None   # SIMPLE_LOOKUP or COMPLEX_REASONING

    # Planner output
    entities: List[str] = []       # e.g. ["Financial Analyst"]
    required_attributes: List[str] = []  # e.g. ["Tier", "WFH Policy"]
    plan_steps: List[str] = []     # reasoning steps

    # Retrieval
    relevant_documents: List[Dict[str, Any]] = []  # {doc_name, metadata, full_text}

    # Long context
    cache_id: Optional[str] = None
    big_context_fallback: Optional[str] = None

    # Analyst output (structured reasoning)
    analysis_json: Optional[Dict[str, Any]] = None

    # Citations
    citations: Dict[str, Any] = {}

    # Final answer
    final_answer: Optional[str] = None

    # RAG simple path fields
    rag_answer: Optional[str] = None
    sources: List[str] = []
    mode: Optional[str] = None
    latency_seconds: Optional[float] = None

    # Semantic search details
    semantic_hit: bool = False
    semantic_score: Optional[float] = None

    # Error handling
    error: Optional[str] = None

