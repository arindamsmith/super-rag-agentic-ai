import logging
import time
from uuid import uuid4
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

from ingestion_agents.ingestion_orchestrator import IngestionOrchestrator
from orchestrator_agent import OrchestratorAgent

from dotenv import load_dotenv
load_dotenv()


# ---------------- Logging ----------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("./logs/app.log"),   # Persist logs
        logging.StreamHandler()                 # Console logs
    ]
)

logger = logging.getLogger("SuperRAGApp")

# ---------------- FastAPI Models ----------------

class IngestRequest(BaseModel):
    data_dir: str = "data"

class ChatRequest(BaseModel):
    query: str

# ---------------- App Lifespan ----------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Super RAG application starting up...")
    yield
    logger.info("Super RAG application shutting down...")

app = FastAPI(
    title="Super RAG – Agentic AI System",
    version="1.0",
    lifespan=lifespan
)

# ---------------- Agents ----------------

ingestion_orchestrator = IngestionOrchestrator()
orchestrator = OrchestratorAgent()

# ---------------- Endpoints ----------------

@app.post("/ingest")
async def ingest_documents(request: IngestRequest):
    """
    Ingest documents from a folder:
    - Load
    - Chunk
    - Embed
    - Store in Qdrant
    """
    try:
        logger.info(f"Received ingestion request for directory: {request.data_dir}")
        start = time.time()

        await ingestion_orchestrator.ingest(request.data_dir)

        latency = round(time.time() - start, 2)
        return {
            "status": "success",
            "message": f"Documents ingested from '{request.data_dir}'",
            "latency_seconds": latency
        }

    except Exception as e:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/superchat")
async def super_chat(request: ChatRequest):
    """
    Main Super RAG endpoint.
    Generates request_id, runs full agentic pipeline.
    """
    request_id = str(uuid4())
    logger.info(f"[{request_id}] Incoming query: {request.query}")

    state = {
        "request_id": request_id,
        "query": request.query
    }

    try:
        start = time.time()
        result_state = await orchestrator.run(state)
        latency = round(time.time() - start, 2)

        response = {
            "request_id": request_id,
            "query": request.query,
            "answer": result_state.get("final_answer"),
            "mode": result_state.get("mode"),
            "sources": result_state.get("sources"),
            "citations": result_state.get("citations"),
            "semantic_hit": result_state.get("semantic_hit", False),
            "latency_seconds": latency,
            "error": result_state.get("error")
        }

        logger.info(f"[{request_id}] Response ready in {latency}s, mode={response['mode']}")
        return response

    except Exception as e:
        logger.exception(f"[{request_id}] Super RAG processing failed")
        raise HTTPException(status_code=500, detail=str(e))
