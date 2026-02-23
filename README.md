# Super RAG — Agentic Multi-Agent AI System

An enterprise-grade, multi-agent RAG system built with FastAPI, Google Gemini, and Qdrant.

## Architecture
- **Tier 1 — Semantic Memory:** Answers similar past questions instantly from vector cache
- **Tier 2 — Simple RAG:** Fast vector search + LLM for straightforward queries  
- **Tier 3 — Agentic Super RAG:** Full 6-agent pipeline for complex multi-document reasoning

## Tech Stack
- **LLM:** Google Gemini 2.5 Flash + Gemini 2.5 Pro
- **Vector DB:** Qdrant (local)
- **Embeddings:** Google text-embedding-004 (768-dim)
- **Framework:** FastAPI + Python async
- **Context:** Gemini Context Caching for long-document reasoning

## Agents
| Agent | Role |
|---|---|
| RouterAgent | Classifies query as SIMPLE_LOOKUP or COMPLEX_REASONING |
| QueryPlannerAgent | Decomposes query into entities, attributes, reasoning steps |
| DocumentHunterAgent | Vector search → full document retrieval |
| LongContextLoaderAgent | Caches full docs in Gemini Context Cache |
| AnalystAgent | Deep cross-document reasoning → structured JSON |
| CitationAgent | Grounds every fact in source document + section |
| ResponseFormatterAgent | Produces final cited answer |
| SemanticMemoryAgent | Semantic Q&A cache using Qdrant |

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env   # Add your API key
uvicorn app:app --reload
```

## API
- `POST /ingest` — Load and embed documents from the data/ folder
- `POST /superchat` — Ask a question, get a cited, grounded answer
