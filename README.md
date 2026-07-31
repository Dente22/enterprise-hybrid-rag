# Enterprise Hybrid RAG

[![CI](https://github.com/Dente22/enterprise-hybrid-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/Dente22/enterprise-hybrid-rag/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-async-009688.svg)](https://fastapi.tiangolo.com/)
[![pgvector](https://img.shields.io/badge/pgvector-hybrid%20search-336791.svg)](https://github.com/pgvector/pgvector)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Hybrid-Search Enterprise Document Q&A** — a production-shaped FastAPI service for enterprise retrieval-augmented generation.

It ingests PDF / Markdown / raw text, indexes chunks with **dense embeddings + full-text search**, retrieves candidates via **Reciprocal Rank Fusion (RRF)**, optionally **cross-encoder reranks** the top passages, and returns a **Pydantic-validated grounded answer** with citations and confidence.

Companion portfolio project to [`enterprise-ai-task-extractor`](https://github.com/Dente22/enterprise-ai-task-extractor).

---

## Why this exists

Vector-only RAG misses exact keywords. Keyword-only search misses paraphrases. This service combines both:

1. **Dense retrieval** — semantic similarity (pgvector / cosine)
2. **Lexical retrieval** — PostgreSQL FTS (`tsvector` / `ts_rank_cd`) or SQLite TF fallback
3. **RRF fusion** — stable merge of ranked lists
4. **Reranking** — CrossEncoder selects the top 3 chunks to reduce tokens and hallucination risk
5. **Structured Outputs** — `{ answer, confidence_score, sources }` enforced with Pydantic retries
6. **Safety gates** — prompt-injection sanitization + low-confidence fallback

---

## Feature matrix

| Capability | Details |
|---|---|
| Ingest | Raw text + file upload (PDF / Markdown / TXT) |
| Chunking | Paragraph/sentence-aware overlapping chunks + metadata |
| Embeddings | Ollama (`nomic-embed-text`) → OpenAI fallback |
| Hybrid search | Vector + FTS fused with weighted RRF |
| Rerank | `cross-encoder/ms-marco-MiniLM-L-6-v2` (optional; graceful fallback) |
| Answering | Llama3 / OpenAI with JSON schema validation |
| Security | Bearer API keys, injection filters, relevance threshold |
| Local DX | SQLite mode without Docker |
| Ops | Docker Compose, Alembic, pytest, GitHub Actions |

---

## Architecture

```mermaid
flowchart TB
  Client["Client / Swagger UI"] -->|Bearer API Key| API["FastAPI async API"]

  API --> Ingest["POST /documents/ingest\nPOST /documents/ingest-file"]
  API --> Query["POST /query"]
  API --> Health["GET /health"]

  Ingest --> Parse["PDF / Markdown / text parse"]
  Parse --> Chunk["Semantic chunking + metadata"]
  Chunk --> Embed["EmbeddingService\nOllama / OpenAI"]
  Embed --> DB[(PostgreSQL + pgvector + FTS\nor SQLite local mode)]

  Query --> Sanitize["Prompt-injection sanitizer"]
  Sanitize --> Hybrid["HybridSearchService"]
  Hybrid --> Vec["Dense top-N"]
  Hybrid --> FTS["Lexical top-N"]
  Vec --> RRF["Reciprocal Rank Fusion"]
  FTS --> RRF
  RRF --> Rerank["Cross-Encoder → top 3"]
  Rerank --> Gate{"Confidence ≥ threshold?"}
  Gate -->|no| Fallback["Deterministic low-confidence response"]
  Gate -->|yes| LLM["LLM Structured Answer"]
  LLM --> Resp["answer + confidence_score + sources"]
  DB --> Vec
  DB --> FTS
```

### Query lifecycle

```text
question
  → sanitize
  → embed(query)
  → dense top-N  +  FTS top-N
  → Reciprocal Rank Fusion
  → cross-encoder rerank → top_k (default 3)
  → relevance gate (MIN_CONFIDENCE_THRESHOLD)
  → LLM grounded JSON (validated + retried)
  → response
```

### Project layout

```text
enterprise-hybrid-rag/
├── app/
│   ├── api/                 # HTTP routers (health, documents, query)
│   ├── core/                # settings, auth, logging
│   ├── schemas/             # Pydantic contracts / Structured Outputs
│   ├── models/              # SQLAlchemy Document / DocumentChunk
│   ├── db/                  # async engine, pgvector bootstrap
│   ├── services/
│   │   ├── chunking.py
│   │   ├── embedding_service.py
│   │   ├── hybrid_search_service.py
│   │   ├── reranker.py
│   │   ├── ingest_service.py
│   │   ├── query_service.py
│   │   ├── llm_service.py
│   │   ├── parsers.py
│   │   └── sanitizer.py
│   └── main.py
├── alembic/
├── tests/
├── docker-compose.yml
├── Dockerfile
└── .env.example
```

---

## Quick start (local — recommended)

**Requirements:** Python 3.11+, [Ollama](https://ollama.com)

```bash
git clone https://github.com/Dente22/enterprise-hybrid-rag.git
cd enterprise-hybrid-rag

python -m venv .venv
# Windows (if Activate.ps1 is blocked by execution policy):
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# or without CrossEncoder / torch (avoids long OneDrive paths on Windows):
.\.venv\Scripts\python.exe -m pip install fastapi "uvicorn[standard]" python-multipart pydantic pydantic-settings "sqlalchemy[asyncio]" asyncpg aiosqlite pgvector alembic httpx python-dotenv pypdf numpy

cp .env.example .env
mkdir data uploads

ollama pull llama3
ollama pull nomic-embed-text

# Windows tip — call uvicorn via venv python:
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8001
```

| Resource | Value |
|---|---|
| Swagger UI | http://127.0.0.1:8001/docs |
| Health | http://127.0.0.1:8001/api/v1/health |
| Default API key | `dev-secret-key-change-me` |

Local defaults:

- **SQLite** persistence
- Hybrid mode: cosine + lightweight lexical TF + RRF
- Set `RERANKER_ENABLED=false` if you skip `sentence-transformers` / torch

> Port **8001** avoids clashing with [`enterprise-ai-task-extractor`](https://github.com/Dente22/enterprise-ai-task-extractor) on `8000`.

---

## Docker Compose (Postgres + pgvector + Ollama)

```bash
cp .env.example .env
# optional: OPENAI_API_KEY=... for LLM/embedding fallback

docker compose up --build -d
docker compose logs -f ollama-init
```

| Service | Role |
|---|---|
| `db` | PostgreSQL 16 + pgvector (host port **5433**) |
| `ollama` | Local LLM + embeddings |
| `ollama-init` | Pulls `llama3` + `nomic-embed-text` |
| `api` | FastAPI on host port **8001** |

```bash
curl -s http://localhost:8001/api/v1/health
docker compose down
# wipe volumes:
docker compose down -v
```

Compose overrides `DATABASE_URL` / `OLLAMA_BASE_URL` for the container network. Full hybrid path uses **pgvector HNSW** + **GIN FTS**.

---

## API reference

All write/query endpoints require:

```http
Authorization: Bearer <API_KEY>
```

### `GET /api/v1/health`

Unauthenticated liveness / DB probe.

### `POST /api/v1/documents/ingest`

Index raw text.

```bash
curl -s -X POST http://127.0.0.1:8001/api/v1/documents/ingest \
  -H "Authorization: Bearer dev-secret-key-change-me" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Q3 planning memo. Alice owns the budget draft due 2026-08-15. Risk: supplier delays in APAC.",
    "source": "q3-planning.md",
    "metadata": { "team": "finance" }
  }'
```

### `POST /api/v1/documents/ingest-file`

Upload PDF / Markdown / TXT.

```bash
curl -s -X POST http://127.0.0.1:8001/api/v1/documents/ingest-file \
  -H "Authorization: Bearer dev-secret-key-change-me" \
  -F "file=@./sample.md" \
  -F "source=sample.md"
```

### `POST /api/v1/query`

Hybrid retrieve → rerank → grounded structured answer.

```bash
curl -s -X POST http://127.0.0.1:8001/api/v1/query \
  -H "Authorization: Bearer dev-secret-key-change-me" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Who owns the Q3 budget draft and when is it due?",
    "top_k": 3
  }'
```

**Example response**

```json
{
  "answer": "Alice owns the Q3 budget draft, due 2026-08-15.",
  "confidence_score": 0.86,
  "sources": [
    {
      "document_id": "…",
      "chunk_id": "…",
      "source": "q3-planning.md",
      "excerpt": "Alice owns the budget draft due 2026-08-15.",
      "score": 0.91,
      "metadata": { "chunk_id": 0 }
    }
  ],
  "model": "llama3",
  "provider": "ollama",
  "retrieval_mode": "sqlite-cosine+fts+rrf",
  "reranked": false,
  "low_confidence": false,
  "candidates_considered": 8
}
```

---

## Hybrid search details

### Reciprocal Rank Fusion

For each retrieval channel \(i\) with weight \(w_i\):

\[
\mathrm{score}(d) = \sum_i w_i \cdot \frac{1}{k + \mathrm{rank}_i(d)}
\]

Defaults: `HYBRID_VECTOR_WEIGHT=0.55`, `HYBRID_FTS_WEIGHT=0.45`, `k=60`.

### Reranking

- Enabled via `RERANKER_ENABLED=true`
- Model default: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- If the model cannot load (missing torch, path limits, etc.), the service continues with **RRF-only** ranking

### Low-confidence fallback

If retrieval/model confidence &lt; `MIN_CONFIDENCE_THRESHOLD`, the API returns a deterministic safe answer instead of forcing a speculative completion.

---

## Security model (MVP)

| Control | Behavior |
|---|---|
| Auth | `Authorization: Bearer <API_KEY>` |
| Keys | `API_KEYS` (comma-separated) |
| Injection filter | Rejects common jailbreak / instruction-override patterns |
| Upload / text limits | `MAX_UPLOAD_BYTES`, `MAX_TEXT_LENGTH` |
| Grounding | Answer only from retrieved chunks; schema-enforced citations |

Suitable for demos and portfolio review — not a hardened multi-tenant production deployment.

---

## Configuration

See [`.env.example`](.env.example).

| Variable | Purpose |
|---|---|
| `API_KEYS` | Accepted Bearer tokens |
| `DATABASE_URL` | Postgres async URL or SQLite |
| `LLM_PROVIDER` | `auto` · `ollama` · `openai` |
| `OLLAMA_MODEL` / `OLLAMA_EMBED_MODEL` | Chat + embedding models |
| `OPENAI_API_KEY` | Enables OpenAI fallback |
| `HYBRID_VECTOR_WEIGHT` / `HYBRID_FTS_WEIGHT` | RRF channel weights |
| `HYBRID_CANDIDATE_LIMIT` | Candidates before rerank |
| `RERANK_TOP_K` | Chunks passed to the LLM (default 3) |
| `MIN_CONFIDENCE_THRESHOLD` | Relevance / answer safety gate |
| `RERANKER_ENABLED` | Toggle CrossEncoder |

---

## Testing & quality gates

```bash
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check app tests scripts
```

CI runs ruff + pytest on push/PR: [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

---

## Related portfolio project

- [`enterprise-ai-task-extractor`](https://github.com/Dente22/enterprise-ai-task-extractor) — Structured Outputs task extraction + lightweight RAG over action items

Together they cover **extraction → indexing → hybrid retrieval → grounded generation**.

---

## License

MIT — see [`LICENSE`](LICENSE). Use and adapt freely for portfolio demos and interviews.
