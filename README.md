# AI Product Creative Generation Pipeline (FastAPI, Celery, LangGraph, Streamlit)

An advanced, production-hardened **AI Product Creative Generation Workflow** designed to orchestrate deep-ingestion DTC marketing research, automated campaign prompt factories, multi-asset canvas generation, and 7-way parallel multimodal visual critiques.

---

## Technical Stack & System Architecture

- **Web Gateway & Ingestion Layer**: FastAPI providing non-blocking asynchronous REST endpoints (`/api/v1/generate`, `/api/v1/bulk`) wrapped with rapid `httpx` and `asyncio.gather` gateway pre-flight validations.
- **Distributed Async Task Orchestration**: Celery workers powered by a high-throughput Redis broker.
- **State Machine Graph Orchestration**: LangGraph running sequential and parallel conditional workflows over a robust shared `ProductState` context.
- **DTC Research & Scrapes**: Crawl4AI fetching raw markdown pages parsed via instructor-validated LLM payloads.
- **Parallel Multi-Asset Critic Agent**: Async VLM reviews executing 7 parallel Together AI VLM calls (`meta-llama/Llama-Vision-Free`) concurrently over 5 PNG images and 2 video frame previews.
- **UI Canvas**: High-end Streamlit control dashboard featuring horizontal state timeline visualizers and batch fleet monitoring.

---

## Known Architectural Trade-offs & Scaling Limitations

While this prototype is engineered with strict error isolation and non-blocking asynchronous gateway controls, there are a few structural trade-offs optimized for ease of deployment, demo scope, and rapid iteration:

### 1. SQLite Persistence Layer Serialization
- **The Design**: The database persistence layer utilizes standard **SQLite** (`sqlite:///shared_data/jobs.db`) configured with `check_same_thread=False` and write-ahead logging (WAL) enabled. SQLite is exceptionally fast, local, serverless, and perfectly suited for standalone prototype or demo environments since it requires zero external service configuration.
- **The Bottleneck**: SQLite enforces database-level locks for write operations (write-serialization). Under highly concurrent bulk ingestion workloads (e.g. multi-tenant bulk fleets uploading dozens of batch CSV files simultaneously), parallel worker processes attempting to write status transitions, critic scores, and ZIP paths will experience locking contention, resulting in database lock wait timeouts (`database is locked` errors).
- **The Production Pathway**: To scale this system to handle millions of jobs and multi-tenant concurrent bulk writes cleanly, the database configuration should be migrated to **PostgreSQL**. A PostgreSQL upgrade allows row-level locking, massive write concurrency, connection pooling (via PgBouncer), and distributed database clustering.

### 2. ComfyUI Parallel Execution Limits
- **The Design**: Generates high-end creative images and Wan 2.1 video assets concurrently via ComfyUI.
- **The Bottleneck**: ComfyUI workflows require dedicated GPU VRAM allocations. Running more parallel generations than available GPU execution slots causes either queue serialization inside the ComfyUI server or memory out-of-memory (OOM) failures.
- **The Production Pathway**: Deploy ComfyUI inside serverless autoscaling clusters (e.g. RunPod, Modal, or Replicate) or front the asset generation workers with an active queue manager that distributes prompt requests dynamically to a fleet of replica ComfyUI instances.

---

## Getting Started

### 1. Environment Configuration
Copy the sample environment configuration:
```bash
cp .env.example .env
```
Fill in the `TOGETHER_API_KEY` to run premium visual critiques and serverless fallbacks.

### 2. Containerized Deployment
Launch the complete containerized stack (FastAPI, Redis, Celery, SQLite, Streamlit):
```bash
docker-compose up --build
```
Once initialized:
- **FastAPI Backend Gateway**: `http://localhost:8000`
- **Streamlit DTC Dashboard**: `http://localhost:8501`

---

## Core Ingestion & Generation Flow

```
   [ DTC URL / CSV Ingest ]
              ↓
  [ Gateway Pre-flight Check ] (Rapid live checks & content length > 500 chars)
              ↓
      [ Celery Enqueue ]
              ↓
     [ LangGraph Nodes ]
              ↓
     - Node 1: Research (Crawl4AI Page Scraper)
     - Node 2: Strategy (DTC Persona Angle Generator)
     - Node 3: Prompts (Visual Creative Prompt Factory)
     - Node 4 & 5: Generation (FLUX + LTX-Video ComfyUI or Serverless Fallbacks)
     - Node 6: Critic (7-Way Parallel VLM Quality Checks)
              ↓
  [ Conditional Routing Edge ] ──(Fail / Attempt < 2)──> [ Refine Prompts Node ]
              ↓
       (Pass or Exhausted)
              ↓
     - Node 7: Packager (Zips Assets & Serializes `"qa_status"`)
```

---

## Verification & Compilation
To compile and test backend files:
```bash
python3 -m py_compile backend/app.py backend/graph/nodes/video_gen.py backend/graph/nodes/critic.py backend/graph/nodes/packager.py
```
To run the automated integration tests:
```bash
python3 backend/tests/test_graph.py
```
