# Technical Synopsis: AI Product Creative Generation Pipeline

This document provides a highly technical, end-to-end architectural manifest of the **AI Product Creative Generation Pipeline**. It details the structural topology, asynchronous scheduling design, state validation models, concurrent execution resolutions, and user-facing control systems of the platform.

---

## 1. System Architectural Overview & Data Flow

The architecture operates on an asynchronous, decoupled, event-driven pattern designed to isolate expensive compute processes (diffusion image and motion video models) from the primary web application threads. 

### 1.1 Ingestion & Execution Lifecycle
The following ASCII data flow tracks the lifecycle of an ingestion payload from the frontend Streamlit controller to final archival packaging:

```
[Streamlit Controller Dashboard]
             │
             │ (HTTP POST /api/v1/generate or /api/v1/bulk)
             ▼
     [FastAPI Backend] ─── (Atomic Transaction) ───► [SQLite DB (jobs.db)]
             │
             │ (Celery Task Enqueue)
             ▼
     [Redis Message Broker]
             │
             │ (3/m Token-Bucket Dequeue)
             ▼
   [Celery Worker Thread]
             │
             ├─► live out-of-band updates ──► [db_update_job_status] ──► [SQLite]
             │
             ▼
   [LangGraph Orchestrator (State Machine Graph)]
             │
             ├──► 🔍 [Node 1: Product Research (Crawl4AI + Llama 3.1 8B)]
             │
             ├──► 💡 [Node 2: Creative Strategy (Llama 3.3 70B Brief)]
             │
             ├──► 📝 [Node 3: Prompts Generation (FLUX/Wan Art Direction)]
             │
             ├──► 🔀 [Parallel Generation Fork (Diamond Shape)]
             │         │
             │         ├─► 🎨 [Node 4: Image Gen] ──► (ComfyUI / Together FLUX)
             │         │
             │         └─► 🎥 [Node 5: Video Gen] ──► (ComfyUI / public CDN mock)
             │
             ├──► 🤝 [Fan-In Convergence (reduce_assets custom list reducer)]
             │
             ├──► 👁️ [Node 6: Multimodal QA Review / Critic Agent]
             │         │
             │         └─► (Together AI Vision Llama-Vision-Free)
             │                 │
             │                 ├─► [Score < 7.0 & Attempts < 2] ─► (Loop back to prompts)
             │                 └─► [Score >= 7.0 or Attempts = 2] ─► (Proceed to packaging)
             │
             └──► 📦 [Node 7: Output Packager] ──► (Zip consolidation + metadata.json)
                                                                 │
                                                                 ▼
                                                        [output.zip on disk]
```

### 1.2 The Decoupled Execution Paradigm
Running heavy text-to-image (FLUX.1-dev) and image-to-video (Wan 2.1) latent diffusion models introduces immense compute constraints, including severe VRAM fragmentation, GPU-bound thread-locking, and high process crashing rates. 

To achieve production-grade stability, our pipeline enforces strict process isolation:
- **Headless ComfyUI API Instances**: All generative diffusion modeling is shifted out-of-band into independent, dedicated ComfyUI process spaces running headless on dedicated GPU nodes.
- **WebSocket/HTTP Client Interface**: Node 4 and Node 5 act purely as high-speed API clients that load workflow templates, dynamically override positive/negative prompt slots, upload image seeds, and poll ComfyUI using robust websocket loops and HTTP polling failsafes.
- **Resource Protection**: The web workers (FastAPI) and task queues (Celery) are entirely insulated from native execution of C-level CUDA code, guaranteeing that runtime driver faults or VRAM depletion inside ComfyUI never crash or corrupt the core orchestrator or persistent state database.

---

## 2. State Management & Graph Topology Specs

### 2.1 Shared State Schema (`ProductState`)
Our LangGraph orchestrator passes a strict `ProductState` schema defined as a Python `TypedDict`. This state accumulates execution data, assets, and diagnostic scores across the graph transitions.

```python
from typing import TypedDict, List, Dict, Any, Annotated, Optional

class ProductState(TypedDict):
    """The shared state schema flowing through the LangGraph orchestrator."""
    # Ingestion Parameters
    url: str                           # Target product landing page URL
    job_id: str                        # Globally unique tracking identifier (UUIDv4)
    retry_count: int                   # Current critique loop iteration attempt (0, 1, or 2)

    # Scraped Data & Briefs
    product_data: Dict[str, Any]       # Title, features, technical specs, pricing, and reviews
    creative_brief: Dict[str, Any]     # Hooks, target demographics, aesthetic palette, and ad copy
    generation_prompts: Dict[str, Any] # 5 detailed visual image prompts + 2 motion video prompts

    # Thread-Safe Custom Reduced Assets
    generated_assets: Annotated[List[str], reduce_assets] # Paths to generated files

    # QA Audit Payloads
    critic_scores: List[float]         # Historic record of visual VLM average QA scores
    critic_feedback: str               # Constructive prompt corrections injected back into Node 3
```

### 2.2 Orchestrator Graph Topology
The graph is configured using LangGraph's compiler. Nodes represent execution routines, and edges direct the execution control path:

1. **Start & Sequential Entry**:
   - `__start__` ──► `research` (Node 1)
   - `research` ──► `strategy` (Node 2)
   - `strategy` ──► `prompts` (Node 3)
2. **Parallel Fan-Out Branch (Diamond Fork)**:
   - `prompts` fanns out concurrently to `image_gen` (Node 4) and `video_gen` (Node 5). These nodes execute their async generation routines concurrently using `asyncio.gather` tasks.
3. **Convergent Fan-In & Review**:
   - Both `image_gen` and `video_gen` converge at `critic` (Node 6).
4. **Conditional Quality Edge Gate**:
   - `critic` triggers a conditional router helper: `should_retry`.
   - If `latest_score < 7.0` and `retry_count < 2` and `critic_feedback` is present:
     - Edge directs: `critic` ──► `prompts` (initiating feedback-injected prompt refinement loop).
   - If quality meets threshold (score $\ge 7.0$) or retry limits are exhausted:
     - Edge directs: `critic` ──► `packager` (Node 7).
5. **Termination**:
   - `packager` ──► `__end__` (Terminates graph execution).

---

## 3. Core Engineering Implementations & Edge-Case Resolutions

### 3.1 The Fan-In Concurrency Resolution
When fanning-in parallel branches in standard graphs, if both branches return writes to the exact same state key at the same execution tick, LangGraph is unable to determine priority. It throws a critical `InvalidUpdateError: At key 'generated_assets': Can receive only one value per step` exception and halts the process.

To resolve this without forcing sequential execution (which would double asset generation times), we engineered a thread-safe list reducer:

```python
def reduce_assets(left: Optional[List[str]], right: Optional[List[str]]) -> List[str]:
    """
    Reducer function to merge generated assets concurrently and remove duplicates.
    Enables parallel nodes image_gen and video_gen to append to the same list.
    """
    if left is None:
        left = []
    if right is None:
        right = []
    res = list(left)
    for item in right:
        if item not in res:
            res.append(item)
    return res
```
By wrapping the state annotation `generated_assets: Annotated[List[str], reduce_assets]`, we instruct LangGraph's Pregel execution engine to resolve concurrent writes by executing our reducer. This merges lists and eliminates duplicates cleanly.

### 3.2 Out-of-Band State Streaming Synchronization
Our worker threads run independent of database connections. Having nodes write directly to the database would introduce tight coupling and compromise unit testing. To resolve this, we leverage LangGraph’s dynamic update streams (`app_graph.astream`) inside the Celery worker task space:

```python
# In celery_worker.py:
async for event in app_graph.astream(initial_state, config=config, stream_mode="updates"):
    for node_name, updates in event.items():
        # Live out-of-band database update
        db_update_job_status(
            job_id=job_id,
            status="running",
            current_node=node_name
        )
```
This design isolates database writes. As the graph executes, it yields intermediate updates step-by-step. The Celery execution frame captures these transitions, updates the SQLite `JobRecord` with the active node name, and preserves atomic isolation.

### 3.3 Closed-Loop Visual Critique & Self-Correction Engine
The review step (Node 6) performs visual inspection before packaging. 

```
┌──────────────────┐      VLM Evaluation      ┌──────────────────┐
│   image_1.png    ├─────────────────────────►│ Together AI VLM  │
│(Primary PNG Asset│                          │(Llama 3.2 11B Vision)
└────────┬─────────┘                          └────────┬─────────┘
         │                                             │
         │ (OpenCV capture /                           │
         │  copy fallback if mock)                     │ (Strict JSON Output schema)
         ▼                                             ▼
┌──────────────────┐                          ┌──────────────────┐
│ video_1_frame.png│                          │ CriticEvaluation │
│(Video Anchor)    │                          │ (Scores & Feedback)
└──────────────────┘                          └────────┬─────────┘
                                                       │
                                              [Score < 7.0 / Loop]
                                                       │
                                                       ▼
                                              ┌──────────────────┐
                                              │ Node 3: Prompts  │
                                              │(Feedback injected│
                                              │ system prompt)   │
                                              └──────────────────┘
```

#### Code Mechanics of Node 6 (`critic.py`):
1. **Video Layout Frame Extraction**: Since VLMs cannot process raw MP4 containers, we built an OpenCV (`cv2`) image capture utility:
   ```python
   def extract_video_frame(video_path: str, anchor_image_path: Optional[str] = None) -> Optional[str]:
       # Try capturing the first frame using OpenCV
       try:
           import cv2
           cap = cv2.VideoCapture(video_path)
           if cap.isOpened():
               success, frame = cap.read()
               if success and frame is not None:
                   frame_path = video_path.replace(".mp4", "_frame.png")
                   cv2.imwrite(frame_path, frame)
                   return frame_path
       except Exception:
           pass
       # Resilient Fallback: Copy pre-bound anchor image
       if anchor_image_path and os.path.exists(anchor_image_path):
           shutil.copy2(anchor_image_path, frame_path)
           return frame_path
   ```
2. **Data URI Base64 Serialization**: The target visual asset is read, base64-encoded, and formatted in a compliant URI schema: `data:image/png;base64,{base64_string}`.
3. **Structured Serverless Together AI Vision Query**:
   To bypass Instructor-multimodal parser conflicts, the node interacts with Together's endpoint (`meta-llama/Llama-Vision-Free` or `meta-llama/Llama-3.2-11b-vision-instruct`) using the standard `openai` library directly, passing `response_format={"type": "json_object", "schema": CriticEvaluation.model_json_schema()}`.
4. **Self-Correction & Mutation Loop**:
   - If `average_score < 7.0` and `retry_count < 2`:
     - Node returns `critic_feedback` set to the VLM's constructive critiques and sets `retry_count = retry_count + 1`.
     - In the subsequent loop, Node 3 (`prompts_node`) checks `state['critic_feedback']`. If populated, it prepends the critic warning block to the model generation prompts, forcing the prompt generator to adjust camera angles, volumetric lighting, or color palette parameters to resolve the visual QA failure.

### 3.4 Fault-Isolated Bulk Queue Logistics
Ingesting hundreds of URLs from bulk operations poses batch-level stability risks. A single invalid URL must never halt or crash the entire execution queue.

Our architecture secures bulk queues through three distinct layers:
1. **Payload Isolation**: The FastAPI backend parses the CSV using pandas but never launches a combined batch job. Instead, it iterates through rows, creates a unique database entry (`JobRecord` with a shared `batch_id` and unique `job_id`), and enqueues **independent** Celery tasks.
2. **Worker Exception Containment**: The worker runs each job in isolation inside a separate thread. If a job fails due to scraping blocks, network anomalies, or API timeouts, the task catches the exception, updates that specific job status to `FAILED` in the database, and exits cleanly. The Celery execution frame remains stable, moving directly to process subsequent tasks.
3. **Token-Bucket Rate Limiting**: Workers enforce a strict rate limit of `3/m` (`rate_limit='3/m'`). This prevents API key exhaustion, respects upstream provider token quotas, and ensures smooth throughput over prolonged batch executions.

---

## 4. Codebase Directory Topology

The project is structured as a clear, modular monorepo, keeping agents, state orchestration, workflows, and frontend controllers cleanly decoupled:

```
new_proj/
├── .env                  # System environment parameters & API keys
├── .env.example          # Public blueprint documentation of environment parameters
├── Dockerfile            # Consolidated Playwright, cv2, and Python environment setup
├── docker-compose.yml    # Service composer (redis, backend, celery_worker, frontend)
├── requirements.txt      # Production library dependencies
├── crawl4ai.py           # Mock stub bypass for local browser isolation tests
├── shared_data/          # SQLite persistent databases mounting directory
│   └── jobs.db           # Persistent database for job records
├── outputs/              # Structured shared file outputs directory (PNGs, MP4s, ZIPs)
├── backend/              # Decoupled backend service directories
│   ├── app.py            # FastAPI service exposing endpoints and bulk CSV parsing
│   ├── celery_worker.py  # Celery initialization, queues, and async graph runners
│   ├── agents/           # Pydantic schemas and client wrapper utilities
│   │   ├── schemas.py    # Strict structured output schemas
│   │   └── llm.py        # Client manager with exponential backoff on 429 exceptions
│   ├── workflows/        # Extracted ComfyUI JSON templates
│   │   ├── flux_image.json  # FLUX KSampler & IP-Adapter workflow
│   │   └── wan_video.json   # Wan 2.1 Latent video generation workflow
│   └── graph/            # LangGraph state machine orchestrator
│       ├── graph.py      # Core graph compiler, edges, and conditional routers
│       ├── state.py      # ProductState and thread-safe custom asset reducers
│       └── nodes/        # Agent node implementations
│           ├── research.py  # Node 1: Crawl4AI / BS4 scraper
│           ├── strategy.py  # Node 2: DTC Marketing briefs
│           ├── prompts.py   # Node 3: Flux/Wan prompting and QA injector
│           ├── image_gen.py # Node 4: Parallel ComfyUI FLUX image generator
│           ├── video_gen.py # Node 5: Parallel ComfyUI video generator
│           ├── critic.py    # Node 6: VLM visual review and frame extractor
│           └── packager.py  # Node 7: Output zip archiver
└── frontend/             # Decoupled user interface dashboard
    └── app.py            # Streamlit dashboard, timeline visualizer, and audit inspector
```

---

## 5. User Interface & Controller Specification

The controller dashboard (`frontend/app.py`) is written as a fully featured, single-page Streamlit application designed for system-level operations.

### 5.1 Real-Time Orchestrator Timeline
As the Celery worker streams node transitions to the SQLite database, the Streamlit interface captures the `current_node` value and renders a horizontal status track:
- **Completed Nodes**: Rendered inside light green glassmorphic containers (`#10B981`) marking them as completed.
- **Active Node**: Styled inside an active purple card border (`#7C3AED`) featuring pulsing opacity animations (`animation: pulse 1.5s infinite`).
- **Awaiting Nodes**: Styled inside subtle, desaturated border outlines indicating they are queued for processing.

### 5.2 Single Job Gallery Grid
Upon successful job completion, the frontend reads the database output path, checks the local shared container volume, or falls back to downloading and extracting the ZIP archive to a temporary cache. It then dynamically populates two presentation elements:
- **Aesthetic 5-Column Image Grid**: Utilizes `st.columns(5)` to render the 5 campaign graphics.
- **Symmetric 2-Column Video Grid**: Utilizes `st.columns(2)` to embed the 2 ad reels using native HTML video tags (`st.video`), providing immediate visual confirmation.

### 5.3 Real-Time Fleet Monitor Table
For batch runs, the bulk processor maps the database query data into a styled Pandas DataFrame and presents it using `st.data_editor`. The editor:
- Renders columns for `Job ID`, `URL`, `Status`, `Active Node`, `Retry Attempts`, and `Elapsed Time`.
- Implements custom HTML links inside the grid using Streamlit's `LinkColumn` to let operators stream the zip package directly from completed rows.

### 5.4 Exploded Audit Diagnostics Inspector
The inspector tab provides deeper insight into the pipeline's execution. Pasting a Job UUID extracts the complete final state dictionary and presents it across 4 themed expander blocks:
1. **Product Research Extraction**: Expands scraped HTML tags and pricing matrices.
2. **DTC Strategy Brief**: Audits target demographics, visual palettes, and ad hooks.
3. **Refined Prompts**: Displays generated FLUX/Wan text prompts.
4. **VLM Critic QC Logs**: Renders a complete interactive **Quality Score Progression Chart** mapping VLM scores history across loop iterations, along with details of the exact prompt adjustments made in response to visual critiques. This serves as an auditable verification trail of the LangGraph self-correction loop.
