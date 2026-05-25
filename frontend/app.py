import os
import sys
import time
import zipfile
import io
import requests
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Any
import streamlit as st

# 1. Page Configuration & Aesthetic Theme Injection
st.set_page_config(
    page_title="AudioAura - AI Creative Pipeline Controller",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Premium HSL-tailored colors, Dark Mode, typography, and glassmorphism styling
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap" rel="stylesheet">

<style>
    /* Global Styles & Typography */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    /* Gradient Background Hero Section */
    .hero-container {
        background: linear-gradient(135deg, #1E1B4B 0%, #0F172A 100%);
        border: 1px solid rgba(124, 58, 237, 0.2);
        border-radius: 16px;
        padding: 30px;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px rgba(124, 58, 237, 0.15);
    }
    
    .hero-title {
        background: linear-gradient(90deg, #A78BFA 0%, #06B6D4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        margin: 0;
        font-weight: 800;
    }
    
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.1rem;
        margin-top: 8px;
        margin-bottom: 0;
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        backdrop-filter: blur(12px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }

    /* Status Badges */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    
    .status-badge.completed {
        background: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .status-badge.running {
        background: rgba(124, 58, 237, 0.15);
        color: #A78BFA;
        border: 1px solid rgba(124, 58, 237, 0.3);
    }
    
    .status-badge.pending {
        background: rgba(245, 158, 11, 0.15);
        color: #F59E0B;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    
    .status-badge.failed {
        background: rgba(239, 68, 68, 0.15);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    /* Pulsing animations for active execution node */
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
</style>
""", unsafe_allow_html=True)

# 2. Environment Variables & Constants
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
CACHE_DIR = ".cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# 3. Connection Diagnostics & Backend Availability Checks
def check_backend_health() -> bool:
    """Safely checks if the FastAPI backend service is online and accepting connections."""
    try:
        # Check docs or root path for simple handshake verification
        resp = requests.get(f"{BACKEND_URL}/docs", timeout=3)
        return resp.status_code == 200
    except requests.exceptions.RequestException:
        return False

# 4. Outbound API Network Helper Functions
def trigger_single_job(url: str) -> Optional[str]:
    """Issues an ingestion request to trigger a single campaign generation workflow."""
    try:
        endpoint = f"{BACKEND_URL}/api/v1/generate"
        payload = {"url": url}
        resp = requests.post(endpoint, json=payload, timeout=10)
        if resp.status_code == 200:
            return resp.json().get("job_id")
        else:
            st.error(f"Backend API returned error code {resp.status_code}: {resp.text}")
    except Exception as e:
        st.error(f"Failed to communicate with pipeline server: {e}")
    return None

def fetch_job_details(job_id: str) -> Optional[dict]:
    """Retrieves full execution details and metadata for a specific job."""
    try:
        endpoint = f"{BACKEND_URL}/api/v1/job/{job_id}"
        resp = requests.get(endpoint, timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        st.warning(f"Could not connect to query job status: {e}")
    return None

def trigger_bulk_jobs(file_bytes: bytes, file_name: str) -> Optional[dict]:
    """Submits a parsed CSV file as multipart form-data to trigger a bulk creative generation run."""
    try:
        endpoint = f"{BACKEND_URL}/api/v1/bulk"
        files = {"file": (file_name, file_bytes, "text/csv")}
        resp = requests.post(endpoint, files=files, timeout=15)
        if resp.status_code == 200:
            return resp.json() # Returns dict containing 'batch_id' and list of 'job_ids'
        else:
            st.error(f"Bulk ingestion failed: {resp.text}")
    except Exception as e:
        st.error(f"Failed to submit bulk batch: {e}")
    return None

def fetch_batch_details(batch_id: str) -> Optional[list]:
    """Pulls status updates for all tracking jobs grouped inside a bulk CSV batch."""
    try:
        endpoint = f"{BACKEND_URL}/api/v1/batch/{batch_id}"
        resp = requests.get(endpoint, timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        st.warning(f"Failed to refresh batch statistics: {e}")
    return None

# 5. Resilient Local & Download ZIP Fallback Asset Handlers
def extract_and_load_visuals(job_id: str):
    """
    Unpacks visual assets (images and videos) for presentation.
    First checks if the assets already exist in the shared output directory path.
    Otherwise, downloads the output ZIP package from FastAPI and extracts it locally to a cache.
    """
    # Look for files inside the shared container volume first
    shared_path = os.path.join("outputs", job_id)
    images_dir = os.path.join(shared_path, "images")
    videos_dir = os.path.join(shared_path, "videos")
    
    if os.path.exists(images_dir) and os.listdir(images_dir):
        images = [os.path.join(images_dir, f) for f in sorted(os.listdir(images_dir)) if f.endswith(".png")]
        videos = [os.path.join(videos_dir, f) for f in sorted(os.listdir(videos_dir)) if f.endswith(".mp4")]
        if len(images) >= 5:
            return images, videos

    # Local scratch directory caching
    cache_path = os.path.join(CACHE_DIR, job_id)
    cache_images = os.path.join(cache_path, "images")
    cache_videos = os.path.join(cache_path, "videos")
    
    if os.path.exists(cache_images) and os.listdir(cache_images):
        images = [os.path.join(cache_images, f) for f in sorted(os.listdir(cache_images)) if f.endswith(".png")]
        videos = [os.path.join(cache_videos, f) for f in sorted(os.listdir(cache_videos)) if f.endswith(".mp4")]
        return images, videos

    # Endpoint downloading fallback
    try:
        os.makedirs(cache_path, exist_ok=True)
        download_url = f"{BACKEND_URL}/api/v1/job/{job_id}/download"
        resp = requests.get(download_url, timeout=30)
        if resp.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zip_ref:
                zip_ref.extractall(cache_path)
            
            # Fetch cache targets
            images = [os.path.join(cache_images, f) for f in sorted(os.listdir(cache_images)) if f.endswith(".png")]
            videos = [os.path.join(cache_videos, f) for f in sorted(os.listdir(cache_videos)) if f.endswith(".mp4")]
            return images, videos
    except Exception as e:
        st.warning(f"Visual retrieval engine was unable to pull target files: {e}")
    
    return [], []

# 6. Streamlit Real-Time Progress Flow Component
def render_orchestration_timeline(active_node: str):
    """
    Renders a stunning horizontal visual progress timeline displaying 
    the active and completed stages of the LangGraph state machine orchestrator.
    """
    stages = ["research", "strategy", "prompts", "image_gen", "video_gen", "critic", "packager"]
    friendly_titles = {
        "research": "🔍 Research",
        "strategy": "💡 Strategy",
        "prompts": "📝 Prompts",
        "image_gen": "🎨 Images",
        "video_gen": "🎥 Videos",
        "critic": "👁️ QA Critic",
        "packager": "📦 Package"
    }

    active_idx = -1
    if active_node in stages:
        active_idx = stages.index(active_node)
    elif active_node == "initializing":
        active_idx = 0
    elif active_node == "packaged":
        active_idx = len(stages) - 1

    cols = st.columns(len(stages))
    for idx, stage in enumerate(stages):
        title = friendly_titles[stage]
        with cols[idx]:
            if idx < active_idx:
                # Node has run and successfully processed
                st.markdown(f"""
                <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid #10B981; border-radius: 8px; padding: 10px 4px; text-align: center; color: #10B981; font-weight: 600; font-size: 0.8rem;">
                    {title}<br><span style="font-size: 0.65rem; font-weight: normal; opacity: 0.85;">Completed</span>
                </div>
                """, unsafe_allow_html=True)
            elif idx == active_idx:
                # Active processing Node
                st.markdown(f"""
                <div style="background: rgba(124, 58, 237, 0.18); border: 2px solid #7C3AED; border-radius: 8px; padding: 9px 4px; text-align: center; color: #C084FC; font-weight: 700; font-size: 0.8rem; box-shadow: 0 0 10px rgba(124, 58, 237, 0.45);">
                    {title}<br><span style="font-size: 0.65rem; font-weight: normal; animation: pulse 1.5s infinite;">Active...</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Pending queue
                st.markdown(f"""
                <div style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 8px; padding: 10px 4px; text-align: center; color: rgba(255, 255, 255, 0.35); font-size: 0.8rem;">
                    {title}<br><span style="font-size: 0.65rem; font-weight: normal; opacity: 0.6;">Awaiting</span>
                </div>
                """, unsafe_allow_html=True)

# 7. Render Hero Header Card
st.markdown("""
<div class="hero-container">
    <h1 class="hero-title">AI Product Creative Generation Controller</h1>
    <p class="hero-subtitle">Production-grade DTC Creative Generation Pipeline Orchestrating Multimodal LangGraph Agents.</p>
</div>
""", unsafe_allow_html=True)

# 8. Check Backend Connectivity
if not check_backend_health():
    st.warning("⚠️ Connection Diagnostics Alert: The primary FastAPI pipeline server is currently offline or unreachable.")
    st.info("Awaiting pipeline connection... Please ensure Docker Compose services or your local uvicorn host is running at " + BACKEND_URL)
    st.stop()

def render_completed_gallery(job_id: str):
    # Fetch output ZIP path
    download_endpoint = f"{BACKEND_URL}/api/v1/job/{job_id}/download"
    st.markdown(f"[📥 Download Creative Asset ZIP Package]({download_endpoint})", unsafe_allow_html=True)
    
    # Retrieve files for gallery rendering
    with st.spinner("Extracting media files for live rendering..."):
        images, videos = extract_and_load_visuals(job_id)
    
    # Grid 1: 5 Generated Marketing Images
    st.markdown("### 🎨 AI Generated Creative Image Campaign (FLUX.1-dev)")
    if images:
        cols_img = st.columns(5)
        for idx, img_path in enumerate(images[:5]):
            with cols_img[idx]:
                if os.path.exists(img_path):
                    st.image(img_path, caption=f"Creative Ad Visual {idx + 1}", use_container_width=True)
                else:
                    st.warning("Asset not found.")
    else:
        st.info("No generated PNG images found in the packaged payload.")

    # Grid 2: 2 Marketing Videos
    st.markdown("### 🎥 AI Generated Motion Video Ad Creatives (Wan 2.1 / LTX)")
    if videos:
        cols_vid = st.columns(2)
        for idx, vid_path in enumerate(videos[:2]):
            with cols_vid[idx]:
                if os.path.exists(vid_path):
                    st.video(vid_path)
                    st.markdown(f"<p style='text-align: center; font-size: 0.85rem; color: #94A3B8;'>Motion Creative Reel {idx + 1}</p>", unsafe_allow_html=True)
                else:
                    st.warning("Video asset not found.")
    else:
        st.info("No generated video reels found in the packaged payload.")

# 9. Main Visual Navigation Layout
tab_single, tab_bulk, tab_inspector = st.tabs([
    "🎯 Single URL Ingestion", 
    "📁 Bulk CSV Batch Processing", 
    "🔬 System Log & Metadata Inspector"
])

# ==========================================
# TAB 1: SINGLE URL INGESTION
# ==========================================
with tab_single:
    st.subheader("Launch Single Product Campaign")
    
    col_input, col_submit = st.columns([5, 1])
    with col_input:
        target_url = st.text_input(
            "Target Product Page URL",
            placeholder="e.g. https://www.amazon.com/dp/B08H75RTZ8",
            label_visibility="collapsed"
        )
    with col_submit:
        launch_btn = st.button("Launch Campaign", use_container_width=True)
        
    st.markdown("---")
    st.subheader("Or Load Completed Job Gallery")
    col_load_id, col_load_btn = st.columns([5, 1])
    with col_load_id:
        load_job_id = st.text_input("Enter Completed Job ID to load gallery", placeholder="e.g. e37f5aa4-81e1-4513-aa1c-287c92dd63da")
    with col_load_btn:
        load_btn = st.button("Load Completed Gallery", use_container_width=True)

    if load_btn and load_job_id:
        st.success(f"Loaded Gallery for Job ID: `{load_job_id}`")
        render_completed_gallery(load_job_id)

    if launch_btn:
        if not target_url or not (target_url.startswith("http://") or target_url.startswith("https://")):
            st.error("Input Validation Error: Please enter a valid URL beginning with 'http://' or 'https://'")
        else:
            with st.spinner("Submitting Creative Job to Ingestion Pipeline..."):
                job_id = trigger_single_job(target_url)

            if job_id:
                st.success(f"Creative Pipeline Triggered Successfully! Assigned Job ID: `{job_id}`")
                
                # Active Status Polling Container UI
                status_container = st.empty()
                timeline_container = st.empty()
                progress_bar = st.progress(0)
                
                status_logs = []
                last_node = None
                
                # Dynamic polling loop
                while True:
                    time.sleep(3)
                    job_data = fetch_job_details(job_id)
                    
                    if not job_data:
                        status_container.error("Failed to query status updates. Re-attempting...")
                        continue
                    
                    status = job_data.get("status", "pending").lower()
                    current_node = job_data.get("current_node", "queued")
                    retry_count = job_data.get("retry_count", 0)
                    
                    # Update horizontal timeline
                    with timeline_container:
                        render_orchestration_timeline(current_node)
                    
                    # Log state updates dynamically
                    node_msg = f"Executing Agent Node: `{current_node}` (Attempts: {retry_count})"
                    if current_node != last_node:
                        status_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Transited state to: **{current_node}**")
                        last_node = current_node
                        
                    # Calculate synthetic loader progress
                    progress_pct = 0
                    nodes = ["research", "strategy", "prompts", "image_gen", "video_gen", "critic", "packager"]
                    if current_node in nodes:
                        progress_pct = int((nodes.index(current_node) + 1) / len(nodes) * 100)
                    elif status == "completed":
                        progress_pct = 100
                    
                    progress_bar.progress(progress_pct)
                    
                    with status_container.container():
                        st.markdown(f"""
                        <div class="glass-card" style="border-left: 4px solid #7C3AED;">
                            <h4 style="margin: 0; color: #C084FC;">⚙️ Active Pipeline Processing Dashboard</h4>
                            <p style="margin: 8px 0; font-size: 0.95rem;">Current Node State: <b>{current_node}</b> | Status: <span class="status-badge running">{status}</span></p>
                            <p style="margin: 0; font-size: 0.85rem; color: #94A3B8;">Job ID: <code>{job_id}</code> | Quality Review Loops: <b>{retry_count}/2</b></p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("Detailed Orchestration Logs", expanded=True):
                            for log in reversed(status_logs):
                                st.markdown(log)

                    if status in ["completed", "failed"]:
                        break
                
                # Final evaluation updates
                final_job = fetch_job_details(job_id)
                status_container.empty()
                timeline_container.empty()
                progress_bar.empty()
                
                if final_job.get("status") == "completed":
                    st.balloons()
                    st.success("🎉 Marketing Creative Package successfully generated and validated!")
                    
                    # Consolidate download and options
                    st.markdown(f"""
                    <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid #10B981; border-radius: 12px; padding: 20px; margin-bottom: 25px; display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h4 style="margin: 0; color: #10B981;">Campaign Package Ready</h4>
                            <p style="margin: 4px 0 0 0; font-size: 0.88rem; color: #94A3B8;">ZIP consolidated package is compiled with metadata and visual assets.</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Fetch output ZIP path
                    download_endpoint = f"{BACKEND_URL}/api/v1/job/{job_id}/download"
                    st.markdown(f"[📥 Download Creative Asset ZIP Package]({download_endpoint})", unsafe_allow_html=True)
                    
                    # Retrieve files for gallery rendering
                    with st.spinner("Extracting media files for live rendering..."):
                        images, videos = extract_and_load_visuals(job_id)
                    
                    # Grid 1: 5 Generated Marketing Images
                    st.markdown("### 🎨 AI Generated Creative Image Campaign (FLUX.1-dev)")
                    if images:
                        cols_img = st.columns(5)
                        for idx, img_path in enumerate(images[:5]):
                            with cols_img[idx]:
                                if os.path.exists(img_path):
                                    st.image(img_path, caption=f"Creative Ad Visual {idx + 1}", use_container_width=True)
                                else:
                                    st.warning("Asset not found.")
                    else:
                        st.info("No generated PNG images found in the packaged payload.")

                    # Grid 2: 2 Marketing Videos
                    st.markdown("### 🎥 AI Generated Motion Video Ad Creatives (Wan 2.1 / LTX)")
                    if videos:
                        cols_vid = st.columns(2)
                        for idx, vid_path in enumerate(videos[:2]):
                            with cols_vid[idx]:
                                if os.path.exists(vid_path):
                                    st.video(vid_path)
                                    st.markdown(f"<p style='text-align: center; font-size: 0.85rem; color: #94A3B8;'>Motion Creative Reel {idx + 1}</p>", unsafe_allow_html=True)
                                else:
                                    st.warning("Video asset not found.")
                    else:
                        st.info("No generated video reels found in the packaged payload.")
                        
                else:
                    error_msg = final_job.get("error_message", "Uncaught Pipeline Exception.")
                    st.markdown(f"""
                    <div style="background: rgba(239, 68, 68, 0.08); border: 1px solid #EF4444; border-radius: 12px; padding: 25px; margin-bottom: 25px;">
                        <h4 style="margin: 0; color: #FCA5A5;">❌ Campaign Execution Failed</h4>
                        <p style="margin: 10px 0; font-size: 0.95rem; color: #FEE2E2;">The creative generation engine failed during orchestration. Details:</p>
                        <div style="background: rgba(0, 0, 0, 0.2); padding: 15px; border-radius: 8px; font-family: monospace; color: #FCA5A5;">
                            {error_msg}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

# ==========================================
# TAB 2: BULK CSV BATCH PROCESSING
# ==========================================
with tab_bulk:
    st.subheader("Bulk Creative CSV Upload Ingestion")
    
    st.markdown("""
    Upload a structured CSV containing bulk product target URLs. The pipeline will automatically 
    parse URLs, enforce asynchronous Celery queue rates (`3/m`), generate independent visual packages, 
    and store records under a single batch tracking index.
    """)
    
    csv_file = st.file_uploader("Upload Product URL Target Fleet", type=["csv"])
    col_header, _ = st.columns([2, 4])
    with col_header:
        url_header_col = st.text_input("CSV Target URL Column Header Key", value="url")
        
    trigger_bulk_btn = st.button("Trigger Fleet Campaign Generation", disabled=(csv_file is None))
    
    if trigger_bulk_btn and csv_file:
        try:
            # Simple pre-validation verify check
            csv_bytes = csv_file.read()
            df_test = pd.read_csv(io.BytesIO(csv_bytes))
            
            if url_header_col not in df_test.columns:
                st.error(f"Schema Alignment Error: Header column `{url_header_col}` was not found in the uploaded file.")
            else:
                with st.spinner("Submitting CSV fleet to batch processing endpoint..."):
                    batch_response = trigger_bulk_jobs(csv_bytes, csv_file.name)
                
                if batch_response:
                    batch_id = batch_response.get("batch_id")
                    job_ids = batch_response.get("job_ids", [])
                    st.success(f"Batch registered! Batch ID: `{batch_id}` with `{len(job_ids)}` jobs enqueued.")
                    
                    st.session_state["active_batch_id"] = batch_id
        except Exception as e:
            st.error(f"Error parsing uploaded target CSV: {e}")

    # Fleet Monitor Grid
    if "active_batch_id" in st.session_state:
        batch_id = st.session_state["active_batch_id"]
        st.markdown(f"### 📊 Real-Time Operational Fleet Monitor: `{batch_id}`")
        
        # Grid layout refresh
        batch_grid_container = st.empty()
        
        col_refresh, col_stop = st.columns([1, 5])
        with col_refresh:
            refresh_active = st.checkbox("Live Refresh", value=True)
            
        while True:
            batch_jobs = fetch_batch_details(batch_id)
            if not batch_jobs:
                batch_grid_container.warning("Failed to refresh batch statistics.")
                break
                
            # Create a structured Pandas DataFrame to show parameters
            grid_data = []
            completed_jobs = 0
            
            for job in batch_jobs:
                created_dt = datetime.fromisoformat(job["created_at"])
                if job.get("completed_at"):
                    completed_dt = datetime.fromisoformat(job["completed_at"])
                    elapsed = str(completed_dt - created_dt).split(".")[0]
                else:
                    elapsed = str(datetime.utcnow() - created_dt).split(".")[0]
                
                status = job["status"].upper()
                if status == "COMPLETED":
                    completed_jobs += 1
                
                grid_data.append({
                    "Job ID": job["job_id"],
                    "Target URL": job["url"],
                    "Status": status,
                    "Current Node Active": job["current_node"],
                    "Retry Attempts": job["retry_count"],
                    "Elapsed Time": elapsed,
                    "Download URL": f"{BACKEND_URL}/api/v1/job/{job['job_id']}/download" if status == "COMPLETED" else "N/A"
                })
                
            df_grid = pd.DataFrame(grid_data)
            
            with batch_grid_container.container():
                # Render visual stats widgets
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Jobs", len(df_grid))
                c2.metric("Completed", completed_jobs)
                c3.metric("Running/Pending", len(df_grid) - completed_jobs)
                
                # Interactive styled data editor grid
                st.data_editor(
                    df_grid,
                    column_config={
                        "Download URL": st.column_config.LinkColumn("Packaged ZIP Payload")
                    },
                    disabled=True,
                    use_container_width=True
                )
                
            if completed_jobs == len(df_grid) or not refresh_active:
                break
                
            time.sleep(4)

# ==========================================
# TAB 3: SYSTEM LOG & METADATA INSPECTOR
# ==========================================
with tab_inspector:
    st.subheader("Pipeline Audit Diagnostics & VLM Quality Scores")
    
    st.markdown("""
    Diagnostics dashboard designed to investigate execution payloads, cross-reference generated specs 
    against web-scraped targets, and inspect mutated inputs that demonstrate the LangGraph self-correction loop.
    """)
    
    search_job_id = st.text_input("Enter Job UUID Index to Query:", placeholder="e.g. 9f8b7a36-b603-44a3-a8e6-26c2d057ef71")
    
    if search_job_id:
        with st.spinner("Fetching system diagnostics records..."):
            job_details = fetch_job_details(search_job_id)
            
        if not job_details:
            st.error("Audit Query Failure: Job ID not found or could not connect to database backend.")
        else:
            st.markdown("### 🔍 System Audit Manifest")
            
            # Simple metadata details
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"**Job Status**: `{job_details.get('status', 'N/A').upper()}`")
            c2.markdown(f"**Target URL**: `{job_details.get('url', 'N/A')}`")
            c3.markdown(f"**Retry Count**: `{job_details.get('retry_count', 0)}`")
            c4.markdown(f"**Created Time**: `{job_details.get('created_at', 'N/A')}`")
            
            # Extract nested metadata state
            meta = job_details.get("metadata_data", {})
            
            if not meta:
                st.info("This job is currently in progress or failed before registering final metadata arrays.")
            else:
                # Plotting Visual Scores history
                scores_history = meta.get("critic_scores", [])
                
                if scores_history:
                    st.markdown("#### 📈 VLM Visual Evaluation Scores History")
                    # Visualise score improvements across retries
                    df_scores = pd.DataFrame({
                        "Attempt Loop": [f"Attempt {idx + 1}" for idx in range(len(scores_history))],
                        "Score Quality (1-10)": scores_history
                    })
                    st.area_chart(df_scores.set_index("Attempt Loop"))
                    
                    cols_score = st.columns(len(scores_history))
                    for idx, sc in enumerate(scores_history):
                        with cols_score[idx]:
                            st.metric(f"Attempt {idx + 1} Score", f"{sc:.2f} / 10.0")
                else:
                    st.info("No visual critic scores registered for this job record.")

                # Exploded JSON expanders representing each stage of the LangGraph orchestrator
                st.markdown("#### 🔬 Structured Component Exploder")

                # Section 1: Product Research
                with st.expander("Expandable Section 1: 🔍 Product Research Data Extraction (`product_data`)", expanded=False):
                    st.markdown("Scrapes HTML and markdown using Crawl4AI + BS4, mapping tags to structured Pydantic representations.")
                    st.json(meta.get("product_data", {}))

                # Section 2: Creative Strategy Brief
                with st.expander("Expandable Section 2: 💡 DTC Performance Strategy Copy & Angles (`creative_brief`)", expanded=False):
                    st.markdown("Analyzes products to formulate lifestyle, utility-focused, and minimalist e-commerce briefs.")
                    st.json(meta.get("creative_brief", {}))

                # Section 3: Prompts Factory
                with st.expander("Expandable Section 3: 📝 Refined Model Generation Prompts (`generation_prompts`)", expanded=False):
                    st.markdown("Generates detailed camera presets (85mm focal, ambient studio light) preserved for FLUX/Wan execution.")
                    st.json(meta.get("generation_prompts", {}))

                # Section 4: VLM Quality Control Loop
                with st.expander("Expandable Section 4: 👁️ Multimodal VLM Quality Assurance & Critique History", expanded=True):
                    st.markdown("Details the Llama 3.2 11B Vision structural critiques. If under 7.0, feedback is prepended directly to prompt nodes.")
                    c_feed = meta.get("critic_feedback")
                    if c_feed:
                        st.warning(f"**Latest Constructive Critique Feedback**:\n\n{c_feed}")
                    else:
                        st.success("**Visual Quality QA Passed**: Visual compositions met all brand specifications without critical defects.")
                    
                    st.markdown("**Complete Evaluation Logs History:**")
                    st.json({
                        "Scores History": scores_history,
                        "Latest Feedback Output": c_feed,
                        "Total Retry Attempts": meta.get("retry_count", 0)
                    })
