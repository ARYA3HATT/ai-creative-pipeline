import os
import uuid
import logging
import csv
import httpx
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, HttpUrl

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backend_app")

# Ensure shared directory exists
os.makedirs("shared_data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///shared_data/jobs.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# SQLAlchemy Model
class JobRecord(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, index=True)
    batch_id = Column(String, index=True, nullable=True)
    url = Column(String, nullable=False)
    status = Column(String, default="pending")  # pending, running, completed, failed
    current_node = Column(String, default="queued")
    retry_count = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    output_zip_path = Column(String, nullable=True)
    metadata_json = Column(Text, nullable=True)  # Store serialized output state
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

# Create tables
Base.metadata.create_all(bind=engine)

# Pydantic Schemas
class GenerateRequest(BaseModel):
    url: str

class JobStatusResponse(BaseModel):
    job_id: str
    batch_id: Optional[str] = None
    url: str
    status: str
    current_node: str
    retry_count: int
    error_message: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None
    output_url: Optional[str] = None
    metadata_data: Optional[Dict[str, Any]] = None

# DB Dependency injection helper
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# FastAPI App setup
app = FastAPI(
    title="AI Product Creative Generation Pipeline API",
    description="Backend service orchestrating the creative research, design, generation, and packaging pipeline.",
    version="1.0.0"
)

# Enable CORS for frontend flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Active db update functions that Celery task can call
def db_update_job_status(job_id: str, status: str, current_node: str = None, error_message: str = None, output_zip_path: str = None, metadata_json: str = None):
    db = SessionLocal()
    try:
        job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
        if job:
            job.status = status
            if current_node:
                job.current_node = current_node
            if error_message:
                job.error_message = error_message
            if output_zip_path:
                job.output_zip_path = output_zip_path
            if metadata_json:
                job.metadata_json = metadata_json
            if status in ["completed", "failed"]:
                job.completed_at = datetime.utcnow()
            db.commit()
            logger.info(f"DB Job {job_id} status updated to {status} ({current_node})")
    except Exception as e:
        logger.error(f"DB update failed for job {job_id}: {e}")
    finally:
        db.close()

async def validate_url_preflight(url: str) -> None:
    """
    Rapid pre-flight check of the target URL.
    Verifies that a GET request returns HTTP 200 and the content is longer than 500 characters.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            response = await client.get(url, headers=headers)
            if response.status_code != 200:
                raise HTTPException(
                    status_code=400,
                    detail=f"URL pre-flight validation failed: Server returned status code {response.status_code} instead of 200."
                )
            
            content_len = len(response.text)
            if content_len < 500:
                raise HTTPException(
                    status_code=400,
                    detail=f"URL pre-flight validation failed: Target page text content length is too short ({content_len} characters, expected > 500)."
                )
    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=400,
            detail=f"URL pre-flight validation failed: Network connection error when reaching {url}. Details: {str(e)}"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"URL pre-flight validation failed: Unexpected error. Details: {str(e)}"
        )

@app.post("/api/v1/generate", response_model=Dict[str, str])
async def generate_creative(payload: GenerateRequest, db: Session = Depends(get_db)):
    """Triggers creative pipeline asynchronously for a single product URL after pre-flight checks."""
    url = str(payload.url)
    
    # Run pre-flight checks to protect downstream worker threads and quota
    await validate_url_preflight(url)
    
    job_id = str(uuid.uuid4())
    
    # Save record to database
    job = JobRecord(id=job_id, url=url, status="pending", current_node="queued")
    db.add(job)
    db.commit()
    
    # Delay import to avoid circular dependency with celery_app definition
    from backend.celery_worker import run_pipeline_task
    run_pipeline_task.delay(url, job_id)
    
    logger.info(f"Enqueued single generate job: {job_id} for URL {url}")
    return {"job_id": job_id}

@app.post("/api/v1/bulk")
async def bulk_generate_creatives(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Triggers creative pipeline for multiple URLs in a uploaded CSV file after parallel pre-flight validation."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a CSV file.")
    
    # Read URLs from CSV file
    content = await file.read()
    decoded = content.decode('utf-8').splitlines()
    reader = csv.reader(decoded)
    
    urls = []
    for row in reader:
        if not row:
            continue
        # Support either simple list of URLs or headered URL column
        url = row[0].strip()
        if url.startswith("http://") or url.startswith("https://"):
            urls.append(url)
            
    if not urls:
        raise HTTPException(status_code=400, detail="No valid URLs found in the uploaded CSV file.")
        
    # Pre-flight check all parsed URLs in parallel to be fast and non-blocking
    # Catch exceptions per URL to avoid failing the entire batch upload
    results = await asyncio.gather(*[validate_url_preflight(url) for url in urls], return_exceptions=True)
        
    batch_id = str(uuid.uuid4())
    job_ids = []
    
    # Import celery worker task
    from backend.celery_worker import run_pipeline_task
    
    # Process each row with standard error isolation
    for url, result in zip(urls, results):
        job_id = str(uuid.uuid4())
        job_ids.append(job_id)
        
        if isinstance(result, Exception):
            # Pre-flight check failed for this URL. Mark as failed in DB immediately
            error_msg = str(result.detail) if hasattr(result, 'detail') else str(result)
            job = JobRecord(
                id=job_id,
                batch_id=batch_id,
                url=url,
                status="failed",
                current_node="pre-flight",
                error_message=f"Pre-flight validation failed: {error_msg}",
                completed_at=datetime.utcnow()
            )
            db.add(job)
            db.commit()
            logger.warning(f"Bulk job {job_id} failed pre-flight check for URL {url}: {error_msg}")
        else:
            # Pre-flight passed! Save pending job record and enqueue on Celery
            job = JobRecord(id=job_id, batch_id=batch_id, url=url, status="pending", current_node="queued")
            db.add(job)
            db.commit()
            
            # Queue the job independently on Celery
            run_pipeline_task.delay(url, job_id)
        
    logger.info(f"Enqueued bulk batch: {batch_id} with {len(job_ids)} jobs.")
    return {"batch_id": batch_id, "job_ids": job_ids}

@app.get("/api/v1/job/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str, db: Session = Depends(get_db)):
    """Fetches full status, current node progression, and metadata payload for a job."""
    job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
        
    # Serialize metadata for log inspector
    import json
    metadata_data = None
    if job.metadata_json:
        try:
            metadata_data = json.loads(job.metadata_json)
        except Exception:
            metadata_data = {"raw_text": job.metadata_json}
            
    output_url = None
    if job.status == "completed" and job.output_zip_path:
        output_url = f"/api/v1/job/{job_id}/download"
        
    return JobStatusResponse(
        job_id=job.id,
        batch_id=job.batch_id,
        url=job.url,
        status=job.status,
        current_node=job.current_node,
        retry_count=job.retry_count,
        error_message=job.error_message,
        created_at=job.created_at.isoformat(),
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        output_url=output_url,
        metadata_data=metadata_data
    )

@app.get("/api/v1/job/{job_id}/download")
def download_job_output(job_id: str, db: Session = Depends(get_db)):
    """Downloads packaged output ZIP containing generated images, videos, and metadata JSON."""
    job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
        
    if job.status != "completed" or not job.output_zip_path or not os.path.exists(job.output_zip_path):
        raise HTTPException(status_code=400, detail="Output package not available or job has not completed successfully.")
        
    filename = f"creative_pack_{job_id[:8]}.zip"
    return FileResponse(
        job.output_zip_path,
        media_type="application/zip",
        filename=filename
    )

@app.get("/api/v1/batch/{batch_id}", response_model=List[JobStatusResponse])
def get_batch_status(batch_id: str, db: Session = Depends(get_db)):
    """Fetches job status details for all jobs listed in a CSV bulk processing batch."""
    jobs = db.query(JobRecord).filter(JobRecord.batch_id == batch_id).all()
    
    response = []
    for job in jobs:
        import json
        metadata_data = None
        if job.metadata_json:
            try:
                metadata_data = json.loads(job.metadata_json)
            except Exception:
                pass
                
        output_url = None
        if job.status == "completed" and job.output_zip_path:
            output_url = f"/api/v1/job/{job.id}/download"
            
        response.append(
            JobStatusResponse(
                job_id=job.id,
                batch_id=job.batch_id,
                url=job.url,
                status=job.status,
                current_node=job.current_node,
                retry_count=job.retry_count,
                error_message=job.error_message,
                created_at=job.created_at.isoformat(),
                completed_at=job.completed_at.isoformat() if job.completed_at else None,
                output_url=output_url,
                metadata_data=metadata_data
            )
        )
        
    return response
