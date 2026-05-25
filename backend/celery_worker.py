import os
import asyncio
import logging
import json
from celery import Celery
from backend.app import db_update_job_status
from backend.graph.graph import app_graph

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("celery_worker")

# Read Redis URL from environment
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Initialize Celery
celery_app = Celery(
    "tasks",
    broker=REDIS_URL,
    backend=REDIS_URL
)

# Optional Celery configurations
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
)

async def run_langgraph_pipeline(url: str, job_id: str):
    """Invokes the LangGraph state machine, streaming node transitions to SQLite in real-time."""
    initial_state = {
        "url": url,
        "job_id": job_id,
        "retry_count": 0,
        "product_data": {},
        "creative_brief": {},
        "generation_prompts": {},
        "generated_assets": [],
        "critic_scores": [],
        "critic_feedback": ""
    }
    
    final_state = initial_state
    
    # Compile execution options
    config = {"configurable": {"thread_id": job_id}}
    
    logger.info(f"Starting async LangGraph execution stream for Job: {job_id}")
    
    # Stream the graph execution nodes step-by-step
    async for event in app_graph.astream(initial_state, config=config, stream_mode="updates"):
        # event is a dictionary containing { node_name: state_updates }
        for node_name, updates in event.items():
            logger.info(f"=== [LangGraph Step] Completed Node: {node_name} ===")
            
            # Update the current executing node in SQLite database
            db_update_job_status(
                job_id=job_id,
                status="running",
                current_node=node_name
            )
            
            # Accumulate final state
            final_state.update(updates)
            
    return final_state

class RecoverableError(Exception):
    """Custom exception indicating a scrap or transient error where individual URL failure shouldn't crash worker."""
    pass

@celery_app.task(bind=True, rate_limit='3/m', max_retries=3, default_retry_delay=20)
def run_pipeline_task(self, url: str, job_id: str):
    """
    Celery task wrapping the asynchronous LangGraph execution.
    Features: rate limit of 3/min, error isolation, and automatic SQLite state synchronization.
    """
    logger.info(f"Starting Celery Pipeline Task for URL: {url} | Job ID: {job_id}")
    
    # 1. Update job to running status in database
    db_update_job_status(
        job_id=job_id,
        status="running",
        current_node="initializing"
    )
    
    try:
        # Run the async LangGraph orchestrator stream synchronously in the worker thread
        final_state = asyncio.run(run_langgraph_pipeline(url, job_id))
        
        # Extract packaged zip path and serialize final metadata
        assets = final_state.get("generated_assets", [])
        zip_path = None
        for asset in assets:
            if asset.endswith(".zip"):
                zip_path = asset
                break
                
        # Update database with successful completion and final metadata JSON
        db_update_job_status(
            job_id=job_id,
            status="completed",
            current_node="packaged",
            output_zip_path=zip_path,
            metadata_json=json.dumps(final_state)
        )
        logger.info(f"Successfully completed pipeline execution for Job {job_id}")
        return {"status": "success", "job_id": job_id}
        
    except RecoverableError as e:
        logger.error(f"Recoverable error in Job {job_id}: {e}")
        db_update_job_status(
            job_id=job_id,
            status="failed",
            current_node="error",
            error_message=str(e)
        )
        # Standard isolated failure: does not raise/retry, allows next queue rows to execute normally
        return {"status": "failed", "job_id": job_id, "error": str(e)}
        
    except Exception as e:
        logger.error(f"Uncaught exception running pipeline for Job {job_id}: {e}", exc_info=True)
        db_update_job_status(
            job_id=job_id,
            status="failed",
            current_node="error",
            error_message=str(e)
        )
        # Attempt to retry the Celery task on unexpected system-level crashes
        try:
            raise self.retry(exc=e)
        except Exception as retry_exc:
            logger.error(f"Max celery retries exceeded or task could not be retried: {retry_exc}")
            return {"status": "failed", "job_id": job_id, "error": str(e)}
