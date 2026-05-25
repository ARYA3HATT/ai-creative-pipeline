import os
import json
import zipfile
import logging
from typing import Dict, Any
from backend.graph.state import ProductState

logger = logging.getLogger("packager_node")

def packager_node(state: ProductState) -> Dict[str, Any]:
    """Node 7: Output Packager - consolidates all generated assets and metadata into a final ZIP package."""
    job_id = state.get("job_id")
    url = state.get("url")
    logger.info(f"[Node 7: Packager] Packaging outputs for job {job_id}")
    
    # Define paths
    job_dir = os.path.join(os.getenv("OUTPUT_DIR", "outputs"), job_id)
    images_dir = os.path.join(job_dir, "images")
    videos_dir = os.path.join(job_dir, "videos")
    
    metadata_path = os.path.join(job_dir, "metadata.json")
    zip_path = os.path.join(job_dir, "output.zip")
    
    # Extract score details and compute qa_status flag
    scores_history = state.get("critic_scores", [])
    retry_count = state.get("retry_count", 0)
    
    qa_status = "PASSED"
    if scores_history:
        final_score = scores_history[-1]
        if final_score < 7.0 and retry_count >= 2:
            qa_status = "FAILED_BUT_SHIPPED_EXHAUSTED"
            
    logger.info(f"[Node 7: Packager] Determined QA Status: {qa_status} based on {len(scores_history)} score attempts.")
    
    # Consolidate all metadata
    metadata = {
        "job_id": job_id,
        "source_url": url,
        "product_data": state.get("product_data", {}),
        "creative_brief": state.get("creative_brief", {}),
        "generation_prompts": state.get("generation_prompts", {}),
        "critic_scores_history": scores_history,
        "critic_final_feedback": state.get("critic_feedback", ""),
        "total_attempts": retry_count + 1,
        "qa_status": qa_status,
        "assets_generated": [
            os.path.basename(p) for p in state.get("generated_assets", [])
            if not p.endswith("output.zip")
        ]
    }
    
    # Save metadata.json
    try:
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"[Node 7: Packager] Consolidated metadata saved to {metadata_path}")
    except Exception as e:
        logger.error(f"[Node 7: Packager] Failed to save metadata.json: {e}")
        raise
        
    # Build ZIP archive containing metadata, images, and videos
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add metadata.json
            if os.path.exists(metadata_path):
                zipf.write(metadata_path, "metadata.json")
                
            # Add images
            if os.path.exists(images_dir):
                for file in os.listdir(images_dir):
                    file_path = os.path.join(images_dir, file)
                    if os.path.isfile(file_path):
                        zipf.write(file_path, os.path.join("images", file))
                        
            # Add videos
            if os.path.exists(videos_dir):
                for file in os.listdir(videos_dir):
                    file_path = os.path.join(videos_dir, file)
                    # Exclude preview frames from final package to keep it clean (or include if desired)
                    if os.path.isfile(file_path) and not file_path.endswith("_frame.png"):
                        zipf.write(file_path, os.path.join("videos", file))
                        
        logger.info(f"[Node 7: Packager] Outputs packaged successfully at {zip_path}")
    except Exception as e:
        logger.error(f"[Node 7: Packager] Failed to build ZIP archive: {e}")
        raise
        
    # Append the zip path to the generated assets list
    current_assets = list(state.get("generated_assets") or [])
    current_assets.append(zip_path)
    
    return {"generated_assets": current_assets}
