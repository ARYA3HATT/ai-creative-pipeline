import os
import json
import logging
import asyncio
import aiohttp
import base64
import hashlib
import requests
import urllib.parse
import time
import cv2
import numpy as np
from typing import Dict, Any, List, Optional
from backend.graph.state import ProductState
from backend.graph.nodes.comfy_client import comfy_client

logger = logging.getLogger("video_gen_node")

def _generate_pollinations_video_sync_and_save(video_prompt: str, video_save_path: str):
    """
    Synchronously generates video using Pollinations.ai,
    writes the binary stream to disk, and logs that Tier 2 has fired.
    If endpoint returns non-MP4 format or unexpected content type,
    it logs the actual response headers and content-type and reports back.
    """
    logger.info("[Pollinations Video] Tier 2 fired")
    encoded_prompt = urllib.parse.quote(video_prompt)
    url = f"https://gen.pollinations.ai/video/{encoded_prompt}"
    
    response = requests.get(url, timeout=300)
    # Check Content-Type and headers regardless of status code to report back
    content_type = response.headers.get("Content-Type", "")
    logger.info(f"[Pollinations Video] Response status: {response.status_code} | Content-Type: {content_type}")
    
    if response.status_code == 200:
        # If it is not an mp4 or video format, log and print headers
        if "video" not in content_type and "mp4" not in content_type:
            logger.warning(f"[Pollinations Video] Unexpected Content-Type received: {content_type}")
            logger.warning(f"[Pollinations Video] Response Headers: {dict(response.headers)}")
            
        with open(video_save_path, "wb") as f:
            f.write(response.content)
        logger.info(f"[Pollinations Video] Successfully saved generated video with HTTP 200 to {video_save_path}")
    else:
        logger.error(f"[Pollinations Video] Unexpected non-200 response returned from Pollinations.")
        logger.error(f"[Pollinations Video] Response Headers: {dict(response.headers)}")
        raise Exception(f"Pollinations video returned {response.status_code}: {response.text[:200]}")

def generate_procedural_video(prompt: str, filepath: str) -> bool:
    """
    Generates a unique, algorithmically distinct MP4 pattern video file based on the prompt.
    Uses cv2.VideoWriter with varying colors, text, and moving patterns to ensure they are distinct.
    """
    try:
        # Create a unique seed from the prompt
        seed = int(hashlib.md5(prompt.encode('utf-8')).hexdigest(), 16) % (2**32)
        np.random.seed(seed)

        # Set video parameters: 48 frames, 24 fps, 640x480 resolution (2 seconds)
        width, height = 640, 480
        fps = 24
        duration_frames = 48
        
        # Use mp4v codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
        if not out.isOpened():
            # If mp4v fails, try avc1
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))

        if not out.isOpened():
            logger.error("[Video Procedural] cv2.VideoWriter failed to open. Falling back to byte stub.")
            return False
        
        # Pick random distinct primary colors based on seed
        color_bg = [int(x) for x in np.random.randint(0, 100, 3)]  # Dark background
        color_pattern = [int(x) for x in np.random.randint(150, 255, 3)]  # Bright shape
        color_text = [int(x) for x in np.random.randint(200, 255, 3)]
        
        # Speed of movement
        speed_x = np.random.randint(3, 8)
        speed_y = np.random.randint(3, 8)
        
        # Draw frames
        for frame_idx in range(duration_frames):
            # Create bg
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :] = color_bg
            
            # Draw moving circle
            cx = int((width / 2 + frame_idx * speed_x) % width)
            cy = int((height / 2 + frame_idx * speed_y) % height)
            radius = int(50 + 20 * np.sin(frame_idx * 0.2))
            cv2.circle(frame, (cx, cy), radius, color_pattern, -1)
            
            # Draw moving grid lines
            for i in range(0, width, 80):
                offset = int((frame_idx * 2) % 80)
                cv2.line(frame, (i + offset, 0), (i + offset, height), (50, 50, 50), 1)
            for j in range(0, height, 80):
                offset = int((frame_idx * 2) % 80)
                cv2.line(frame, (0, j + offset), (width, j + offset), (50, 50, 50), 1)
            
            # Overlay prompt text snippet and frame info
            text_snippet = prompt[:30] + "..." if len(prompt) > 30 else prompt
            cv2.putText(frame, f"Prompt: {text_snippet}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_text, 2)
            cv2.putText(frame, f"Frame {frame_idx + 1}/{duration_frames}", (20, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            
            # Save frame
            out.write(frame)
            
        out.release()
        logger.info(f"[Video Procedural] Successfully generated unique procedural video at {filepath}")
        return True
    except Exception as e:
        logger.error(f"[Video Procedural] Failed to generate procedural video: {e}")
        return False

async def process_single_video(
    prompt: str,
    index: int,
    anchor_image_path: Optional[str],
    output_dir: str,
    comfy_online: bool,
    local_fallback: bool
) -> str:
    """Generates a single marketing video clip using ComfyUI, Pollinations.ai Video, or OpenCV procedural fallback."""
    filepath = os.path.join(output_dir, f"video_{index + 1}.mp4")
    
    # 1. Option A: ComfyUI (Primary production flow - if online and fallback disabled)
    if comfy_online and not local_fallback:
        logger.info(f"[Node 5: Video Gen] Attempting ComfyUI workflow for video {index + 1}")
        try:
            workflow_path = "backend/workflows/wan_video.json"
            if os.path.exists(workflow_path):
                with open(workflow_path, "r") as f:
                    workflow = json.load(f)
                    
                # A. Upload the anchor image frame if available from Node 4
                uploaded_name = None
                if anchor_image_path and os.path.exists(anchor_image_path):
                    logger.info(f"[Node 5: Video Gen] Uploading anchor frame image: {anchor_image_path}")
                    with open(anchor_image_path, "rb") as f:
                        img_bytes = f.read()
                        
                    async with aiohttp.ClientSession() as session:
                        data = aiohttp.FormData()
                        data.add_field('image', img_bytes, filename=f"anchor_{index + 1}.png", content_type='image/png')
                        async with session.post(f"{comfy_client.base_url}/upload/image", data=data) as upload_resp:
                            if upload_resp.status == 200:
                                upload_result = await upload_resp.json()
                                uploaded_name = upload_result.get("name")
                                
                # B. Locate nodes dynamically
                sampler_id = comfy_client.find_node_by_class(workflow, "KSampler")
                clip_ids = comfy_client.find_nodes_by_class(workflow, "CLIPTextEncode")
                load_img_id = comfy_client.find_node_by_class(workflow, "LoadImage")
                
                # C. Override inputs
                if load_img_id and uploaded_name:
                    workflow[load_img_id]["inputs"]["image"] = uploaded_name
                    
                for cid in clip_ids:
                    # Override positive prompt encoder
                    workflow[cid]["inputs"]["text"] = prompt
                    break
                    
                # D. Submit prompt and capture output MP4 bytes
                binaries = await comfy_client.submit_and_wait(workflow)
                if binaries:
                    with open(filepath, "wb") as f:
                        f.write(binaries[0])
                    logger.info(f"[Node 5: Video Gen] Successfully completed ComfyUI video prompt execution for video {index + 1}")
                    return filepath
            else:
                logger.error(f"[Node 5: Video Gen] wan_video.json template not found at {workflow_path}")
        except Exception as e:
            logger.error(f"[Node 5: Video Gen] ComfyUI video execution failed: {e}. Slipping to fallbacks.")

    # 2. Option B: Pollinations.ai Video Fallback (Tier 2)
    logger.info(f"[Node 5: Video Gen] Using Pollinations.ai Video Fallback for video {index + 1}")
    try:
        await asyncio.to_thread(_generate_pollinations_video_sync_and_save, prompt, filepath)
        return filepath
    except Exception as e:
        logger.error(f"[Node 5: Video Gen] Pollinations.ai video generation failed: {e}. Sliding to Tier 3 (OpenCV).")

    # 3. Option C: OpenCV Procedural Video Fallback (Tier 3)
    logger.warning(f"[Node 5: Video Gen] Triggering unique OpenCV procedural video generator for video {index + 1}")
    success = generate_procedural_video(prompt, filepath)
    
    if not success:
        # Failsafe unique binary stub creation if cv2 fails
        seed = int(hashlib.md5(prompt.encode('utf-8')).hexdigest(), 16) % (2**32)
        with open(filepath, "wb") as f:
            f.write(f"MOCK VIDEO DATA FOR PROMPT: {prompt}\nSEED: {seed}".encode('utf-8'))
        logger.critical(f"[Node 5: Video Gen] Created unique mock placeholder stub for video {index + 1} due to cv2/system failures.")
        
    return filepath

async def video_gen_node(state: ProductState) -> Dict[str, Any]:
    """
    Node 5: Video Generation Node
    Loads the ComfyUI Wan API workflow, uploads anchor frames from Node 4, and generates 2 videos.
    Falls back gracefully to Pollinations.ai video generator sequentially or OpenCV procedural MP4s.
    """
    job_id = state.get("job_id")
    prompts = state.get("generation_prompts", {}).get("video_prompts", [])
    generated_assets = state.get("generated_assets", [])
    
    logger.info(f"[Node 5: Video Gen] Initiating sequential generation of 2 ad videos for Job {job_id}")
    
    # Establish absolute local output path
    output_dir = os.path.join(os.getenv("OUTPUT_DIR", "outputs"), job_id, "videos")
    os.makedirs(output_dir, exist_ok=True)
    
    # Try to find a generated image from Node 4 to act as the starting anchor frame
    anchor_image = None
    for asset in generated_assets:
        if asset.endswith(".png") and "image_1" in asset:
            anchor_image = asset
            break
            
    # Verify client connectivity & read local environment fallback properties
    comfy_online = await comfy_client.check_connection()
    local_fallback = os.getenv("LOCAL_DEV_FALLBACK", "True").lower() == "true"
    
    logger.info(f"[Node 5: Video Gen] ComfyUI Connection Online: {comfy_online} | Local Fallback Enabled: {local_fallback}")
    
    # Trigger both async video tasks sequentially to comply with Pollinations rate-limiting
    generated_filepaths = []
    for idx, prompt in enumerate(prompts[:2]):
        filepath = await process_single_video(
            prompt=prompt,
            index=idx,
            anchor_image_path=anchor_image,
            output_dir=output_dir,
            comfy_online=comfy_online,
            local_fallback=local_fallback
        )
        generated_filepaths.append(filepath)
        await asyncio.sleep(2.0) # Graceful delay to permit Pollinations queue processing
        
    logger.info(f"[Node 5: Video Gen] Finished sequential video execution. Deployed {len(generated_filepaths)} files.")
    
    # Merge newly created video assets with the accumulated assets list
    current_assets = list(state.get("generated_assets") or [])
    current_assets.extend(generated_filepaths)
    
    return {"generated_assets": current_assets}
