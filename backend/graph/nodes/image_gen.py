import os
import json
import logging
import asyncio
import aiohttp
import time
import requests
import urllib.parse
from typing import Dict, Any, List, Optional
from backend.graph.state import ProductState
from backend.graph.nodes.comfy_client import comfy_client

logger = logging.getLogger("image_gen_node")

def _generate_pollinations_sync_and_save(image_prompt: str, image_save_path: str):
    """
    Synchronously generates an image via Pollinations.ai FLUX model,
    writes the binary stream to disk, and logs that Tier 2 has fired.
    Includes a retry loop to handle temporary concurrency or queue blocks.
    """
    encoded_prompt = urllib.parse.quote(image_prompt)
    url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&model=flux&nologo=true"
    
    max_attempts = 4
    for attempt in range(1, max_attempts + 1):
        logger.info(f"[Pollinations] Tier 2 fired - Attempt {attempt}/{max_attempts}")
        try:
            response = requests.get(url, timeout=120)
            if response.status_code == 200:
                with open(image_save_path, "wb") as f:
                    f.write(response.content)
                logger.info(f"[Pollinations] Successfully saved generated image with HTTP 200 to {image_save_path}")
                return
            elif response.status_code == 402 or "Queue full" in response.text:
                wait_time = 4 * attempt
                logger.warning(f"[Pollinations] Queue full (402). Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise Exception(f"Pollinations.ai returned status code {response.status_code}: {response.text[:200]}")
        except Exception as e:
            if attempt == max_attempts:
                raise e
            logger.warning(f"[Pollinations] Attempt {attempt} failed: {e}. Retrying...")
            time.sleep(3)

async def download_mock_unsplash(session: aiohttp.ClientSession, url: str) -> Optional[bytes]:
    """Helper to fetch a pre-curated photo if all AI generation tools fail."""
    try:
        async with session.get(url, timeout=15) as resp:
            if resp.status == 200:
                return await resp.read()
    except Exception as e:
        logger.error(f"[Image Gen Fallback] Unsplash mock retrieval failed: {e}")
    return None

async def process_single_image(
    prompt: str,
    index: int,
    ref_image_url: str,
    output_dir: str,
    comfy_online: bool,
    local_fallback: bool
) -> str:
    """Generates a single product campaign image using the best available tool, saving to disk."""
    filepath = os.path.join(output_dir, f"image_{index + 1}.png")
    
    # Pre-curated photo URLs for fallback
    unsplash_urls = [
        "https://images.unsplash.com/photo-1522335789203-aabd1fc54bc9?q=80&w=800&auto=format&fit=crop",
        "https://images.unsplash.com/photo-1515378791036-0648a3ef77b2?q=80&w=800&auto=format&fit=crop",
        "https://images.unsplash.com/photo-1513151233558-d860c5398176?q=80&w=800&auto=format&fit=crop",
        "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?q=80&w=800&auto=format&fit=crop",
        "https://images.unsplash.com/photo-1540555700478-4be289fbecef?q=80&w=800&auto=format&fit=crop"
    ]
    unsplash_url = unsplash_urls[index % len(unsplash_urls)]
    
    # 1. Option A: ComfyUI (Primary production flow - if online and fallback disabled)
    if comfy_online and not local_fallback:
        logger.info(f"[Node 4: Image Gen] Attempting ComfyUI workflow for image {index + 1}")
        try:
            # Load the exported API JSON template
            workflow_path = "backend/workflows/flux_image.json"
            if os.path.exists(workflow_path):
                with open(workflow_path, "r") as f:
                    workflow = json.load(f)
                    
                # A. Upload the target product image to ComfyUI input folder
                uploaded_name = await comfy_client.upload_image(ref_image_url)
                if uploaded_name:
                    # B. Locate nodes dynamically by Class Type
                    sampler_id = comfy_client.find_node_by_class(workflow, "KSampler")
                    clip_ids = comfy_client.find_nodes_by_class(workflow, "CLIPTextEncode")
                    load_img_id = comfy_client.find_node_by_class(workflow, "LoadImage")
                    
                    # C. Override prompt parameters and loader values
                    if load_img_id:
                        workflow[load_img_id]["inputs"]["image"] = uploaded_name
                        
                    # Find KSampler positive clip node (usually node 6 in default)
                    # We override the positive text encoder
                    for cid in clip_ids:
                        workflow[cid]["inputs"]["text"] = prompt
                        break # Update positive prompt
                        
                    # D. Submit and pull output bytes
                    binaries = await comfy_client.submit_and_wait(workflow)
                    if binaries:
                        with open(filepath, "wb") as f:
                            f.write(binaries[0])
                        logger.info(f"[Node 4: Image Gen] Successfully completed ComfyUI prompt execution for image {index + 1}")
                        return filepath
            else:
                logger.error(f"[Node 4: Image Gen] flux_image.json template not found at {workflow_path}")
        except Exception as e:
            logger.error(f"[Node 4: Image Gen] ComfyUI generation failed: {e}. Sliding to fallbacks.")
 
    # 2. Option B: Pollinations.ai FLUX Fallback
    logger.info(f"[Node 4: Image Gen] Using Pollinations.ai FLUX Fallback for image {index + 1}")
    try:
        await asyncio.to_thread(_generate_pollinations_sync_and_save, prompt, filepath)
        return filepath
    except Exception as e:
        logger.error(f"[Node 4: Image Gen] Pollinations.ai generation failed: {e}. Sliding to Unsplash.")

    # 3. Option C: Unsplash Mock Downloader (Resilient static fallback)
    logger.warning(f"[Node 4: Image Gen] AI tools offline. Running Unsplash static downloader for image {index + 1}")
    try:
        async with aiohttp.ClientSession() as session:
            image_bytes = await download_mock_unsplash(session, unsplash_url)
            if image_bytes:
                with open(filepath, "wb") as f:
                    f.write(image_bytes)
                logger.info(f"[Node 4: Image Gen] Saved Unsplash fallback image to {filepath}")
                return filepath
    except Exception as e:
        logger.error(f"[Node 4: Image Gen] Unsplash fallback downloader failed: {e}")

    # Extreme backup stub creation
    with open(filepath, "wb") as f:
        f.write(b"MOCK IMAGE BYTES FAILSAFE")
    logger.critical(f"[Node 4: Image Gen] Image {index + 1} written as blank files due to complete downstream network outage.")
    return filepath

async def image_gen_node(state: ProductState) -> Dict[str, Any]:
    """
    Node 4: Image Generation Node
    Loads the ComfyUI FLUX workflow API, overrides prompts, uploads product seeds, and generates 5 images.
    Falls back gracefully to Pollinations.ai or Unsplash mocks during local dev.
    """
    job_id = state.get("job_id")
    prompts = state.get("generation_prompts", {}).get("image_prompts", [])
    product_data = state.get("product_data", {})
    
    # Isolate first product image URL as seed reference for IP-Adapter consistency
    ref_images = product_data.get("product_images", [])
    ref_image_url = ref_images[0] if ref_images else "https://images.unsplash.com/photo-1523275335684-37898b6baf30?q=80&w=800"
    
    logger.info(f"[Node 4: Image Gen] Initiating sequential generation of 5 ad images for Job {job_id}")
    
    # Establish absolute local output path
    output_dir = os.path.join(os.getenv("OUTPUT_DIR", "outputs"), job_id, "images")
    os.makedirs(output_dir, exist_ok=True)
    
    # Verify client connectivity & read local environment fallback properties
    comfy_online = await comfy_client.check_connection()
    local_fallback = os.getenv("LOCAL_DEV_FALLBACK", "True").lower() == "true"
    
    logger.info(f"[Node 4: Image Gen] ComfyUI Connection Online: {comfy_online} | Local Fallback Enabled: {local_fallback}")
    
    # Trigger 5 async tasks sequentially to comply with Pollinations rate-limiting
    generated_filepaths = []
    for idx, prompt in enumerate(prompts[:5]):
        filepath = await process_single_image(
            prompt=prompt,
            index=idx,
            ref_image_url=ref_image_url,
            output_dir=output_dir,
            comfy_online=comfy_online,
            local_fallback=local_fallback
        )
        generated_filepaths.append(filepath)
        await asyncio.sleep(2.0) # Graceful delay to permit Pollinations queue processing
        
    logger.info(f"[Node 4: Image Gen] Finished sequential image execution. Deployed {len(generated_filepaths)} files.")
    
    # Merge newly created assets with the accumulated state assets list
    current_assets = list(state.get("generated_assets") or [])
    current_assets.extend(generated_filepaths)
    
    return {"generated_assets": current_assets}
