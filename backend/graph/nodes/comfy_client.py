import os
import uuid
import json
import logging
import asyncio
import aiohttp
import websockets
from typing import Dict, Any, Optional, List

logger = logging.getLogger("comfy_client")

class ComfyUIClient:
    """Async Client for ComfyUI REST and WebSocket API."""
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (base_url or os.getenv("COMFYUI_URL", "http://localhost:8188")).rstrip("/")
        # Extract host and port for websocket connection
        # e.g., http://localhost:8188 -> localhost:8188
        self.ws_host = self.base_url.replace("http://", "").replace("https://", "")
        self.client_id = str(uuid.uuid4())
        
    async def check_connection(self) -> bool:
        """Verifies if the ComfyUI server is reachable."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/object_info", timeout=3) as r:
                    return r.status == 200
        except Exception:
            return False

    async def upload_image(self, image_url: str) -> Optional[str]:
        """Downloads an image from a URL and uploads it directly to the ComfyUI input folder."""
        logger.info(f"[ComfyUI Client] Downloading product reference image: {image_url}")
        try:
            async with aiohttp.ClientSession() as session:
                # 1. Fetch raw image bytes from URL
                async with session.get(image_url, timeout=15) as img_resp:
                    if img_resp.status != 200:
                        logger.error(f"[ComfyUI Client] Failed to download image from {image_url}: {img_resp.status}")
                        return None
                    image_bytes = await img_resp.read()
                    
                # 2. Extract original name or generate unique name
                filename = f"ref_{uuid.uuid4().hex[:8]}.png"
                
                # 3. POST multipart request to ComfyUI upload endpoint
                data = aiohttp.FormData()
                data.add_field('image', image_bytes, filename=filename, content_type='image/png')
                
                logger.info(f"[ComfyUI Client] Uploading reference image {filename} to ComfyUI input folder...")
                async with session.post(f"{self.base_url}/upload/image", data=data) as upload_resp:
                    if upload_resp.status == 200:
                        upload_result = await upload_resp.json()
                        logger.info(f"[ComfyUI Client] Reference image successfully uploaded: {upload_result.get('name')}")
                        return upload_result.get('name') # Returns ComfyUI relative input name
                    else:
                        logger.error(f"[ComfyUI Client] Upload failed with status: {upload_resp.status}")
                        return None
        except Exception as e:
            logger.error(f"[ComfyUI Client] Exception encountered uploading image: {e}")
            return None

    def find_node_by_class(self, workflow: Dict[str, Any], class_type: str) -> Optional[str]:
        """Dynamically finds a ComfyUI node ID by its registered Class Type (e.g. KSampler, CLIPTextEncode)."""
        for node_id, node_config in workflow.items():
            if node_config.get("class_type") == class_type:
                return node_id
        return None

    def find_nodes_by_class(self, workflow: Dict[str, Any], class_type: str) -> List[str]:
        """Dynamically finds all node IDs of a certain class type in the workflow."""
        return [nid for nid, cfg in workflow.items() if cfg.get("class_type") == class_type]

    async def submit_and_wait(self, workflow: Dict[str, Any], output_node_class: str = "SaveImage") -> List[bytes]:
        """
        Submits a custom workflow JSON payload to ComfyUI and blocks until completion.
        Utilizes high-speed WebSockets to track execution states with an HTTP polling fallback.
        """
        logger.info("[ComfyUI Client] Submitting generation prompt workflow...")
        payload = {
            "prompt": workflow,
            "client_id": self.client_id
        }
        
        async with aiohttp.ClientSession() as session:
            # 1. POST prompt to queue
            async with session.post(f"{self.base_url}/prompt", json=payload) as resp:
                if resp.status != 200:
                    resp_text = await resp.text()
                    raise RuntimeError(f"ComfyUI rejected prompt: {resp_text}")
                result = await resp.json()
                prompt_id = result.get("prompt_id")
                logger.info(f"[ComfyUI Client] Prompt queued successfully. Queue Prompt ID: {prompt_id}")
                
            # 2. Connect to WebSocket to wait for prompt execution sequence
            ws_url = f"ws://{self.ws_host}/ws?clientId={self.client_id}"
            completed = False
            
            try:
                logger.info(f"[ComfyUI Client] Connecting to WebSocket: {ws_url}")
                async with websockets.connect(ws_url, timeout=10) as websocket:
                    while not completed:
                        message_str = await websocket.recv()
                        message = json.loads(message_str)
                        msg_type = message.get("type")
                        
                        if msg_type == "executing":
                            data = message.get("data", {})
                            current_node = data.get("node")
                            exec_prompt_id = data.get("prompt_id")
                            
                            if exec_prompt_id == prompt_id:
                                if current_node is None:
                                    logger.info(f"[ComfyUI Client] WebSocket indicates queue completion for Prompt {prompt_id}")
                                    completed = True
                                else:
                                    logger.info(f"[ComfyUI Client] ComfyUI executing Node ID: {current_node}")
                                    
                        elif msg_type == "executed":
                            data = message.get("data", {})
                            exec_prompt_id = data.get("prompt_id")
                            if exec_prompt_id == prompt_id:
                                logger.info(f"[ComfyUI Client] Execution completed for Prompt ID: {prompt_id}")
                                completed = True
            except Exception as ws_err:
                logger.warning(f"[ComfyUI Client] WebSocket connection failed or interrupted: {ws_err}. Slipping to HTTP Polling fallback.")
                
            # 3. HTTP Polling Fallback (runs if WebSocket fails or disconnects)
            if not completed:
                logger.info("[ComfyUI Client] Polling /history API for completion...")
                for poll_attempt in range(120): # Cap at 10 minutes (120 * 5s)
                    await asyncio.sleep(5)
                    async with session.get(f"{self.base_url}/history/{prompt_id}") as hist_resp:
                        if hist_resp.status == 200:
                            hist_data = await hist_resp.json()
                            if prompt_id in hist_data:
                                logger.info(f"[ComfyUI Client] Polling confirms execution completion for {prompt_id}")
                                completed = True
                                break
                if not completed:
                    raise TimeoutError("ComfyUI generation task timed out.")

            # 4. Fetch the final output images / videos from /history
            async with session.get(f"{self.base_url}/history/{prompt_id}") as final_hist_resp:
                if final_hist_resp.status != 200:
                    raise RuntimeError("Failed to fetch execution history metadata.")
                history_payload = await final_hist_resp.json()
                prompt_history = history_payload.get(prompt_id, {})
                outputs = prompt_history.get("outputs", {})
                
                output_files = []
                for node_id, node_outputs in outputs.items():
                    # We check for images or gifs or animations
                    if "images" in node_outputs:
                        for img_meta in node_outputs["images"]:
                            filename = img_meta.get("filename")
                            subfolder = img_meta.get("subfolder", "")
                            folder_type = img_meta.get("type", "input")
                            output_files.append((filename, subfolder, folder_type))
                    elif "gifs" in node_outputs:
                        for gif_meta in node_outputs["gifs"]:
                            filename = gif_meta.get("filename")
                            subfolder = gif_meta.get("subfolder", "")
                            folder_type = gif_meta.get("type", "input")
                            output_files.append((filename, subfolder, folder_type))
                            
                if not output_files:
                    logger.warning("[ComfyUI Client] No generated files found in history outputs dictionary.")
                    return []
                    
                # Download each file and return raw binary lists
                binaries = []
                for name, sub, ftype in output_files:
                    params = {"filename": name, "subfolder": sub, "type": ftype}
                    logger.info(f"[ComfyUI Client] Fetching file from ComfyUI: {name}")
                    async with session.get(f"{self.base_url}/view", params=params) as view_resp:
                        if view_resp.status == 200:
                            file_bytes = await view_resp.read()
                            binaries.append(file_bytes)
                            
                return binaries

# Global ComfyUI client instance
comfy_client = ComfyUIClient()
