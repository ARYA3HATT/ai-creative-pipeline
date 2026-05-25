import logging
from typing import Dict, Any
from backend.graph.state import ProductState
from backend.agents.schemas import PromptSet
from backend.agents.llm import llm_manager

logger = logging.getLogger("prompts_node")

def prompts_node(state: ProductState) -> Dict[str, Any]:
    """
    Node 3: Prompt Generation Agent
    Translates e-commerce marketing briefs into 5 highly specific FLUX.1-dev image prompts
    and 2 motion-parameterized Wan 2.1 video prompts.
    Integrates a dynamic feedback loop that modifies prompt synthesis if previous attempts were critiqued by the VLM.
    """
    creative_brief = state.get("creative_brief", {})
    product_data = state.get("product_data", {})
    title = product_data.get("title", "Product")
    critic_feedback = state.get("critic_feedback", "")
    retry_count = state.get("retry_count", 0)
    
    logger.info(f"[Node 3: Prompts] Initiating prompt generation for {title} (Attempt {retry_count + 1})")
    
    # 1. format the angles list for clear LLM presentation
    angles_prompt_text = ""
    for idx, angle in enumerate(creative_brief.get("angles", [])):
        angles_prompt_text += (
            f"Creative Angle {idx + 1} ({angle.get('angle_type')}):\n"
            f"  - Hook Headline: \"{angle.get('hook')}\"\n"
            f"  - Target Audience: {angle.get('target_audience')}\n"
            f"  - Visual Mood & Setting: {angle.get('visual_mood')}\n"
            f"  - Associated Color Theme: {', '.join(angle.get('color_palette', []))}\n"
            f"  - Ad Social Copy: \"{angle.get('caption')}\"\n\n"
        )
        
    # 2. Establish strict visual styling guidelines for image and video diffusion models
    system_instruction = (
        "You are an elite AI Art Director, senior Prompt Engineer, and digital advertising cinematographer. "
        "Your expertise is formulating mathematical, photorealistic, and highly stylized visual prompts for "
        "diffusion models (FLUX.1-dev and Wan 2.1). You know how to structure prompts using professional photographic "
        "terms: lens lengths (e.g. 85mm, 35mm), lighting types (e.g. key light, volumetric mist, high-key, low-key, "
        "soft diffuse), camera directions (e.g. orbital pan, dolly-in, slow tilt), styling keywords, and negative keywords."
    )
    
    prompt = (
        f"We are launching an ad campaign for the product: {title}.\n"
        f"You must translate the following marketing Creative Brief into structured prompts:\n\n"
        f"{angles_prompt_text}\n"
        f"Please generate exactly 5 Image Prompts and exactly 2 Video Prompts matching these exact rules:\n\n"
        f"--- IMAGE PROMPTS (5 REQUIRED) ---\n"
        f"- Format each prompt as a single cohesive string. Structure it in this format: "
        f"[Detailed Scene Description], [Studio Lighting and Focus details], [Visual Mood/Aesthetics], "
        f"[Style Direction - e.g. 'award-winning commercial advertising photography, photorealistic, sharp focus, 8k resolution'], "
        f"negative prompt: [undesirables to filter out, e.g. 'distorted, blurry, low resolution, cheap plastic texture'].\n"
        f"- CRITICAL REQUIREMENT: Each of the 5 prompts MUST strictly contain this exact literal anchor string: "
        f"\"product: {title}, preserve exact product shape and color\" to guarantee IP-Adapter consistent rendering.\n"
        f"- Allocation: Create 1 prompt for Angle 1, 1 prompt for Angle 2, 1 prompt for Angle 3, and then 2 additional "
        f"visually distinct variations for the strongest angle to explore alternative layouts (e.g. macro zoom, spa setting variation).\n\n"
        f"--- VIDEO PROMPTS (2 REQUIRED) ---\n"
        f"- Structure with concrete cinematic camera motion language (e.g. 'slow dolly-in', '360 degree smooth orbital panning', "
        f"'slow tilt-up showing mist escape', 'crane-down with soft depth-of-field').\n"
        f"- Must specify movement speed ('slow', 'gentle'), volumetric parameters, studio soft lighting, "
        f"and target duration ('6 seconds' or '8 seconds').\n"
        f"- Ensure they feel high-end, smooth, and cohesive with the product's brand tone: \"{product_data.get('brand_tone')}\".\n"
    )
    
    # 3. Dynamic Critic Feedback Injection (Feedback loop correction)
    if critic_feedback:
        logger.warning(f"[Node 3: Prompts] QA Feedback loop detected! Injecting VLM review constraints.")
        prompt += (
            f"\n\n"
            f"========================================================================\n"
            f"⚠️ CRITICAL: PREVIOUS ATTEMPT FAILED QA EVALUATION\n"
            f"The prior prompts generated assets that did not meet quality standards.\n"
            f"Multimodal VLM Critic Feedback from QA Review (Attempt {retry_count}):\n"
            f"\"\"\"\n{critic_feedback}\n\"\"\"\n\n"
            f"You MUST read this feedback closely and completely reformulate your prompt definitions to resolve "
            f"these errors! Adjust the lighting, settings, background elements, or focal lengths as demanded by the critic."
            f"========================================================================\n"
        )
        
    logger.info("[Node 3: Prompts] Invoking structured completion (Llama 3.3 70B primary model) for PromptSet.")
    
    try:
        # Run using high-capacity model (Llama-3.3-70B) for deep art direction and precise prompt engineering
        prompts_object: PromptSet = llm_manager.get_structured_completion(
            prompt=prompt,
            response_model=PromptSet,
            system_instruction=system_instruction,
            prefer_high_capacity=True
        )
        
        result_dict = prompts_object.model_dump()
        
        # Log prompt examples for transparency
        logger.info(f"[Node 3: Prompts] Prompts created successfully.")
        logger.info(f"Image Prompt 1 snippet: {result_dict['image_prompts'][0][:80]}...")
        logger.info(f"Video Prompt 1 snippet: {result_dict['video_prompts'][0][:80]}...")
        
        return {"generation_prompts": result_dict}
        
    except Exception as e:
        logger.error(f"[Node 3: Prompts] Prompts generation structured parsing failed: {e}")
        raise
