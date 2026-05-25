import logging
from typing import Dict, Any
from backend.graph.state import ProductState
from backend.agents.schemas import CreativeBrief
from backend.agents.llm import llm_manager

logger = logging.getLogger("strategy_node")

def strategy_node(state: ProductState) -> Dict[str, Any]:
    """
    Node 2: Creative Strategy Agent
    Employs a senior DTC performance marketing persona to analyze product features and specifications
    and formulate 3 highly targeted performance marketing angles (Aspirational, Problem-Solution, Minimalist).
    """
    product_data = state.get("product_data", {})
    title = product_data.get("title", "Product")
    brand = product_data.get("brand", "Unknown Brand")
    
    logger.info(f"[Node 2: Strategy] Formulating performance creative strategy for {brand} - {title}")
    
    # 1. Structure the raw product characteristics into clean prompt listings
    features_list = "\n".join([f"- {f}" for f in product_data.get("features", [])])
    specs_list = "\n".join([f"- {k}: {v}" for k, v in product_data.get("specs", {}).items()])
    price = product_data.get("price", "TBD")
    reviews = product_data.get("reviews_summary", "No reviews available.")
    tone = product_data.get("brand_tone", "modern, friendly, premium")
    
    # 2. Define marketing system instructions and creative brief prompt
    system_instruction = (
        "You are a world-class DTC Performance Marketing Director and elite copywriting executive. Your genius "
        "is translating raw, dry engineering specs and product specifications into scroll-stopping, high-converting "
        "creative angles. You understand consumer psychology, hook rates, visual styling, and social media copy layout."
    )
    
    prompt = (
        f"Create a premium social-first Creative Brief for the following e-commerce product:\n\n"
        f"Brand Name: {brand}\n"
        f"Product Title: {title}\n"
        f"Retail Price: {price}\n"
        f"Target Brand Tone: {tone}\n\n"
        f"Product Bullet Features:\n"
        f"{features_list}\n\n"
        f"Technical Specifications:\n"
        f"{specs_list}\n\n"
        f"Customer Feedback Summary:\n"
        f"{reviews}\n\n"
        f"Your task is to generate EXACTLY 3 DISTINCT creative angles to test in paid acquisition campaigns. "
        f"You must strictly define these angles:\n"
        f"1. A lifestyle/aspirational angle: Elevates the buyer's status, focuses on daily wellness rituals, aesthetic desks/homes, and emotional satisfaction.\n"
        f"2. A problem-solution/utility angle: Target direct friction points (e.g. skin irritation, sleep disruptions, sinus discomfort) with clear, analytical relief messaging.\n"
        f"3. A minimalist product-focused angle: Highlights premium materials, sleek macro zoom angles, whisper-quiet operations, and geometric perfection (Quiet Luxury vibe).\n\n"
        f"For each angle, provide:\n"
        f"- angle_type: One of 'lifestyle/aspirational', 'problem-solution/utility', or 'minimalist product-focused'.\n"
        f"- hook: A high-impact, scroll-stopping headline text (max 12 words).\n"
        f"- target_audience: A specific, hyper-targeted demographic descriptor (e.g. 'Eco-conscious design geeks working remotely').\n"
        f"- visual_mood: Highly detailed layout instructions specifying lighting, focus, environment, and physical details for photo/video tools.\n"
        f"- color_palette: A list of 3 matching color names that represent the psychological theme.\n"
        f"- caption: Direct-response marketing caption copy with formatting, spacing, call-to-actions, and relevant tags."
    )
    
    logger.info("[Node 2: Strategy] Launching Instructor LLM request (Llama 3.3 70B primary model) for CreativeBrief.")
    
    try:
        # Run using high-capacity model (Llama-3.3-70B) for deep marketing and strategy analysis
        brief_object: CreativeBrief = llm_manager.get_structured_completion(
            prompt=prompt,
            response_model=CreativeBrief,
            system_instruction=system_instruction,
            prefer_high_capacity=True
        )
        
        result_dict = brief_object.model_dump()
        logger.info(f"[Node 2: Strategy] Successfully generated {len(result_dict['angles'])} distinct creative angles.")
        return {"creative_brief": result_dict}
        
    except Exception as e:
        logger.error(f"[Node 2: Strategy] Strategy structured parser execution failed: {e}")
        raise
