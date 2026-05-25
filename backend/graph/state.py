from typing import TypedDict, List, Dict, Any, Annotated, Optional

def reduce_assets(left: Optional[List[str]], right: Optional[List[str]]) -> List[str]:
    """Reducer function to merge generated assets concurrently and remove duplicates."""
    if left is None:
        left = []
    if right is None:
        right = []
    res = list(left)
    for item in right:
        if item not in res:
            res.append(item)
    return res

class ProductState(TypedDict):
    """The shared state schema that flows through the LangGraph orchestrator."""
    # Input parameters
    url: str
    job_id: str
    retry_count: int                    # Max 2 retries before marking failed

    # Node outputs (accumulated as graph runs)
    product_data: Dict[str, Any]        # Node 1: title, features, specs, price, reviews, brand, product_images
    creative_brief: Dict[str, Any]      # Node 2: hooks, audience angles, visual themes, captions
    generation_prompts: Dict[str, Any]  # Node 3: 5 image prompts + 2 video prompts

    # Assets
    generated_assets: Annotated[List[str], reduce_assets]         # Filepaths or URIs of generated files

    # QA
    critic_scores: List[float]          # Score history across retry attempts
    critic_feedback: str                # VLM feedback payload injected back into Node 3

