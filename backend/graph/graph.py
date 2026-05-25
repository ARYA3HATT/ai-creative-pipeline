import logging
from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END

# Import the shared state schema
from backend.graph.state import ProductState

# Import node functions
from backend.graph.nodes.research import research_node
from backend.graph.nodes.strategy import strategy_node
from backend.graph.nodes.prompts import prompts_node
from backend.graph.nodes.image_gen import image_gen_node
from backend.graph.nodes.video_gen import video_gen_node
from backend.graph.nodes.critic import critic_node
from backend.graph.nodes.packager import packager_node

logger = logging.getLogger("graph_orchestrator")

def should_retry(state: ProductState) -> Literal["prompts", "packager"]:
    """Conditional router determining if the output fails critique and needs refinement."""
    scores = state.get("critic_scores", [])
    retry_count = state.get("retry_count", 0)
    feedback = state.get("critic_feedback", "")
    
    # If no scores are recorded yet, bypass safety check and package (failsafe)
    if not scores:
        logger.warning("[Graph Router] No critic scores found. Proceeding to packager.")
        return "packager"
        
    latest_score = scores[-1]
    logger.info(f"[Graph Router] Latest critic score: {latest_score:.2f} | Retry Count: {retry_count}/2")
    
    # Route back to prompts if we have critic feedback (meaning critic qualified the retry and incremented retry_count)
    if feedback and latest_score < 7.0 and retry_count <= 2:
        logger.warning(f"[Graph Router] Quality underperforming and retry qualified. Routing back to 'prompts' for refinement. Feedback: {feedback[:100]}...")
        return "prompts"
    else:
        if latest_score < 7.0:
            logger.error(f"[Graph Router] Max retries (2) reached or quality check failed without retry qualification. Forcing progression to packager despite score {latest_score:.2f}.")
        else:
            logger.info("[Graph Router] Quality check passed. Routing to packager.")
        return "packager"

# Initialize state graph builder
workflow = StateGraph(ProductState)

# Add all agent nodes to the graph
workflow.add_node("research", research_node)
workflow.add_node("strategy", strategy_node)
workflow.add_node("prompts", prompts_node)
workflow.add_node("image_gen", image_gen_node)
workflow.add_node("video_gen", video_gen_node)
workflow.add_node("critic", critic_node)
workflow.add_node("packager", packager_node)

# Set starting entrypoint node
workflow.set_entry_point("research")

# Define sequential execution edges
workflow.add_edge("research", "strategy")
workflow.add_edge("strategy", "prompts")

# Fan-out parallel generation path (Diamond Shape)
workflow.add_edge("prompts", "image_gen")
workflow.add_edge("prompts", "video_gen")

# Fan-in synchronization edge to Critic Node
workflow.add_edge("image_gen", "critic")
workflow.add_edge("video_gen", "critic")

# Add conditional execution loop based on Critic Node evaluation
workflow.add_conditional_edges(
    "critic",
    should_retry,
    {
        "prompts": "prompts",
        "packager": "packager"
    }
)

# Terminate pipeline after packaging completes
workflow.add_edge("packager", END)

# Compile LangGraph State Machine Graph
app_graph = workflow.compile()
logger.info("LangGraph state machine graph successfully compiled.")
