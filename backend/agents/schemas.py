from typing import List, Dict, Any
from pydantic import BaseModel, Field

class ProductSchema(BaseModel):
    """Structured data extracted from the crawled product page."""
    url: str = Field(description="The source URL of the product page.")
    title: str = Field(description="The official title or name of the product.")
    brand: str = Field(default="Unknown", description="The brand name of the product.")
    features: List[str] = Field(default_factory=list, description="Key features, benefits, or bullet points of the product.")
    specs: Dict[str, str] = Field(default_factory=dict, description="Technical specifications table, dimensions, materials, etc.")
    price: str = Field(default="", description="The price of the product including currency symbol.")
    reviews_summary: str = Field(default="", description="A synthesized summary of customer feedback and ratings.")
    brand_tone: str = Field(default="modern, friendly, premium", description="The overall brand voice and tone (e.g. bold, minimalist, luxury).")
    product_images: List[str] = Field(default_factory=list, description="A list of image URLs discovered on the product page.")

class CreativeAngle(BaseModel):
    """A marketing and creative direction angle for the product."""
    angle_type: str = Field(description="Type of angle: lifestyle/aspirational, problem-solution/utility, or minimalist product-focused.")
    hook: str = Field(description="An attention-grabbing performance marketing hook text.")
    target_audience: str = Field(description="A highly specific description of the target audience persona.")
    visual_mood: str = Field(description="Visual mood, lighting theme, and environment direction.")
    color_palette: List[str] = Field(description="List of primary and accent colors to dominate the visuals.")
    caption: str = Field(description="The marketing copy or social caption matching this creative angle.")

class CreativeBrief(BaseModel):
    """The full performance marketing creative brief containing 3 distinct angles."""
    product_title: str = Field(description="Confirm the title of the product for reference.")
    angles: List[CreativeAngle] = Field(description="Exactly 3 distinct creative marketing angles.")

class PromptSet(BaseModel):
    """The generated text prompts for image and video generation tools."""
    image_prompts: List[str] = Field(
        description="Exactly 5 detailed image prompts. Format: scene description, lighting, mood, style, negative prompt. Must contain product reference: 'product: {title}, preserve exact product shape and color'."
    )
    video_prompts: List[str] = Field(
        description="Exactly 2 detailed video prompts. Must contain explicit motion parameters (camera movement, pacing, loop instructions, scene transitions)."
    )

class CriticEvaluation(BaseModel):
    """The VLM review evaluation results of generated assets."""
    scores: List[float] = Field(
        description="List of exactly 4 scores (1-10) for: (1) product visual accuracy, (2) brand consistency, (3) hallucination absence, (4) marketing effectiveness."
    )
    average: float = Field(description="Average score of the 4 evaluation categories.")
    feedback: str = Field(description="Multimodal VLM feedback detailing why any score is low and what specific elements to refine on the next pass.")
