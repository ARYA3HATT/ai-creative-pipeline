import os
import time
import random
import logging
from typing import Type, TypeVar, Optional, Any
from pydantic import BaseModel
import instructor
from groq import Groq
import openai

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llm_wrapper")

T = TypeVar('T', bound=BaseModel)

class LLMClientManager:
    """Manages LLM clients with robust rate limiting and multi-provider fallbacks."""
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.together_api_key = os.getenv("TOGETHER_API_KEY")
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        
        # Initialize Groq client with Instructor
        if self.groq_api_key:
            self.groq_client = instructor.from_groq(Groq(api_key=self.groq_api_key))
            logger.info("Groq client initialized with Instructor.")
        else:
            self.groq_client = None
            logger.warning("GROQ_API_KEY is not set in environment.")

        # Initialize Together client with Instructor (Together offers OpenAI-compatible API)
        if self.together_api_key:
            self.together_client = instructor.from_openai(
                openai.OpenAI(
                    base_url="https://api.together.xyz/v1",
                    api_key=self.together_api_key
                )
            )
            logger.info("Together AI client initialized with Instructor.")
        else:
            self.together_client = None

        # Initialize OpenRouter client with Instructor
        if self.openrouter_api_key:
            self.openrouter_client = instructor.from_openai(
                openai.OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=self.openrouter_api_key,
                    default_headers={"HTTP-Referer": "https://localhost", "X-Title": "AI Creative Pipeline"}
                )
            )
            logger.info("OpenRouter client initialized with Instructor.")
        else:
            self.openrouter_client = None

    def _generate_mock_response(self, response_model: Type[T]) -> T:
        """Generates a high-quality simulated mock response matching the requested schema."""
        name = response_model.__name__
        if name == "ProductSchema":
            return response_model(
                url="https://example.com/product-a",
                title="AudioAura Premium Headphones",
                brand="AudioAura",
                features=[
                    "Minimalist silver finish with matte anodized steel band",
                    "Volumetric spatial soundscape isolation",
                    "Cozy memory foam earcups",
                    "40-hour battery life with USB-C quick charge"
                ],
                specs={
                    "Weight": "280 grams",
                    "Material": "Aluminum and leather"
                },
                price="$299.00",
                reviews_summary="Highly praised for beautiful silver styling, superb volumetric noise cancellation, and premium leather feel.",
                brand_tone="modern, luxury, minimalist",
                product_images=["https://images.unsplash.com/photo-1523275335684-37898b6baf30?q=80&w=800"]
            )
        elif name == "CreativeBrief":
            # Direct import inside helper to prevent circular dependency
            from backend.agents.schemas import CreativeAngle
            return response_model(
                product_title="AudioAura Premium Headphones",
                angles=[
                    CreativeAngle(
                        angle_type="lifestyle/aspirational",
                        hook="Elevate your silence. Premium sound meets high-fashion styling.",
                        target_audience="Urban professionals and design-conscious audiophiles",
                        visual_mood="cozy warm diffuse studio lighting, luxurious copper and beige tones, elegant model pose",
                        color_palette=["beige", "copper", "soft silver"],
                        caption="A masterpiece for your ears. Immerse in pure soundscape bliss."
                    ),
                    CreativeAngle(
                        angle_type="problem-solution/utility",
                        hook="Block out the noise. Immerse in the music.",
                        target_audience="Commuters, travelers, and open-office remote workers",
                        visual_mood="high-contrast moody city evening light, soft volumetric bokeh background",
                        color_palette=["dark charcoal", "slate gray", "minimal silver"],
                        caption="Tune out the world. AudioAura advanced spatial isolation keeps you focused."
                    ),
                    CreativeAngle(
                        angle_type="minimalist product-focused",
                        hook="Pure metal. Pure sound. Pure simplicity.",
                        target_audience="Minimalists and tech purists who appreciate luxury materials",
                        visual_mood="high-key bright white studio, soft diffuse shadows, sharp focus on silver aluminum texture",
                        color_palette=["bright silver", "matte white", "platinum"],
                        caption="Crafted from premium anodized steel. Designed to look as stunning as it sounds."
                    )
                ]
            )
        elif name == "PromptSet":
            return response_model(
                image_prompts=[
                    "product: AudioAura Premium Headphones, preserve exact product shape and color, resting on a polished travertine podium, cozy warm diffuse studio lighting, 85mm lens, sharp focus, negative prompt: distorted, blurry, plastic, cheap",
                    "product: AudioAura Premium Headphones, preserve exact product shape and color, worn by an elegant model in a modern warm beige minimalist living room, soft volumetric window light, 35mm lens, cinematic photography, negative prompt: low quality, distorted",
                    "product: AudioAura Premium Headphones, preserve exact product shape and color, in a city commuter scene, soft moody evening neon lighting in background, high-contrast bokeh, 50mm, photorealistic, negative prompt: low res, noisy",
                    "product: AudioAura Premium Headphones, preserve exact product shape and color, macro close-up showing the silver aluminum texture and leather stitching, high-key bright white studio lighting, sharp focus, negative prompt: text, watermark",
                    "product: AudioAura Premium Headphones, preserve exact product shape and color, floating gracefully amidst gentle volumetric mist and soft cloud elements, luxurious platinum atmosphere, commercial product photography, negative prompt: cheap, noise"
                ],
                video_prompts=[
                    "Slow orbital panning shot around the product: AudioAura Premium Headphones, resting on premium marble in studio, 360 degree smooth movement, soft volumetric lighting, cinematic 8k, duration: 6 seconds",
                    "Gentle dolly-in macro shot showcasing the minimalist silver finish of AudioAura Premium Headphones, showing light glinting off the metal texture, sharp depth of field, duration: 8 seconds"
                ]
            )
        elif name == "CriticEvaluation":
            return response_model(
                scores=[8.0, 8.5, 8.0, 8.5],
                average=8.25,
                feedback="Visual evaluation passed successfully. Composition has excellent product details."
            )
        return response_model()

    def get_structured_completion(
        self,
        prompt: str,
        response_model: Type[T],
        system_instruction: str = "You are a professional performance marketing assistant.",
        prefer_high_capacity: bool = True,
        max_retries: int = 4,
        initial_backoff: float = 2.0
    ) -> T:
        """
        Executes a LLM structured call, forcing response into the response_model.
        Includes automatic exponential backoff on 429s and automatic provider fallbacks.
        Falls back to highly robust offline simulated Pydantic stubs if no API keys are set.
        """
        # Robust Offline Simulator Fallback: If no API keys are configured, return mock Pydantic object
        if not self.groq_api_key and not self.together_api_key and not self.openrouter_api_key:
            logger.warning(f"[LLM Wrapper] No active LLM API keys configured. Generating resilient simulated {response_model.__name__} response.")
            return self._generate_mock_response(response_model)
        # Determine preferred model sizes
        # Groq models: llama-3.3-70b-specdec or llama-3.3-70b-versatile, llama-3.1-8b-instant
        if prefer_high_capacity:
            groq_model = "llama-3.3-70b-versatile"
            together_model = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
            openrouter_model = "meta-llama/llama-3.3-70b-instruct"
        else:
            groq_model = "llama-3.1-8b-instant"
            together_model = "meta-llama/Llama-3.2-3B-Instruct"
            openrouter_model = "meta-llama/llama-3.2-3b-instruct"

        backoff = initial_backoff
        last_exception = None

        # 1. Try Together AI (Primary - highly generous free tier for vision + structured completions)
        for attempt in range(max_retries):
            try:
                if self.together_client:
                    logger.info(f"Attempting Together AI primary request (attempt {attempt + 1}/{max_retries}) using {together_model}")
                    return self.together_client.chat.completions.create(
                        model=together_model,
                        response_model=response_model,
                        messages=[
                            {"role": "system", "content": system_instruction},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                    )
                else:
                    raise ValueError("Together AI client not available.")
            except Exception as e:
                err_str = str(e).lower()
                last_exception = e
                # Check for rate limit (429) or other API errors
                if "429" in err_str or "rate limit" in err_str or "too many requests" in err_str:
                    sleep_time = backoff + random.uniform(0.1, 0.5)
                    logger.warning(f"Together AI Rate Limited (429). Retrying in {sleep_time:.2f}s... Error: {e}")
                    time.sleep(sleep_time)
                    backoff *= 2  # Exponential backoff
                else:
                    logger.error(f"Together AI API error encountered: {e}. Moving to fallbacks immediately.")
                    break  # Break out of loop to trigger fallback providers immediately

        # 2. Try Groq (First Fallback)
        if self.groq_client:
            try:
                logger.info(f"Attempting Groq fallback using {groq_model}")
                return self.groq_client.chat.completions.create(
                    model=groq_model,
                    response_model=response_model,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                )
            except Exception as e:
                logger.error(f"Groq fallback failed: {e}")
                last_exception = e

        # 3. Try OpenRouter (Second Fallback)
        if self.openrouter_client:
            try:
                logger.info(f"Attempting OpenRouter fallback using {openrouter_model}")
                return self.openrouter_client.chat.completions.create(
                    model=openrouter_model,
                    response_model=response_model,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                )
            except Exception as e:
                logger.error(f"OpenRouter fallback failed: {e}")
                last_exception = e

        raise RuntimeError(f"All LLM providers (Together AI, Groq, OpenRouter) failed. Last error: {last_exception}")

# Instantiate global manager
llm_manager = LLMClientManager()
