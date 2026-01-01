"""
Configuration for AI Research Agent.
Central place to manage models, rate limits, and API settings.
"""

import os
from dataclasses import dataclass, field
from typing import Literal

from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# Model Configuration
# =============================================================================

ModelType = Literal["cheap", "reasoning"]

@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str
    provider: str  # litellm provider prefix
    rate_limit: float  # requests per second
    model_type: ModelType
    extra_params: dict = field(default_factory=dict)
    
    @property
    def full_name(self) -> str:
        """Get the full model name for litellm."""
        return f"{self.provider}/{self.name}" if self.provider else self.name


# =============================================================================
# FAST MODELS - Used for parallel code generation (cheap, high throughput)
# These models prioritize speed and cost-efficiency for generating many candidates
# =============================================================================

CHEAP_MODELS: list[ModelConfig] = [
    ModelConfig(
        name="deepseek/deepseek-chat",        # Fast, cheap, good at code
        provider="openrouter",
        rate_limit=5.0,
        model_type="cheap",
    ),
    ModelConfig(
        name="meta-llama/llama-3.1-70b-instruct",  # Good balance of speed/quality
        provider="openrouter",
        rate_limit=5.0,
        model_type="cheap",
    ),
    ModelConfig(
        name="qwen/qwen-2.5-72b-instruct",    # Fast, strong at coding
        provider="openrouter",
        rate_limit=5.0,
        model_type="cheap",
    ),
]

# =============================================================================
# REASONING MODELS - Used for test generation, context synthesis, critique
# These models prioritize accuracy and deep thinking over speed
# =============================================================================

REASONING_MODELS: list[ModelConfig] = [
    ModelConfig(
        name="deepseek/deepseek-r1",          # Strong reasoning, good at complex problems
        provider="openrouter",
        rate_limit=1.0,
        model_type="reasoning",
    ),
    ModelConfig(
        name="anthropic/claude-3.5-sonnet",   # Excellent at analysis and test generation
        provider="openrouter",
        rate_limit=2.0,
        model_type="reasoning",
    ),
    ModelConfig(
        name="openai/gpt-4o",                 # Strong all-around reasoning
        provider="openrouter",
        rate_limit=2.0,
        model_type="reasoning",
    ),
]


def get_model(model_type: ModelType, index: int = 0) -> ModelConfig:
    """Get a model configuration by type and index."""
    models = CHEAP_MODELS if model_type == "cheap" else REASONING_MODELS
    return models[index % len(models)]


def get_default_cheap_model() -> ModelConfig:
    """Get the default cheap model (first in list)."""
    return CHEAP_MODELS[0]


def get_default_reasoning_model() -> ModelConfig:
    """Get the default reasoning model (first in list)."""
    return REASONING_MODELS[0]


# =============================================================================
# Sandbox Configuration
# =============================================================================

SANDBOX_TIMEOUT_SECONDS: float = 10.0
SANDBOX_MAX_OUTPUT_CHARS: int = 50000


# =============================================================================
# Evolution Configuration
# =============================================================================

EVOLUTION_PARALLEL_CANDIDATES: int = 5  # Number of parallel code proposals
EVOLUTION_MAX_GENERATIONS: int = 20
EVOLUTION_EARLY_STOP_SCORE: float = 1.0  # Stop if this score is reached


# =============================================================================
# API Keys (from environment)
# =============================================================================

def get_api_key(provider: str) -> str | None:
    """Get API key for a provider from environment."""
    key_map = {
        "openrouter": "OPENROUTER_API_KEY",  # Main LLM provider
        "parallel": "PARALLEL_API_KEY",       # Deep search
    }
    env_var = key_map.get(provider.lower())
    return os.getenv(env_var) if env_var else None
