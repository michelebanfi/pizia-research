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
# LLM Provider Configuration
# =============================================================================

# Provider selection: "google" for Google AI Studio, "openrouter" for OpenRouter
# Use "google" for free tier testing, "openrouter" for production
LLMProvider = Literal["google", "openrouter"]
LLM_PROVIDER: LLMProvider = os.getenv("LLM_PROVIDER", "google").lower()  # type: ignore


# =============================================================================
# Model Configuration
# =============================================================================

ModelType = Literal["cheap", "reasoning"]

@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str
    provider: str  # "google" or "openrouter" (litellm provider prefix for openrouter)
    rate_limit: float  # requests per second
    model_type: ModelType
    extra_params: dict = field(default_factory=dict)
    
    @property
    def full_name(self) -> str:
        """Get the full model name for litellm (OpenRouter only)."""
        if self.provider == "google":
            return self.name  # Google uses plain model names
        return f"{self.provider}/{self.name}" if self.provider else self.name


# =============================================================================
# GOOGLE AI STUDIO MODELS (Free tier for testing)
# =============================================================================

GOOGLE_CHEAP_MODELS: list[ModelConfig] = [
    ModelConfig(
        name="models/gemini-2.5-flash",       # Fast, efficient, good at code
        provider="google",
        rate_limit=10.0,  # Google has generous rate limits in free tier
        model_type="cheap",
    ),
]

GOOGLE_REASONING_MODELS: list[ModelConfig] = [
    ModelConfig(
        name="models/gemini-3-flash-preview",         # Strong reasoning capabilities
        provider="google",
        rate_limit=2.0,
        model_type="reasoning",
    ),
]


# =============================================================================
# OPENROUTER MODELS (Production)
# =============================================================================

OPENROUTER_CHEAP_MODELS: list[ModelConfig] = [
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

OPENROUTER_REASONING_MODELS: list[ModelConfig] = [
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


# =============================================================================
# Dynamic Model Selection Based on Provider
# =============================================================================

def get_cheap_models() -> list[ModelConfig]:
    """Get cheap models based on current provider."""
    if LLM_PROVIDER == "google":
        return GOOGLE_CHEAP_MODELS
    return OPENROUTER_CHEAP_MODELS


def get_reasoning_models() -> list[ModelConfig]:
    """Get reasoning models based on current provider."""
    if LLM_PROVIDER == "google":
        return GOOGLE_REASONING_MODELS
    return OPENROUTER_REASONING_MODELS


# For backwards compatibility
CHEAP_MODELS = get_cheap_models()
REASONING_MODELS = get_reasoning_models()


def get_model(model_type: ModelType, index: int = 0) -> ModelConfig:
    """Get a model configuration by type and index."""
    models = get_cheap_models() if model_type == "cheap" else get_reasoning_models()
    return models[index % len(models)]


def get_default_cheap_model() -> ModelConfig:
    """Get the default cheap model (first in list)."""
    return get_cheap_models()[0]


def get_default_reasoning_model() -> ModelConfig:
    """Get the default reasoning model (first in list)."""
    return get_reasoning_models()[0]


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
        "google": "GOOGLE_API_KEY",           # Google AI Studio
        "openrouter": "OPENROUTER_API_KEY",   # OpenRouter
        "parallel": "PARALLEL_API_KEY",       # Deep search
    }
    env_var = key_map.get(provider.lower())
    return os.getenv(env_var) if env_var else None
