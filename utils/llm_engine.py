"""
Rate-Limited Async LLM Engine.

Provides async LLM calls with per-model rate limiting using asynciolimiter.
Supports both Google AI Studio (google-genai) and OpenRouter (litellm).
"""

import asyncio
import os
from dataclasses import dataclass
from typing import Any

import litellm
from asynciolimiter import Limiter
from litellm import acompletion
from litellm import exceptions as litellm_exceptions

from config import (
    ModelConfig,
    LLM_PROVIDER,
    get_default_cheap_model,
    get_default_reasoning_model,
    get_api_key,
)
from utils.logger import get_logger

# Silence unnecessary litellm debug logs
litellm.suppress_debug_info = True

# Retry configuration
RETRIES = 3
RETRY_DELAY_SEC = 5


@dataclass
class LLMResponse:
    """Structured response from LLM call."""
    content: str
    duration: float
    prompt_tokens: int
    completion_tokens: int
    model: str
    success: bool = True
    error: str | None = None


class LLMEngine:
    """
    Rate-limited async LLM caller.
    
    Uses asynciolimiter to prevent 429 errors when making parallel requests.
    Supports Google AI Studio (google-genai) and OpenRouter (litellm).
    """
    
    def __init__(self):
        # Per-model rate limiters (rate = requests per second)
        self._limiters: dict[str, Limiter] = {}
        self._google_client = None
    
    def _get_limiter(self, model: ModelConfig) -> Limiter:
        """Get or create a rate limiter for the given model."""
        model_key = model.full_name
        if model_key not in self._limiters:
            self._limiters[model_key] = Limiter(model.rate_limit)
        return self._limiters[model_key]
    
    def _get_google_client(self):
        """Lazy-load the Google genai client."""
        if self._google_client is None:
            from google import genai
            api_key = get_api_key("google")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set")
            self._google_client = genai.Client(api_key=api_key)
        return self._google_client
    
    async def _call_google(
        self,
        model: ModelConfig,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        timeout: float = 300.0,
    ) -> LLMResponse:
        """Make an async call to Google AI Studio."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            client = self._get_google_client()
            
            # Convert messages to Google format
            # Google uses a simpler format: system instruction + contents
            system_instruction = None
            contents = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_instruction = msg["content"]
                elif msg["role"] == "user":
                    contents.append(msg["content"])
                elif msg["role"] == "assistant":
                    # For multi-turn, we'd need to handle this differently
                    # For now, just append to contents
                    contents.append(f"Assistant: {msg['content']}")
            
            # Combine all user messages into one content string
            combined_content = "\n\n".join(contents)
            
            # Build generation config
            from google.genai import types
            
            generation_config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            
            if system_instruction:
                generation_config.system_instruction = system_instruction
            
            # Run the sync call in a thread pool to make it async
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=model.name,
                    contents=combined_content,
                    config=generation_config,
                )
            )
            
            end_time = asyncio.get_event_loop().time()
            duration = end_time - start_time
            
            # Extract text from response
            content = response.text if hasattr(response, 'text') else str(response)
            
            # Try to get usage metadata if available
            prompt_tokens = 0
            completion_tokens = 0
            if hasattr(response, 'usage_metadata'):
                usage = response.usage_metadata
                prompt_tokens = getattr(usage, 'prompt_token_count', 0)
                completion_tokens = getattr(usage, 'candidates_token_count', 0)
            
            return LLMResponse(
                content=content.strip(),
                duration=duration,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                model=model.full_name,
                success=True,
            )
            
        except Exception as e:
            end_time = asyncio.get_event_loop().time()
            duration = end_time - start_time
            return LLMResponse(
                content="",
                duration=duration,
                prompt_tokens=0,
                completion_tokens=0,
                model=model.full_name,
                success=False,
                error=str(e),
            )
    
    async def _call_openrouter(
        self,
        model: ModelConfig,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        timeout: float = 300.0,
    ) -> LLMResponse:
        """Make an async call via litellm (OpenRouter)."""
        limiter = self._get_limiter(model)
        attempt = 1
        
        while attempt <= RETRIES:
            # Wait for rate limit
            await limiter.wait()
            
            start_time = asyncio.get_event_loop().time()
            
            try:
                # Build request params
                params: dict[str, Any] = {
                    "model": model.full_name,
                    "messages": messages,
                    "temperature": temperature,
                    "timeout": timeout,
                    "num_retries": 0,  # We handle retries ourselves
                    **model.extra_params,
                }
                if max_tokens:
                    params["max_tokens"] = max_tokens
                
                # Make the async call
                resp: Any = await acompletion(**params)
                
                end_time = asyncio.get_event_loop().time()
                duration = end_time - start_time
                
                # Extract usage info
                usage = resp.model_extra.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                
                return LLMResponse(
                    content=resp["choices"][0]["message"]["content"].strip(),
                    duration=duration,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    model=model.full_name,
                    success=True,
                )
                
            except (
                litellm_exceptions.RateLimitError,
                litellm_exceptions.InternalServerError,
                litellm_exceptions.ServiceUnavailableError,
                litellm_exceptions.APIConnectionError,
                litellm_exceptions.APIError,
            ) as e:
                # Retryable errors - don't count against retry limit
                print(f"Ignoring {type(e).__name__}, retrying attempt {attempt}: {e}")
                await asyncio.sleep(RETRY_DELAY_SEC)
                continue
                
            except asyncio.TimeoutError:
                end_time = asyncio.get_event_loop().time()
                duration = end_time - start_time
                
                if attempt == RETRIES:
                    return LLMResponse(
                        content="",
                        duration=duration,
                        prompt_tokens=0,
                        completion_tokens=0,
                        model=model.full_name,
                        success=False,
                        error="Timeout",
                    )
                print(f"Timeout on attempt {attempt}, retrying...")
                attempt += 1
                await asyncio.sleep(RETRY_DELAY_SEC)
                
            except Exception as e:
                end_time = asyncio.get_event_loop().time()
                duration = end_time - start_time
                
                if attempt == RETRIES:
                    print(f"Max retry limit reached. Last exception: {e}")
                    return LLMResponse(
                        content="",
                        duration=duration,
                        prompt_tokens=0,
                        completion_tokens=0,
                        model=model.full_name,
                        success=False,
                        error=str(e),
                    )
                    
                print(f"Exception during request: {e}. Retry {attempt}.")
                await asyncio.sleep(RETRY_DELAY_SEC)
                attempt += 1
        
        # Should not reach here, but safety return
        return LLMResponse(
            content="",
            duration=0,
            prompt_tokens=0,
            completion_tokens=0,
            model=model.full_name,
            success=False,
            error="Retries exceeded",
        )
    
    async def call(
        self,
        model: ModelConfig,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        timeout: float = 300.0,
    ) -> LLMResponse:
        """
        Make a single async LLM call with rate limiting and retries.
        
        Routes to Google AI Studio or OpenRouter based on model provider.
        
        Args:
            model: Model configuration
            messages: List of message dicts (role, content)
            temperature: Sampling temperature
            max_tokens: Optional max tokens for response
            timeout: Request timeout in seconds
            
        Returns:
            LLMResponse with content and metadata
        """
        # Wait for rate limit (for Google too)
        limiter = self._get_limiter(model)
        await limiter.wait()
        
        if model.provider == "google":
            response = await self._call_google(model, messages, temperature, max_tokens, timeout)
        else:
            response = await self._call_openrouter(model, messages, temperature, max_tokens, timeout)
        
        # Log the call if logger is active
        logger = get_logger()
        if logger:
            prompt_preview = messages[-1]["content"] if messages else ""
            logger.log_llm_call(
                model=model.full_name,
                prompt_preview=prompt_preview,
                response_preview=response.content,
                duration=response.duration,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                success=response.success,
                error=response.error,
            )
        
        return response
    
    async def parallel_generate(
        self,
        model: ModelConfig,
        prompts: list[str],
        system_prompt: str | None = None,
        temperature: float = 0.8,
        max_tokens: int | None = None,
    ) -> list[LLMResponse]:
        """
        Generate multiple responses in parallel with rate limiting.
        
        This is the core method for the AlphaEvolve-style parallel generation.
        The rate limiter ensures we don't exceed the per-model request limit.
        
        Args:
            model: Model configuration to use
            prompts: List of user prompts to send
            system_prompt: Optional system prompt (same for all)
            temperature: Sampling temperature (higher for diversity)
            max_tokens: Optional max tokens per response
            
        Returns:
            List of LLMResponses in the same order as prompts
        """
        async def make_call(prompt: str) -> LLMResponse:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            return await self.call(model, messages, temperature, max_tokens)
        
        # Launch all calls concurrently - rate limiter handles throttling
        tasks = [make_call(prompt) for prompt in prompts]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error responses
        results: list[LLMResponse] = []
        for i, resp in enumerate(responses):
            if isinstance(resp, Exception):
                results.append(LLMResponse(
                    content="",
                    duration=0,
                    prompt_tokens=0,
                    completion_tokens=0,
                    model=model.full_name,
                    success=False,
                    error=str(resp),
                ))
            else:
                results.append(resp)
        
        return results
    
    async def generate_with_cheap_model(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Convenience method to generate with the default cheap model."""
        model = get_default_cheap_model()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return await self.call(model, messages, temperature)
    
    async def generate_with_reasoning_model(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """Convenience method to generate with the default reasoning model."""
        model = get_default_reasoning_model()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return await self.call(model, messages, temperature)


# Global engine instance
_engine: LLMEngine | None = None


def get_engine() -> LLMEngine:
    """Get the global LLM engine instance."""
    global _engine
    if _engine is None:
        _engine = LLMEngine()
    return _engine


# =============================================================================
# Example usage
# =============================================================================

async def _demo():
    """Demo the LLM engine functionality."""
    from config import get_cheap_models, LLM_PROVIDER
    
    engine = get_engine()
    models = get_cheap_models()
    model = models[0]
    
    print(f"Using provider: {LLM_PROVIDER}")
    print(f"Testing with model: {model.full_name}")
    
    # Simple single call test
    print("\n--- Single Call Test ---")
    response = await engine.generate_with_cheap_model(
        prompt="Write a Python function to reverse a string. Return only the code.",
        system_prompt="You are a helpful coding assistant.",
    )
    
    print(f"Success: {response.success}")
    print(f"Duration: {response.duration:.2f}s")
    print(f"Tokens: {response.prompt_tokens} prompt, {response.completion_tokens} completion")
    if response.success:
        print(f"Content:\n{response.content[:500]}...")
    else:
        print(f"Error: {response.error}")


if __name__ == "__main__":
    asyncio.run(_demo())
