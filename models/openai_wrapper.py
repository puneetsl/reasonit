"""
OpenAI GPT-4o Mini wrapper with optimized cost tracking and error handling.

This module provides a concrete implementation of the BaseLLMWrapper for OpenAI's
GPT-4o Mini model, with specific cost calculations and error handling.
"""

import logging
import os
from datetime import datetime
from typing import Any

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from .base_model import BaseLLMWrapper
from .exceptions import (
    AuthenticationError,
    LLMGenerationError,
    RateLimitError,
)
from .types import SystemConfiguration, UsageMetrics

logger = logging.getLogger(__name__)


class OpenAIWrapper(BaseLLMWrapper):
    """OpenAI GPT-4o Mini wrapper with cost optimization."""

    # OpenAI pricing (per 1M tokens) - as of 2024
    PRICING = {
        "gpt-4o-mini": {
            "input": 0.15,   # $0.15 per 1M input tokens
            "output": 0.60,  # $0.60 per 1M output tokens
        },
        "gpt-4": {
            "input": 30.0,   # $30.00 per 1M input tokens
            "output": 60.0,  # $60.00 per 1M output tokens
        },
        "gpt-4-turbo": {
            "input": 10.0,   # $10.00 per 1M input tokens
            "output": 30.0,  # $30.00 per 1M output tokens
        }
    }

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: str | None = None,
        config: SystemConfiguration | None = None,
        **kwargs
    ):
        super().__init__(model_name=model_name, config=config, api_key=api_key, **kwargs)

        # Initialize OpenAI client
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise AuthenticationError("OpenAI API key not provided")

        self.client = AsyncOpenAI(api_key=self.api_key)

        # Model configuration
        self.temperature = kwargs.get('temperature', 0.7)
        self.max_tokens = kwargs.get('max_tokens', 2000)
        self.top_p = kwargs.get('top_p', 1.0)
        self.frequency_penalty = kwargs.get('frequency_penalty', 0.0)
        self.presence_penalty = kwargs.get('presence_penalty', 0.0)

        # Validate model name
        if self.model_name not in self.PRICING:
            logger.warning(f"Unknown model {self.model_name}, using default pricing")

    async def _make_api_call(
        self,
        prompt: str,
        session_id: str,
        **kwargs
    ) -> dict[str, Any]:
        """Make API call to OpenAI."""
        try:
            # Prepare messages
            messages = self._prepare_messages(prompt, kwargs.get('messages', []))

            # Extract parameters
            temperature = kwargs.get('temperature', self.temperature)
            max_tokens = kwargs.get('max_tokens', self.max_tokens)
            top_p = kwargs.get('top_p', self.top_p)
            frequency_penalty = kwargs.get('frequency_penalty', self.frequency_penalty)
            presence_penalty = kwargs.get('presence_penalty', self.presence_penalty)
            functions = kwargs.get('functions')
            function_call = kwargs.get('function_call')
            tools = kwargs.get('tools')
            tool_choice = kwargs.get('tool_choice')

            # Make API call
            call_kwargs = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
            }

            # Add function/tool calling if specified
            if functions:
                call_kwargs["functions"] = functions
                if function_call:
                    call_kwargs["function_call"] = function_call

            if tools:
                call_kwargs["tools"] = tools
                if tool_choice:
                    call_kwargs["tool_choice"] = tool_choice

            response: ChatCompletion = await self.client.chat.completions.create(**call_kwargs)

            # Extract content
            content = ""
            if response.choices and response.choices[0].message:
                message = response.choices[0].message
                if message.content:
                    content = message.content

                # Handle function calls
                if hasattr(message, 'function_call') and message.function_call:
                    content += f"\n[Function Call: {message.function_call}]"

                # Handle tool calls
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    for tool_call in message.tool_calls:
                        content += f"\n[Tool Call: {tool_call}]"

            return {
                "content": content,
                "usage": response.usage.model_dump() if response.usage else {},
                "response": response,
                "finish_reason": response.choices[0].finish_reason if response.choices else None,
            }

        except openai.RateLimitError as e:
            raise RateLimitError(
                f"OpenAI rate limit exceeded: {e}",
                retry_after=getattr(e, 'retry_after', None)
            )
        except openai.AuthenticationError as e:
            raise AuthenticationError(f"OpenAI authentication failed: {e}")
        except openai.APIError as e:
            raise LLMGenerationError(f"OpenAI API error: {e}")
        except Exception as e:
            raise LLMGenerationError(f"Unexpected OpenAI error: {e}")

    def _prepare_messages(self, prompt: str, existing_messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Prepare messages for OpenAI API call."""
        messages = []

        # Add existing messages (for conversation context)
        for msg in existing_messages:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                messages.append(msg)

        # Add current prompt as user message
        messages.append({
            "role": "user",
            "content": prompt
        })

        return messages

    def _calculate_cost(self, usage_data: dict[str, Any]) -> float:
        """Calculate cost based on OpenAI token usage."""
        if not usage_data:
            return 0.0

        input_tokens = usage_data.get('prompt_tokens', 0)
        output_tokens = usage_data.get('completion_tokens', 0)

        # Get pricing for this model
        model_pricing = self.PRICING.get(self.model_name, self.PRICING['gpt-4o-mini'])

        # Calculate cost (pricing is per 1M tokens)
        input_cost = (input_tokens / 1_000_000) * model_pricing['input']
        output_cost = (output_tokens / 1_000_000) * model_pricing['output']

        total_cost = input_cost + output_cost

        logger.debug(
            f"Cost calculation: {input_tokens} input + {output_tokens} output tokens "
            f"= ${total_cost:.6f} ({self.model_name})"
        )

        return total_cost

    def _extract_usage_metrics(
        self,
        usage_data: dict[str, Any],
        session_id: str
    ) -> UsageMetrics:
        """Extract usage metrics from OpenAI response."""
        input_tokens = usage_data.get('prompt_tokens', 0)
        output_tokens = usage_data.get('completion_tokens', 0)
        total_tokens = usage_data.get('total_tokens', input_tokens + output_tokens)

        cost = self._calculate_cost(usage_data)

        return UsageMetrics(
            session_id=session_id,
            model_name=self.model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost=cost,
            timestamp=datetime.now(),
            tool_calls=0,  # Will be updated by caller if tools were used
            reasoning_strategy=None,  # Will be set by reasoning agent
        )

    async def generate_with_functions(
        self,
        prompt: str,
        functions: list[dict[str, Any]],
        function_call: str | dict[str, str] | None = None,
        session_id: str | None = None,
        **kwargs
    ) -> dict[str, Any]:
        """Generate with OpenAI function calling support."""
        try:
            kwargs.update({
                'functions': functions,
                'function_call': function_call
            })

            # Use base generate method but return full response
            session_id = session_id or self._session_id or "default"

            # Rate limiting
            await self.rate_limiter.acquire()

            # Make API call
            response = await self._make_api_call(prompt, session_id, **kwargs)

            # Track usage
            usage_data = response.get('usage', {})
            cost = self._calculate_cost(usage_data)
            usage_metrics = self._extract_usage_metrics(usage_data, session_id)

            self.cost_tracker.add_usage(usage_metrics)
            self._last_usage = usage_metrics
            self.rate_limiter.record_success()

            return response

        except Exception:
            self.rate_limiter.record_failure()
            raise

    async def generate_with_tools(
        self,
        prompt: str,
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any] | None = None,
        session_id: str | None = None,
        **kwargs
    ) -> dict[str, Any]:
        """Generate with OpenAI tools (newer function calling API)."""
        try:
            kwargs.update({
                'tools': tools,
                'tool_choice': tool_choice
            })

            return await self.generate_with_functions(
                prompt, [], None, session_id, **kwargs
            )

        except Exception as e:
            logger.error(f"Tool generation failed: {e}")
            raise

    async def estimate_cost(self, prompt: str, **kwargs) -> float:
        """Estimate cost for a prompt without making the API call."""
        # Rough token estimation (OpenAI uses tiktoken, but this is approximate)
        estimated_input_tokens = len(prompt.split()) * 1.3  # Rough approximation
        estimated_output_tokens = kwargs.get('max_tokens', self.max_tokens)

        model_pricing = self.PRICING.get(self.model_name, self.PRICING['gpt-4o-mini'])

        input_cost = (estimated_input_tokens / 1_000_000) * model_pricing['input']
        output_cost = (estimated_output_tokens / 1_000_000) * model_pricing['output']

        return input_cost + output_cost

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the current model."""
        pricing = self.PRICING.get(self.model_name, self.PRICING['gpt-4o-mini'])

        return {
            "model_name": self.model_name,
            "provider": "openai",
            "pricing": pricing,
            "context_length": 128000 if "gpt-4o" in self.model_name else 8192,
            "supports_functions": True,
            "supports_tools": True,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

    async def health_check(self) -> bool:
        """Check OpenAI API health."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=1,
                temperature=0
            )
            return bool(response.choices)
        except Exception as e:
            logger.error(f"OpenAI health check failed: {e}")
            return False

    def set_model(self, model_name: str) -> None:
        """Switch to a different OpenAI model."""
        if model_name not in self.PRICING:
            logger.warning(f"Unknown model {model_name}, pricing may be inaccurate")

        self.model_name = model_name
        logger.info(f"Switched to model: {model_name}")

    async def close(self) -> None:
        """Clean up resources."""
        if hasattr(self.client, 'close'):
            await self.client.close()


# Factory function for easy instantiation
def create_openai_wrapper(
    model_name: str = "gpt-4o-mini",
    api_key: str | None = None,
    config: SystemConfiguration | None = None,
    **kwargs
) -> OpenAIWrapper:
    """Create an OpenAI wrapper instance."""
    return OpenAIWrapper(
        model_name=model_name,
        api_key=api_key,
        config=config,
        **kwargs
    )


# Common model configurations
OPENAI_MODELS = {
    "gpt-4o-mini": {
        "name": "gpt-4o-mini",
        "context_length": 128000,
        "best_for": ["reasoning", "cost-efficiency", "general tasks"],
        "temperature": 0.7,
    },
    "gpt-4": {
        "name": "gpt-4",
        "context_length": 8192,
        "best_for": ["complex reasoning", "high accuracy"],
        "temperature": 0.7,
    },
    "gpt-4-turbo": {
        "name": "gpt-4-turbo",
        "context_length": 128000,
        "best_for": ["complex tasks", "function calling"],
        "temperature": 0.7,
    },
}
