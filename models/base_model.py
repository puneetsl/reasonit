"""
Base LLM wrapper with rate limiting, cost tracking, and error handling.

This module provides the abstract base class for all LLM implementations
in the ReasonIt system, ensuring consistent behavior across different providers.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any

from .exceptions import (
    AuthenticationError,
    CostLimitError,
    LLMGenerationError,
    RateLimitError,
    ReasonItException,
)
from .types import SystemConfiguration, UsageMetrics

logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter with exponential backoff for API calls."""

    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_day: int = 10000,
        backoff_factor: float = 2.0,
        max_retries: int = 3
    ):
        self.requests_per_minute = requests_per_minute
        self.requests_per_day = requests_per_day
        self.backoff_factor = backoff_factor
        self.max_retries = max_retries

        # Track requests
        self.minute_requests: list[float] = []
        self.daily_requests: list[float] = []

        # Backoff state
        self.consecutive_failures = 0
        self.last_failure_time = 0.0

    async def acquire(self) -> None:
        """Acquire permission to make a request, with rate limiting."""
        now = time.time()

        # Clean old requests
        self._clean_old_requests(now)

        # Check if we need to wait due to rate limits
        minute_wait = self._calculate_minute_wait(now)
        daily_wait = self._calculate_daily_wait(now)
        backoff_wait = self._calculate_backoff_wait(now)

        wait_time = max(minute_wait, daily_wait, backoff_wait)

        if wait_time > 0:
            logger.info(f"Rate limiting: waiting {wait_time:.2f} seconds")
            await asyncio.sleep(wait_time)
            now = time.time()

        # Record this request
        self.minute_requests.append(now)
        self.daily_requests.append(now)

    def record_success(self) -> None:
        """Record a successful request."""
        self.consecutive_failures = 0

    def record_failure(self) -> None:
        """Record a failed request for backoff calculation."""
        self.consecutive_failures += 1
        self.last_failure_time = time.time()

    def _clean_old_requests(self, now: float) -> None:
        """Remove requests outside the tracking window."""
        # Remove requests older than 1 minute
        minute_ago = now - 60
        self.minute_requests = [req for req in self.minute_requests if req > minute_ago]

        # Remove requests older than 1 day
        day_ago = now - 86400
        self.daily_requests = [req for req in self.daily_requests if req > day_ago]

    def _calculate_minute_wait(self, now: float) -> float:
        """Calculate wait time based on per-minute limits."""
        if len(self.minute_requests) >= self.requests_per_minute:
            oldest_request = min(self.minute_requests)
            return max(0, oldest_request + 60 - now)
        return 0

    def _calculate_daily_wait(self, now: float) -> float:
        """Calculate wait time based on daily limits."""
        if len(self.daily_requests) >= self.requests_per_day:
            oldest_request = min(self.daily_requests)
            return max(0, oldest_request + 86400 - now)
        return 0

    def _calculate_backoff_wait(self, now: float) -> float:
        """Calculate exponential backoff wait time."""
        if self.consecutive_failures == 0:
            return 0

        backoff_time = (self.backoff_factor ** self.consecutive_failures) - 1
        elapsed_since_failure = now - self.last_failure_time
        return max(0, backoff_time - elapsed_since_failure)


class CostTracker:
    """Tracks API usage costs and enforces limits."""

    def __init__(self, max_daily_cost: float = 10.0):
        self.max_daily_cost = max_daily_cost
        self.daily_usage: list[UsageMetrics] = []
        self.session_usage: dict[str, list[UsageMetrics]] = {}

    def add_usage(self, usage: UsageMetrics) -> None:
        """Add a usage record and check limits."""
        # Clean old daily usage
        now = datetime.now()
        day_ago = now - timedelta(days=1)
        self.daily_usage = [u for u in self.daily_usage if u.timestamp > day_ago]

        # Add new usage
        self.daily_usage.append(usage)

        # Track by session
        if usage.session_id not in self.session_usage:
            self.session_usage[usage.session_id] = []
        self.session_usage[usage.session_id].append(usage)

        # Check daily limit
        daily_cost = sum(u.cost for u in self.daily_usage)
        if daily_cost > self.max_daily_cost:
            raise CostLimitError(
                f"Daily cost limit exceeded: ${daily_cost:.4f} > ${self.max_daily_cost:.4f}",
                current_cost=daily_cost,
                limit=self.max_daily_cost
            )

    def get_daily_cost(self) -> float:
        """Get total cost for the current day."""
        now = datetime.now()
        day_ago = now - timedelta(days=1)
        recent_usage = [u for u in self.daily_usage if u.timestamp > day_ago]
        return sum(u.cost for u in recent_usage)

    def get_session_cost(self, session_id: str) -> float:
        """Get total cost for a specific session."""
        session_usage = self.session_usage.get(session_id, [])
        return sum(u.cost for u in session_usage)

    def get_cost_breakdown(self) -> dict[str, Any]:
        """Get detailed cost breakdown."""
        daily_cost = self.get_daily_cost()
        model_costs = {}
        strategy_costs = {}

        for usage in self.daily_usage:
            # By model
            if usage.model_name not in model_costs:
                model_costs[usage.model_name] = 0
            model_costs[usage.model_name] += usage.cost

            # By strategy
            if usage.reasoning_strategy:
                strategy = usage.reasoning_strategy
                if strategy not in strategy_costs:
                    strategy_costs[strategy] = 0
                strategy_costs[strategy] += usage.cost

        return {
            "daily_total": daily_cost,
            "daily_limit": self.max_daily_cost,
            "remaining_budget": max(0, self.max_daily_cost - daily_cost),
            "by_model": model_costs,
            "by_strategy": strategy_costs,
            "total_requests": len(self.daily_usage),
        }


class BaseLLMWrapper(ABC):
    """Abstract base class for all LLM implementations."""

    def __init__(
        self,
        model_name: str,
        config: SystemConfiguration | None = None,
        api_key: str | None = None,
        **kwargs
    ):
        self.model_name = model_name
        self.config = config or SystemConfiguration()
        self.api_key = api_key

        # Initialize rate limiter and cost tracker
        self.rate_limiter = RateLimiter(
            requests_per_minute=kwargs.get('requests_per_minute', 60),
            requests_per_day=kwargs.get('requests_per_day', 10000),
            backoff_factor=kwargs.get('backoff_factor', 2.0),
            max_retries=kwargs.get('max_retries', 3)
        )

        self.cost_tracker = CostTracker(
            max_daily_cost=kwargs.get('max_daily_cost', self.config.max_daily_cost)
        )

        # State tracking
        self._last_usage: UsageMetrics | None = None
        self._session_id: str | None = None

    @abstractmethod
    async def _make_api_call(
        self,
        prompt: str,
        session_id: str,
        **kwargs
    ) -> dict[str, Any]:
        """Make the actual API call to the LLM provider.
        
        Returns:
            Dict containing 'content', 'usage', and any other relevant data
        """
        pass

    @abstractmethod
    def _calculate_cost(self, usage_data: dict[str, Any]) -> float:
        """Calculate cost based on usage data from the API."""
        pass

    @abstractmethod
    def _extract_usage_metrics(
        self,
        usage_data: dict[str, Any],
        session_id: str
    ) -> UsageMetrics:
        """Extract usage metrics from API response."""
        pass

    async def generate(
        self,
        prompt: str,
        session_id: str | None = None,
        max_retries: int | None = None,
        **kwargs
    ) -> str:
        """Generate text using the LLM with rate limiting and cost tracking."""
        session_id = session_id or self._session_id or "default"
        max_retries = max_retries or self.rate_limiter.max_retries

        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                # Rate limiting
                await self.rate_limiter.acquire()

                # Make API call
                start_time = time.time()
                response = await self._make_api_call(prompt, session_id, **kwargs)

                # Extract content and usage
                content = response.get('content', '')
                usage_data = response.get('usage', {})

                # Calculate cost and create metrics
                cost = self._calculate_cost(usage_data)
                usage_metrics = self._extract_usage_metrics(usage_data, session_id)

                # Track cost
                self.cost_tracker.add_usage(usage_metrics)
                self._last_usage = usage_metrics

                # Record success
                self.rate_limiter.record_success()

                elapsed_time = time.time() - start_time
                logger.info(
                    f"LLM generation successful: {usage_metrics.total_tokens} tokens, "
                    f"${cost:.4f}, {elapsed_time:.2f}s"
                )
                
                logger.info(f"Base model returning content length: {len(content)}")
                logger.info(f"Base model content starts: {content[:100]}...")

                return content

            except RateLimitError as e:
                logger.warning(f"Rate limit hit on attempt {attempt + 1}: {e}")
                self.rate_limiter.record_failure()
                last_exception = e

                if attempt < max_retries:
                    wait_time = e.retry_after or (2 ** attempt)
                    await asyncio.sleep(wait_time)
                    continue

            except (CostLimitError, AuthenticationError) as e:
                # Don't retry these errors
                logger.error(f"Non-retryable error: {e}")
                raise

            except Exception as e:
                logger.error(f"LLM generation error on attempt {attempt + 1}: {e}")
                self.rate_limiter.record_failure()
                last_exception = e

                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)
                    continue

        # All retries exhausted
        if isinstance(last_exception, ReasonItException):
            raise last_exception
        else:
            raise LLMGenerationError(
                f"Failed to generate after {max_retries + 1} attempts: {last_exception}",
                details={"original_error": str(last_exception)}
            )

    async def generate_with_backoff(
        self,
        prompt: str,
        session_id: str | None = None,
        **kwargs
    ) -> str:
        """Generate with automatic exponential backoff on failures."""
        return await self.generate(prompt, session_id, **kwargs)

    def get_usage_metrics(self) -> UsageMetrics | None:
        """Get the last usage metrics."""
        return self._last_usage

    def get_cost_summary(self) -> dict[str, Any]:
        """Get cost summary and breakdown."""
        return self.cost_tracker.get_cost_breakdown()

    def set_session_id(self, session_id: str) -> None:
        """Set the session ID for tracking."""
        self._session_id = session_id

    async def health_check(self) -> bool:
        """Check if the LLM service is healthy."""
        try:
            await self.generate("Test", max_retries=1)
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def reset_rate_limiter(self) -> None:
        """Reset the rate limiter state."""
        self.rate_limiter.minute_requests.clear()
        self.rate_limiter.daily_requests.clear()
        self.rate_limiter.consecutive_failures = 0

    def get_rate_limit_status(self) -> dict[str, Any]:
        """Get current rate limit status."""
        now = time.time()
        self.rate_limiter._clean_old_requests(now)

        return {
            "requests_this_minute": len(self.rate_limiter.minute_requests),
            "requests_per_minute_limit": self.rate_limiter.requests_per_minute,
            "requests_today": len(self.rate_limiter.daily_requests),
            "requests_per_day_limit": self.rate_limiter.requests_per_day,
            "consecutive_failures": self.rate_limiter.consecutive_failures,
            "backoff_wait": self.rate_limiter._calculate_backoff_wait(now),
        }
