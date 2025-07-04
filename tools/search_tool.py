"""
Web search tool for information retrieval and fact verification.

This module provides web search capabilities using DuckDuckGo for the ReasonIt
reasoning system, with result processing and summarization.
"""

import asyncio
import logging
import re
from datetime import datetime, timedelta
from typing import Any

from duckduckgo_search import DDGS

from models import SearchError, ToolType

from .base_tool import BaseTool, ToolConfig, ToolMetadata, tool

logger = logging.getLogger(__name__)


class SearchResult:
    """Represents a single search result."""

    def __init__(
        self,
        title: str,
        url: str,
        snippet: str,
        source: str = "web",
        relevance_score: float = 0.0,
        timestamp: datetime | None = None
    ):
        self.title = title
        self.url = url
        self.snippet = snippet
        self.source = source
        self.relevance_score = relevance_score
        self.timestamp = timestamp or datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
            "relevance_score": self.relevance_score,
            "timestamp": self.timestamp.isoformat()
        }

    def __repr__(self) -> str:
        return f"SearchResult(title='{self.title[:50]}...', url='{self.url}')"


class SearchProcessor:
    """Processes and filters search results."""

    def __init__(self):
        # Common words to filter out for relevance scoring
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can'
        }

    def calculate_relevance(self, query: str, result: SearchResult) -> float:
        """Calculate relevance score between query and result."""
        query_words = set(self._extract_keywords(query))
        title_words = set(self._extract_keywords(result.title))
        snippet_words = set(self._extract_keywords(result.snippet))

        # Calculate overlap scores
        title_overlap = len(query_words.intersection(title_words)) / max(len(query_words), 1)
        snippet_overlap = len(query_words.intersection(snippet_words)) / max(len(query_words), 1)

        # Weight title matches higher than snippet matches
        relevance = (title_overlap * 0.7) + (snippet_overlap * 0.3)

        # Boost score for exact phrase matches
        if query.lower() in result.title.lower():
            relevance += 0.2
        if query.lower() in result.snippet.lower():
            relevance += 0.1

        return min(relevance, 1.0)

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract keywords from text, removing stop words."""
        # Simple tokenization and cleaning
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
        return [word for word in words if word not in self.stop_words]

    def filter_and_rank(
        self,
        query: str,
        results: list[SearchResult],
        min_relevance: float = 0.1,
        max_results: int = 10
    ) -> list[SearchResult]:
        """Filter and rank search results by relevance."""

        # Calculate relevance scores
        for result in results:
            result.relevance_score = self.calculate_relevance(query, result)

        # Filter by minimum relevance
        filtered_results = [
            result for result in results
            if result.relevance_score >= min_relevance
        ]

        # Sort by relevance (descending)
        filtered_results.sort(key=lambda r: r.relevance_score, reverse=True)

        # Limit number of results
        return filtered_results[:max_results]

    def summarize_results(self, results: list[SearchResult]) -> str:
        """Create a summary of search results."""
        if not results:
            return "No relevant results found."

        summary_parts = []
        summary_parts.append(f"Found {len(results)} relevant results:")
        summary_parts.append("")

        for i, result in enumerate(results[:5], 1):  # Limit to top 5 for summary
            summary_parts.append(f"{i}. {result.title}")
            summary_parts.append(f"   {result.snippet[:200]}...")
            summary_parts.append(f"   Source: {result.url}")
            summary_parts.append(f"   Relevance: {result.relevance_score:.2f}")
            summary_parts.append("")

        return "\n".join(summary_parts)


class WebSearchTool(BaseTool):
    """Web search tool using DuckDuckGo."""

    def __init__(self, config: ToolConfig | None = None):
        config = config or ToolConfig(
            timeout=15.0,
            max_retries=2,
            cost_per_use=0.002,  # Small cost for API usage
            rate_limit=1.0  # 1 request per second to be respectful
        )
        super().__init__("web_search", ToolType.SEARCH, config)

        self.processor = SearchProcessor()
        self.cache: dict[str, list[SearchResult]] = {}
        self.cache_expiry: dict[str, datetime] = {}
        self.cache_duration = timedelta(hours=1)  # Cache for 1 hour

    async def _execute(
        self,
        query: str,
        max_results: int = 10,
        region: str = "us-en",
        time_range: str | None = None,
        **kwargs
    ) -> dict[str, Any]:
        """Execute web search."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string")

        query = query.strip()
        max_results = min(max_results, 20)  # Limit to prevent abuse

        # Check cache first
        cache_key = f"{query}:{max_results}:{region}:{time_range}"
        if self._is_cached(cache_key):
            logger.info(f"Using cached results for query: {query}")
            results = self.cache[cache_key]
        else:
            # Perform search
            results = await self._perform_search(query, max_results, region, time_range)

            # Cache results
            self.cache[cache_key] = results
            self.cache_expiry[cache_key] = datetime.now() + self.cache_duration

        # Process and rank results
        processed_results = self.processor.filter_and_rank(query, results, max_results=max_results)

        # Create summary
        summary = self.processor.summarize_results(processed_results)

        return {
            "query": query,
            "results": [result.to_dict() for result in processed_results],
            "summary": summary,
            "total_found": len(results),
            "total_returned": len(processed_results),
            "search_engine": "DuckDuckGo",
            "timestamp": datetime.now().isoformat()
        }

    async def _perform_search(
        self,
        query: str,
        max_results: int,
        region: str,
        time_range: str | None
    ) -> list[SearchResult]:
        """Perform the actual search using DuckDuckGo."""

        try:
            logger.info(f"Searching for: {query}")

            # Initialize DuckDuckGo search
            ddgs = DDGS()

            # Perform search
            search_results = []

            # Use asyncio to run the synchronous DDGS search
            loop = asyncio.get_event_loop()

            def search_sync():
                results = ddgs.text(
                    query,
                    region=region,
                    safesearch='moderate',
                    timelimit=time_range,
                    max_results=max_results
                )
                return list(results)

            raw_results = await loop.run_in_executor(None, search_sync)

            # Convert to SearchResult objects
            for result in raw_results:
                search_result = SearchResult(
                    title=result.get('title', ''),
                    url=result.get('href', ''),
                    snippet=result.get('body', ''),
                    source="web"
                )
                search_results.append(search_result)

            logger.info(f"Found {len(search_results)} raw results")
            return search_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise SearchError(f"Web search failed: {e}", query)

    def _is_cached(self, cache_key: str) -> bool:
        """Check if query results are cached and still valid."""
        if cache_key not in self.cache:
            return False

        expiry_time = self.cache_expiry.get(cache_key)
        if not expiry_time or datetime.now() > expiry_time:
            # Cache expired, remove it
            self.cache.pop(cache_key, None)
            self.cache_expiry.pop(cache_key, None)
            return False

        return True

    def clear_cache(self) -> None:
        """Clear the search cache."""
        self.cache.clear()
        self.cache_expiry.clear()

    def get_metadata(self) -> ToolMetadata:
        """Get metadata for the web search tool."""
        return ToolMetadata(
            name=self.name,
            tool_type=self.tool_type,
            description="Search the web for information using DuckDuckGo",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 20
                    },
                    "region": {
                        "type": "string",
                        "description": "Search region (e.g., 'us-en', 'uk-en')",
                        "default": "us-en"
                    },
                    "time_range": {
                        "type": "string",
                        "description": "Time range for results (e.g., 'd' for day, 'w' for week)",
                        "enum": ["d", "w", "m", "y"]
                    }
                },
                "required": ["query"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "items": {"type": "object"}
                    },
                    "summary": {"type": "string"},
                    "total_found": {"type": "integer"},
                    "total_returned": {"type": "integer"}
                }
            },
            capabilities=[
                "web_search",
                "information_retrieval",
                "fact_verification",
                "current_events",
                "general_knowledge"
            ],
            limitations=[
                "Rate limited to prevent abuse",
                "Results cached for 1 hour",
                "Limited to text-based results",
                "Subject to search engine availability"
            ],
            examples=[
                {
                    "input": "latest developments in artificial intelligence 2024",
                    "output": "Recent news and articles about AI developments"
                },
                {
                    "input": "Python programming tutorial",
                    "output": "Educational resources and tutorials for Python"
                }
            ]
        )


# Register search tools
@tool(
    name="search_web",
    tool_type=ToolType.SEARCH,
    description="Search the web for information",
    timeout=15.0,
    cost_per_use=0.002
)
async def search_web(
    query: str,
    max_results: int = 10,
    recent_only: bool = False
) -> dict[str, Any]:
    """Search the web for information.
    
    Args:
        query: Search query
        max_results: Maximum number of results to return (1-20)
        recent_only: If True, limit to recent results
        
    Returns:
        Dictionary containing search results and summary
    """

    search_tool = WebSearchTool()

    # Set time range if recent results requested
    time_range = "w" if recent_only else None

    result = await search_tool.execute(
        query=query,
        max_results=max_results,
        time_range=time_range
    )

    if not result.success:
        raise SearchError(
            result.error_message or "Web search failed",
            query
        )

    return result.output_data


@tool(
    name="fact_check",
    tool_type=ToolType.SEARCH,
    description="Verify facts by searching for reliable sources",
    timeout=20.0,
    cost_per_use=0.003
)
async def fact_check(claim: str) -> dict[str, Any]:
    """Verify a factual claim by searching for reliable sources.
    
    Args:
        claim: Factual claim to verify
        
    Returns:
        Dictionary with verification results and sources
    """

    # Create search queries for fact-checking
    queries = [
        f'"{claim}" fact check',
        f'"{claim}" true false',
        f'"{claim}" verification',
        claim  # Original claim
    ]

    all_results = []

    for query in queries:
        try:
            search_result = await search_web(query, max_results=5)
            if search_result.get('results'):
                all_results.extend(search_result['results'])
        except Exception as e:
            logger.warning(f"Fact check query failed: {query} - {e}")

    # Analyze results for fact-checking indicators
    verification_indicators = {
        "supporting": 0,
        "contradicting": 0,
        "neutral": 0
    }

    reliable_sources = {
        "reuters.com", "ap.org", "bbc.com", "npr.org", "snopes.com",
        "factcheck.org", "politifact.com", "nature.com",
        "science.org", "nih.gov", "cdc.gov", "who.int"
    }

    credible_results = []

    for result in all_results:
        # Check if source is from a reliable domain
        url = result.get('url', '')
        domain = url.split('/')[2] if '/' in url else ''

        if any(reliable in domain for reliable in reliable_sources):
            credible_results.append(result)

        # Analyze content for verification keywords
        content = (result.get('title', '') + ' ' + result.get('snippet', '')).lower()

        if any(word in content for word in ['true', 'confirmed', 'verified', 'correct']):
            verification_indicators["supporting"] += 1
        elif any(word in content for word in ['false', 'debunked', 'incorrect', 'myth']):
            verification_indicators["contradicting"] += 1
        else:
            verification_indicators["neutral"] += 1

    # Determine verification status
    total_results = sum(verification_indicators.values())
    if total_results == 0:
        status = "insufficient_data"
        confidence = 0.0
    else:
        support_ratio = verification_indicators["supporting"] / total_results
        contradict_ratio = verification_indicators["contradicting"] / total_results

        if support_ratio > 0.6:
            status = "likely_true"
            confidence = support_ratio
        elif contradict_ratio > 0.6:
            status = "likely_false"
            confidence = contradict_ratio
        else:
            status = "disputed"
            confidence = max(support_ratio, contradict_ratio)

    return {
        "claim": claim,
        "verification_status": status,
        "confidence": confidence,
        "indicators": verification_indicators,
        "credible_sources": len(credible_results),
        "total_sources": len(all_results),
        "credible_results": credible_results[:5],  # Top 5 credible results
        "summary": f"Verification status: {status} (confidence: {confidence:.2f})"
    }
