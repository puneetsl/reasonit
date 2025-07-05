"""
Wikipedia API Integration and Content Processing.

This module provides comprehensive Wikipedia search capabilities with intelligent
content extraction, disambiguation handling, and structured data processing.
"""

import asyncio
import json
import logging
import re
import time
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib

import aiohttp
from bs4 import BeautifulSoup


@dataclass
class WikipediaPage:
    """Represents a Wikipedia page with processed content."""
    title: str
    page_id: int
    url: str
    summary: str
    full_text: str
    infobox: Dict[str, str] = field(default_factory=dict)
    sections: Dict[str, str] = field(default_factory=dict)
    categories: List[str] = field(default_factory=list)
    links: List[str] = field(default_factory=list)
    images: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    last_modified: Optional[datetime] = None
    language: str = "en"
    disambiguation: bool = False
    redirect_target: Optional[str] = None


@dataclass
class SearchResult:
    """Result of Wikipedia search with multiple pages and metadata."""
    query: str
    pages: List[WikipediaPage] = field(default_factory=list)
    disambiguation_pages: List[WikipediaPage] = field(default_factory=list)
    related_searches: List[str] = field(default_factory=list)
    search_time: float = 0.0
    total_results: int = 0
    search_metadata: Dict[str, Any] = field(default_factory=dict)


class WikipediaSearchTool:
    """
    Advanced Wikipedia search and content extraction tool.
    
    Provides intelligent search with disambiguation handling, content processing,
    and structured data extraction from Wikipedia articles.
    """
    
    def __init__(self, 
                 language: str = "en",
                 user_agent: str = "ReasonIt/1.0 (Educational Research Tool)",
                 rate_limit_delay: float = 0.1,
                 max_concurrent_requests: int = 5):
        self.language = language
        self.user_agent = user_agent
        self.rate_limit_delay = rate_limit_delay
        self.max_concurrent_requests = max_concurrent_requests
        
        self.logger = logging.getLogger(__name__)
        
        # API endpoints
        self.base_url = f"https://{language}.wikipedia.org/api/rest_v1"
        self.api_url = f"https://{language}.wikipedia.org/w/api.php"
        
        # Session for connection pooling
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Rate limiting
        self.last_request_time = 0.0
        self.request_semaphore = asyncio.Semaphore(max_concurrent_requests)
        
        # Search cache
        self.search_cache: Dict[str, SearchResult] = {}
        self.cache_ttl = timedelta(hours=1)
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def _ensure_session(self):
        """Ensure aiohttp session is available."""
        if not self.session or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers={"User-Agent": self.user_agent}
            )
    
    async def _rate_limit(self):
        """Apply rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        
        self.last_request_time = time.time()
    
    async def _make_request(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make a rate-limited HTTP request to Wikipedia API."""
        async with self.request_semaphore:
            await self._rate_limit()
            await self._ensure_session()
            
            try:
                async with self.session.get(url, params=params) as response:
                    response.raise_for_status()
                    return await response.json()
            except Exception as e:
                self.logger.error(f"Wikipedia API request failed: {e}")
                raise
    
    async def search_articles(self, 
                            query: str, 
                            limit: int = 10,
                            include_content: bool = True,
                            handle_disambiguation: bool = True) -> SearchResult:
        """
        Search Wikipedia articles with intelligent content processing.
        
        Args:
            query: Search query
            limit: Maximum number of results
            include_content: Whether to fetch full content
            handle_disambiguation: Whether to handle disambiguation pages
            
        Returns:
            SearchResult with processed pages
        """
        start_time = time.time()
        
        # Check cache first
        cache_key = self._create_cache_key(query, limit, include_content)
        if cache_key in self.search_cache:
            cached_result = self.search_cache[cache_key]
            if datetime.now() - datetime.fromtimestamp(cached_result.search_metadata.get("cached_at", 0)) < self.cache_ttl:
                self.logger.debug(f"Using cached result for: {query}")
                return cached_result
        
        self.logger.info(f"Searching Wikipedia for: {query}")
        
        # Step 1: Search for articles
        search_results = await self._search_titles(query, limit)
        
        # Step 2: Get page details for each result
        pages = []
        disambiguation_pages = []
        
        for title in search_results["titles"]:
            try:
                page = await self._get_page_details(title, include_content)
                
                if page.disambiguation:
                    disambiguation_pages.append(page)
                    
                    # If handling disambiguation, search for specific pages
                    if handle_disambiguation:
                        disambig_options = await self._extract_disambiguation_options(page)
                        for option in disambig_options[:3]:  # Limit to top 3 options
                            try:
                                option_page = await self._get_page_details(option, include_content)
                                if not option_page.disambiguation:
                                    pages.append(option_page)
                            except Exception as e:
                                self.logger.debug(f"Failed to get disambiguation option {option}: {e}")
                else:
                    pages.append(page)
                    
            except Exception as e:
                self.logger.warning(f"Failed to get page details for {title}: {e}")
                continue
        
        # Step 3: Generate related searches
        related_searches = await self._generate_related_searches(query, pages)
        
        search_time = time.time() - start_time
        
        result = SearchResult(
            query=query,
            pages=pages,
            disambiguation_pages=disambiguation_pages,
            related_searches=related_searches,
            search_time=search_time,
            total_results=len(pages) + len(disambiguation_pages),
            search_metadata={
                "api_calls": len(search_results["titles"]) + 1,
                "cached_at": time.time(),
                "disambiguation_handled": handle_disambiguation
            }
        )
        
        # Cache the result
        self.search_cache[cache_key] = result
        
        self.logger.info(f"Search completed: {len(pages)} pages, {len(disambiguation_pages)} disambiguation pages in {search_time:.2f}s")
        
        return result
    
    async def _search_titles(self, query: str, limit: int) -> Dict[str, Any]:
        """Search for Wikipedia article titles."""
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "srinfo": "totalhits",
            "srprop": "size|snippet|timestamp"
        }
        
        response = await self._make_request(self.api_url, params)
        
        search_data = response.get("query", {}).get("search", [])
        titles = [item["title"] for item in search_data]
        
        return {
            "titles": titles,
            "total_hits": response.get("query", {}).get("searchinfo", {}).get("totalhits", 0),
            "search_data": search_data
        }
    
    async def _get_page_details(self, title: str, include_content: bool = True) -> WikipediaPage:
        """Get detailed information for a Wikipedia page."""
        # Get basic page info
        page_info = await self._get_page_info(title)
        
        # Check if it's a disambiguation page
        is_disambiguation = "disambiguation" in page_info.get("categories", [])
        
        # Get page content
        if include_content:
            content_data = await self._get_page_content(title)
        else:
            content_data = {"extract": "", "sections": {}, "infobox": {}}
        
        # Extract structured data
        infobox = await self._extract_infobox(title) if include_content else {}
        categories = await self._extract_categories(title)
        
        return WikipediaPage(
            title=page_info["title"],
            page_id=page_info["pageid"],
            url=f"https://{self.language}.wikipedia.org/wiki/{title.replace(' ', '_')}",
            summary=content_data.get("extract", "")[:2000],  # First 2000 chars
            full_text=content_data.get("extract", ""),
            infobox=infobox,
            sections=content_data.get("sections", {}),
            categories=categories,
            links=[],  # Can be populated if needed
            images=[],  # Can be populated if needed
            references=[],  # Can be populated if needed
            last_modified=datetime.fromisoformat(page_info.get("touched", "").replace("Z", "+00:00")) if page_info.get("touched") else None,
            language=self.language,
            disambiguation=is_disambiguation,
            redirect_target=page_info.get("redirect_target")
        )
    
    async def _get_page_info(self, title: str) -> Dict[str, Any]:
        """Get basic page information."""
        params = {
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "info|categories",
            "inprop": "url",
            "cllimit": "max"
        }
        
        response = await self._make_request(self.api_url, params)
        pages = response.get("query", {}).get("pages", {})
        
        page_data = next(iter(pages.values()))
        
        # Extract categories
        categories = []
        if "categories" in page_data:
            categories = [cat["title"].replace("Category:", "") for cat in page_data["categories"]]
        
        return {
            "title": page_data.get("title", title),
            "pageid": page_data.get("pageid", 0),
            "touched": page_data.get("touched"),
            "categories": categories,
            "redirect_target": page_data.get("redirect", {}).get("to") if "redirect" in page_data else None
        }
    
    async def _get_page_content(self, title: str) -> Dict[str, Any]:
        """Get page content with sections."""
        # Get page extract
        extract_params = {
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts",
            "exintro": False,
            "explaintext": True,
            "exsectionformat": "wiki"
        }
        
        extract_response = await self._make_request(self.api_url, extract_params)
        extract_pages = extract_response.get("query", {}).get("pages", {})
        extract_data = next(iter(extract_pages.values()))
        extract_text = extract_data.get("extract", "")
        
        # Get sections
        sections = await self._get_page_sections(title)
        
        return {
            "extract": extract_text,
            "sections": sections
        }
    
    async def _get_page_sections(self, title: str) -> Dict[str, str]:
        """Get page sections with content."""
        # Get section structure
        sections_params = {
            "action": "parse",
            "format": "json",
            "page": title,
            "prop": "sections"
        }
        
        try:
            sections_response = await self._make_request(self.api_url, sections_params)
            sections_data = sections_response.get("parse", {}).get("sections", [])
            
            sections = {}
            for section in sections_data[:10]:  # Limit to first 10 sections
                section_title = section.get("line", "")
                section_index = section.get("index", "")
                
                if section_title and section_index:
                    # Get section content
                    section_content = await self._get_section_content(title, section_index)
                    sections[section_title] = section_content
            
            return sections
            
        except Exception as e:
            self.logger.debug(f"Failed to get sections for {title}: {e}")
            return {}
    
    async def _get_section_content(self, title: str, section_index: str) -> str:
        """Get content for a specific section."""
        params = {
            "action": "parse",
            "format": "json",
            "page": title,
            "section": section_index,
            "prop": "text"
        }
        
        try:
            response = await self._make_request(self.api_url, params)
            html_content = response.get("parse", {}).get("text", {}).get("*", "")
            
            # Convert HTML to plain text
            soup = BeautifulSoup(html_content, 'html.parser')
            return soup.get_text(strip=True)[:3000]  # Limit section content
            
        except Exception as e:
            self.logger.debug(f"Failed to get section content: {e}")
            return ""
    
    async def _extract_infobox(self, title: str) -> Dict[str, str]:
        """Extract infobox data from a Wikipedia page."""
        params = {
            "action": "parse",
            "format": "json",
            "page": title,
            "prop": "text"
        }
        
        try:
            response = await self._make_request(self.api_url, params)
            html_content = response.get("parse", {}).get("text", {}).get("*", "")
            
            # Parse HTML to extract infobox
            soup = BeautifulSoup(html_content, 'html.parser')
            infobox = soup.find("table", class_=re.compile("infobox"))
            
            if not infobox:
                return {}
            
            infobox_data = {}
            rows = infobox.find_all("tr")
            
            for row in rows:
                # Look for rows with header and data
                header = row.find("th")
                data = row.find("td")
                
                if header and data:
                    key = header.get_text(strip=True)
                    value = data.get_text(strip=True)
                    
                    if key and value:
                        infobox_data[key] = value
            
            return infobox_data
            
        except Exception as e:
            self.logger.debug(f"Failed to extract infobox for {title}: {e}")
            return {}
    
    async def _extract_categories(self, title: str) -> List[str]:
        """Extract categories for a Wikipedia page."""
        params = {
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "categories",
            "cllimit": "max"
        }
        
        try:
            response = await self._make_request(self.api_url, params)
            pages = response.get("query", {}).get("pages", {})
            page_data = next(iter(pages.values()))
            
            categories = []
            if "categories" in page_data:
                for cat in page_data["categories"]:
                    cat_title = cat["title"].replace("Category:", "")
                    categories.append(cat_title)
            
            return categories
            
        except Exception as e:
            self.logger.debug(f"Failed to extract categories for {title}: {e}")
            return []
    
    async def _extract_disambiguation_options(self, disambiguation_page: WikipediaPage) -> List[str]:
        """Extract options from a disambiguation page."""
        content = disambiguation_page.full_text
        
        # Simple extraction of linked terms from disambiguation content
        options = []
        
        # Look for patterns like "* [[Link|Text]]" or "* [[Link]]"
        link_patterns = [
            r'\[\[([^\]|]+)\|[^\]]+\]\]',  # [[Link|Text]]
            r'\[\[([^\]]+)\]\]'           # [[Link]]
        ]
        
        for pattern in link_patterns:
            matches = re.findall(pattern, content)
            options.extend(matches)
        
        # Clean up and deduplicate
        clean_options = []
        for option in options:
            option = option.strip()
            if option and option not in clean_options and len(option) > 2:
                clean_options.append(option)
        
        return clean_options[:10]  # Return top 10 options
    
    async def _generate_related_searches(self, original_query: str, pages: List[WikipediaPage]) -> List[str]:
        """Generate related search suggestions based on retrieved pages."""
        related = set()
        
        # Extract key terms from page titles and categories
        for page in pages[:3]:  # Use top 3 pages
            # Add related terms from categories
            for category in page.categories[:5]:
                # Skip overly generic categories
                if not any(generic in category.lower() for generic in ["articles", "pages", "wikipedia", "disambiguation"]):
                    related.add(category)
            
            # Extract key terms from title
            title_words = page.title.split()
            for word in title_words:
                if len(word) > 3 and word.lower() != original_query.lower():
                    related.add(word)
        
        # Convert to list and limit
        return list(related)[:5]
    
    def _create_cache_key(self, query: str, limit: int, include_content: bool) -> str:
        """Create a cache key for search parameters."""
        key_data = f"{query}_{limit}_{include_content}_{self.language}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def get_page_by_title(self, title: str, include_content: bool = True) -> Optional[WikipediaPage]:
        """Get a specific Wikipedia page by exact title."""
        try:
            return await self._get_page_details(title, include_content)
        except Exception as e:
            self.logger.error(f"Failed to get page '{title}': {e}")
            return None
    
    async def search_in_category(self, category: str, limit: int = 20) -> List[str]:
        """Search for pages in a specific Wikipedia category."""
        params = {
            "action": "query",
            "format": "json",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmlimit": limit,
            "cmtype": "page"
        }
        
        try:
            response = await self._make_request(self.api_url, params)
            members = response.get("query", {}).get("categorymembers", [])
            return [member["title"] for member in members]
        except Exception as e:
            self.logger.error(f"Failed to search category '{category}': {e}")
            return []
    
    def clear_cache(self):
        """Clear the search cache."""
        self.search_cache.clear()
        self.logger.info("Search cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache usage statistics."""
        return {
            "cache_size": len(self.search_cache),
            "cached_queries": list(self.search_cache.keys()),
            "cache_ttl_hours": self.cache_ttl.total_seconds() / 3600
        }


# Convenience functions for easy integration

async def quick_wikipedia_search(query: str, limit: int = 5) -> SearchResult:
    """Quick Wikipedia search with default settings."""
    async with WikipediaSearchTool() as search_tool:
        return await search_tool.search_articles(query, limit=limit)


async def get_wikipedia_page(title: str) -> Optional[WikipediaPage]:
    """Get a single Wikipedia page by title."""
    async with WikipediaSearchTool() as search_tool:
        return await search_tool.get_page_by_title(title)