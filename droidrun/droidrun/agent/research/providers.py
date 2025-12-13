"""
Search Providers for Research Agent.

Production-ready implementations of various search APIs
with offline fallback support.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import asyncio
import hashlib
import logging
import os
import time

logger = logging.getLogger("droidrun.research")


@dataclass
class SearchResult:
    """Single search result."""

    title: str
    url: str
    snippet: str
    score: float = 0.0
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class SearchProvider(ABC):
    """Abstract base class for search providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass

    @abstractmethod
    async def search(
        self,
        query: str,
        max_results: int = 10,
        **kwargs,
    ) -> List[SearchResult]:
        """Execute search and return results."""
        pass

    @property
    def is_available(self) -> bool:
        """Check if provider is available (has API key, etc.)."""
        return True


class TavilyProvider(SearchProvider):
    """Tavily search provider - optimized for AI research."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        search_depth: str = "advanced",
        include_raw_content: bool = False,
    ):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self.search_depth = search_depth
        self.include_raw_content = include_raw_content
        self._client = None

        if self.api_key:
            try:
                from tavily import TavilyClient
                self._client = TavilyClient(api_key=self.api_key)
            except ImportError:
                logger.warning("tavily-python not installed. Install: pip install tavily-python")

    @property
    def name(self) -> str:
        return "tavily"

    @property
    def is_available(self) -> bool:
        return self._client is not None

    async def search(
        self,
        query: str,
        max_results: int = 10,
        **kwargs,
    ) -> List[SearchResult]:
        if not self._client:
            raise RuntimeError("Tavily client not initialized")

        # Run in executor since tavily is sync
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._client.search(
                query=query,
                search_depth=self.search_depth,
                max_results=max_results,
                include_raw_content=self.include_raw_content,
            )
        )

        results = []
        for item in response.get("results", []):
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", ""),
                score=item.get("score", 0.0),
                source=self.name,
                metadata={
                    "raw_content": item.get("raw_content", ""),
                    "published_date": item.get("published_date"),
                },
            ))

        return results


class PerplexityProvider(SearchProvider):
    """Perplexity search provider - AI-powered search."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.1-sonar-small-128k-online",
    ):
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        self.model = model

    @property
    def name(self) -> str:
        return "perplexity"

    @property
    def is_available(self) -> bool:
        return self.api_key is not None

    async def search(
        self,
        query: str,
        max_results: int = 10,
        **kwargs,
    ) -> List[SearchResult]:
        if not self.api_key:
            raise RuntimeError("Perplexity API key not configured")

        import aiohttp

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": query}],
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.perplexity.ai/chat/completions",
                headers=headers,
                json=payload,
            ) as response:
                if response.status != 200:
                    raise RuntimeError(f"Perplexity API error: {response.status}")

                data = await response.json()

        # Parse response
        results = []
        if "choices" in data and data["choices"]:
            content = data["choices"][0].get("message", {}).get("content", "")

            # Perplexity returns a synthesized answer, not individual results
            results.append(SearchResult(
                title=f"Perplexity answer for: {query[:50]}...",
                url="https://perplexity.ai",
                snippet=content,
                score=1.0,
                source=self.name,
                metadata={
                    "citations": data.get("citations", []),
                    "model": self.model,
                },
            ))

        return results


class BraveSearchProvider(SearchProvider):
    """Brave Search provider - privacy-focused search."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        country: str = "us",
        search_lang: str = "en",
    ):
        self.api_key = api_key or os.getenv("BRAVE_SEARCH_API_KEY")
        self.country = country
        self.search_lang = search_lang
        self.base_url = "https://api.search.brave.com/res/v1/web/search"

    @property
    def name(self) -> str:
        return "brave"

    @property
    def is_available(self) -> bool:
        return self.api_key is not None

    async def search(
        self,
        query: str,
        max_results: int = 10,
        **kwargs,
    ) -> List[SearchResult]:
        if not self.api_key:
            raise RuntimeError("Brave Search API key not configured")

        import aiohttp

        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key,
        }

        params = {
            "q": query,
            "count": max_results,
            "country": self.country,
            "search_lang": self.search_lang,
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(
                self.base_url,
                headers=headers,
                params=params,
            ) as response:
                if response.status != 200:
                    raise RuntimeError(f"Brave Search API error: {response.status}")

                data = await response.json()

        results = []
        web_results = data.get("web", {}).get("results", [])

        for i, item in enumerate(web_results[:max_results]):
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("description", ""),
                score=1.0 - (i / max_results),  # Rank-based score
                source=self.name,
                metadata={
                    "age": item.get("age"),
                    "family_friendly": item.get("family_friendly", True),
                },
            ))

        return results


class MockSearchProvider(SearchProvider):
    """
    Mock search provider for offline development and testing.

    Returns deterministic results based on query hash.
    """

    def __init__(self, latency_ms: float = 100):
        self.latency_ms = latency_ms

    @property
    def name(self) -> str:
        return "mock"

    @property
    def is_available(self) -> bool:
        return True

    def _generate_mock_results(self, query: str, count: int) -> List[SearchResult]:
        """Generate deterministic mock results."""
        # Use query hash for determinism
        query_hash = hashlib.md5(query.encode()).hexdigest()

        results = []
        for i in range(count):
            # Generate unique but deterministic content
            seed = f"{query_hash}_{i}"
            seed_hash = hashlib.md5(seed.encode()).hexdigest()

            results.append(SearchResult(
                title=f"Mock Result {i+1}: {query[:30]}...",
                url=f"https://example.com/mock/{seed_hash[:8]}",
                snippet=(
                    f"This is a mock search result for '{query}'. "
                    f"Result {i+1} of {count}. "
                    f"In production, this would contain real search content. "
                    f"Hash: {seed_hash[:16]}"
                ),
                score=1.0 - (i * 0.1),
                source=self.name,
                metadata={
                    "mock": True,
                    "seed": seed_hash[:8],
                    "generated_at": time.time(),
                },
            ))

        return results

    async def search(
        self,
        query: str,
        max_results: int = 10,
        **kwargs,
    ) -> List[SearchResult]:
        # Simulate latency
        await asyncio.sleep(self.latency_ms / 1000)

        return self._generate_mock_results(query, max_results)


class CompositeSearchProvider(SearchProvider):
    """
    Composite provider that combines results from multiple providers.

    Features:
    - Parallel execution
    - Result merging and deduplication
    - Fallback to available providers
    """

    def __init__(
        self,
        providers: List[SearchProvider],
        fallback_to_mock: bool = True,
    ):
        self.providers = providers
        self.fallback_to_mock = fallback_to_mock

    @property
    def name(self) -> str:
        return "composite"

    @property
    def is_available(self) -> bool:
        return any(p.is_available for p in self.providers) or self.fallback_to_mock

    async def search(
        self,
        query: str,
        max_results: int = 10,
        **kwargs,
    ) -> List[SearchResult]:
        # Get available providers
        available = [p for p in self.providers if p.is_available]

        if not available and self.fallback_to_mock:
            logger.warning("No search providers available, using mock")
            available = [MockSearchProvider()]

        if not available:
            raise RuntimeError("No search providers available")

        # Execute searches in parallel
        tasks = [
            provider.search(query, max_results=max_results, **kwargs)
            for provider in available
        ]

        results_lists = await asyncio.gather(*tasks, return_exceptions=True)

        # Merge results
        all_results = []
        seen_urls = set()

        for provider, results in zip(available, results_lists):
            if isinstance(results, Exception):
                logger.warning(f"Provider {provider.name} failed: {results}")
                continue

            for result in results:
                # Deduplicate by URL
                if result.url not in seen_urls:
                    seen_urls.add(result.url)
                    all_results.append(result)

        # Sort by score
        all_results.sort(key=lambda x: x.score, reverse=True)

        return all_results[:max_results]
