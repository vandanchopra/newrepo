"""
Research Agent for DroidRun.

Production-ready deep research capabilities with:
- Multi-provider web search
- Memory integration for context
- Source merging and deduplication
- Offline mode support
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import asyncio
import logging
import time

try:
    from .providers import (
        SearchProvider,
        SearchResult,
        MockSearchProvider,
        CompositeSearchProvider,
    )
except ImportError:
    # Fallback for standalone testing
    from providers import (
        SearchProvider,
        SearchResult,
        MockSearchProvider,
        CompositeSearchProvider,
    )

logger = logging.getLogger("droidrun.research")


@dataclass
class ResearchAgentConfig:
    """Configuration for Research Agent."""

    # Search configuration
    max_results: int = 10
    max_results_per_provider: int = 5
    timeout_seconds: float = 30.0

    # Offline mode
    offline_mode: bool = False
    mock_external_apis: bool = False

    # Memory integration
    include_memory: bool = True
    memory_top_k: int = 3

    # Result processing
    deduplicate: bool = True
    min_score: float = 0.1

    # Caching
    cache_results: bool = True
    cache_ttl_seconds: float = 300.0


@dataclass
class ResearchResult:
    """Result from research operation."""

    query: str
    results: List[SearchResult] = field(default_factory=list)
    memory_context: str = ""
    summary: str = ""
    sources_used: List[str] = field(default_factory=list)
    duration_ms: float = 0.0
    cached: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class ResearchAgent:
    """
    Production-ready research agent for web search and information gathering.

    Features:
    - Multi-provider search (Tavily, Perplexity, Brave)
    - Automatic fallback to mock provider in offline mode
    - Memory integration for context-aware research
    - Result caching
    - Source merging and deduplication
    """

    def __init__(
        self,
        providers: Optional[List[SearchProvider]] = None,
        config: Optional[ResearchAgentConfig] = None,
        memory_manager=None,
    ):
        self.config = config or ResearchAgentConfig()
        self.memory_manager = memory_manager

        # Initialize providers
        if providers:
            self._providers = providers
        else:
            self._providers = self._create_default_providers()

        # Cache
        self._cache: Dict[str, tuple] = {}  # query -> (result, timestamp)

    def _create_default_providers(self) -> List[SearchProvider]:
        """Create default providers based on configuration."""
        if self.config.offline_mode or self.config.mock_external_apis:
            logger.info("Research agent running in offline mode with mock provider")
            return [MockSearchProvider()]

        providers = []

        # Try to create real providers
        try:
            try:
                from .providers import TavilyProvider
            except ImportError:
                from providers import TavilyProvider
            tavily = TavilyProvider()
            if tavily.is_available:
                providers.append(tavily)
                logger.info("Tavily provider available")
        except Exception as e:
            logger.debug(f"Tavily provider not available: {e}")

        try:
            try:
                from .providers import BraveSearchProvider
            except ImportError:
                from providers import BraveSearchProvider
            brave = BraveSearchProvider()
            if brave.is_available:
                providers.append(brave)
                logger.info("Brave Search provider available")
        except Exception as e:
            logger.debug(f"Brave provider not available: {e}")

        try:
            try:
                from .providers import PerplexityProvider
            except ImportError:
                from providers import PerplexityProvider
            perplexity = PerplexityProvider()
            if perplexity.is_available:
                providers.append(perplexity)
                logger.info("Perplexity provider available")
        except Exception as e:
            logger.debug(f"Perplexity provider not available: {e}")

        # Fallback to mock if no providers available
        if not providers:
            logger.warning(
                "No search providers available. Using mock provider. "
                "Set API keys for Tavily, Brave, or Perplexity for real search."
            )
            providers.append(MockSearchProvider())

        return providers

    def _check_cache(self, query: str) -> Optional[ResearchResult]:
        """Check if query is cached and not expired."""
        if not self.config.cache_results:
            return None

        if query in self._cache:
            result, timestamp = self._cache[query]
            age = time.time() - timestamp

            if age < self.config.cache_ttl_seconds:
                result.cached = True
                return result
            else:
                # Expired
                del self._cache[query]

        return None

    def _cache_result(self, query: str, result: ResearchResult):
        """Cache a research result."""
        if self.config.cache_results:
            self._cache[query] = (result, time.time())

    async def _get_memory_context(self, query: str) -> str:
        """Get relevant context from memory."""
        if not self.config.include_memory or not self.memory_manager:
            return ""

        try:
            context = await self.memory_manager.get_context_for_task(
                task=query,
                goal=query,
                max_context_length=1000,
            )
            return context
        except Exception as e:
            logger.warning(f"Failed to get memory context: {e}")
            return ""

    def _deduplicate_results(
        self,
        results: List[SearchResult],
    ) -> List[SearchResult]:
        """Remove duplicate results by URL."""
        seen_urls = set()
        unique = []

        for result in results:
            # Normalize URL
            url = result.url.lower().rstrip("/")

            if url not in seen_urls:
                seen_urls.add(url)
                unique.append(result)

        return unique

    def _generate_summary(
        self,
        query: str,
        results: List[SearchResult],
        memory_context: str,
    ) -> str:
        """Generate a summary of research findings."""
        if not results:
            return f"No results found for: {query}"

        summary_parts = [f"Research findings for: {query}\n"]

        if memory_context:
            summary_parts.append("Context from memory:\n" + memory_context[:500] + "\n")

        summary_parts.append(f"\nTop {len(results)} results:\n")

        for i, result in enumerate(results[:5], 1):
            summary_parts.append(
                f"{i}. {result.title}\n"
                f"   Source: {result.source} | Score: {result.score:.2f}\n"
                f"   {result.snippet[:200]}...\n"
            )

        return "".join(summary_parts)

    async def research(
        self,
        query: str,
        max_results: Optional[int] = None,
        include_memory: Optional[bool] = None,
    ) -> ResearchResult:
        """
        Perform research on a query.

        Args:
            query: The research query
            max_results: Override max results setting
            include_memory: Override memory inclusion setting

        Returns:
            ResearchResult with findings
        """
        start_time = time.time()
        max_results = max_results or self.config.max_results
        include_memory = include_memory if include_memory is not None else self.config.include_memory

        # Check cache
        cached = self._check_cache(query)
        if cached:
            logger.debug(f"Cache hit for query: {query[:50]}")
            return cached

        # Get memory context
        memory_context = ""
        if include_memory:
            memory_context = await self._get_memory_context(query)

        # Execute searches
        all_results = []
        sources_used = []

        try:
            # Create tasks for each provider
            async def search_provider(provider: SearchProvider) -> List[SearchResult]:
                try:
                    results = await asyncio.wait_for(
                        provider.search(
                            query,
                            max_results=self.config.max_results_per_provider,
                        ),
                        timeout=self.config.timeout_seconds,
                    )
                    return results
                except asyncio.TimeoutError:
                    logger.warning(f"Provider {provider.name} timed out")
                    return []
                except Exception as e:
                    logger.warning(f"Provider {provider.name} failed: {e}")
                    return []

            # Run all providers in parallel
            tasks = [search_provider(p) for p in self._providers]
            results_lists = await asyncio.gather(*tasks)

            for provider, results in zip(self._providers, results_lists):
                if results:
                    sources_used.append(provider.name)
                    all_results.extend(results)

        except Exception as e:
            logger.error(f"Research failed: {e}")
            raise

        # Deduplicate
        if self.config.deduplicate:
            all_results = self._deduplicate_results(all_results)

        # Filter by score
        all_results = [r for r in all_results if r.score >= self.config.min_score]

        # Sort by score
        all_results.sort(key=lambda x: x.score, reverse=True)

        # Limit results
        all_results = all_results[:max_results]

        # Generate summary
        summary = self._generate_summary(query, all_results, memory_context)

        duration_ms = (time.time() - start_time) * 1000

        result = ResearchResult(
            query=query,
            results=all_results,
            memory_context=memory_context,
            summary=summary,
            sources_used=sources_used,
            duration_ms=duration_ms,
            cached=False,
            metadata={
                "provider_count": len(self._providers),
                "total_raw_results": len(all_results),
            },
        )

        # Cache result
        self._cache_result(query, result)

        logger.info(
            f"Research complete: {len(all_results)} results from {sources_used} "
            f"in {duration_ms:.0f}ms"
        )

        return result

    async def research_task(
        self,
        task: str,
        additional_context: str = "",
    ) -> ResearchResult:
        """
        Research a task with optional additional context.

        This is a convenience method that builds a more detailed query
        from the task description.

        Args:
            task: Task description to research
            additional_context: Optional additional context

        Returns:
            ResearchResult
        """
        # Build research query
        query_parts = [task]

        if additional_context:
            query_parts.append(additional_context)

        query = " ".join(query_parts)

        return await self.research(query)

    async def find_similar_solutions(
        self,
        problem: str,
        domain: str = "android automation",
    ) -> ResearchResult:
        """
        Find similar solutions to a problem.

        Args:
            problem: Problem description
            domain: Domain context

        Returns:
            ResearchResult with potential solutions
        """
        query = f"how to {problem} in {domain} solution tutorial"
        return await self.research(query)

    async def get_app_information(
        self,
        app_name: str,
        info_type: str = "usage",
    ) -> ResearchResult:
        """
        Get information about a mobile app.

        Args:
            app_name: Name of the app
            info_type: Type of info (usage, features, automation, etc.)

        Returns:
            ResearchResult about the app
        """
        query = f"{app_name} app {info_type} guide"
        return await self.research(query)

    def clear_cache(self):
        """Clear the result cache."""
        self._cache.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics."""
        available_providers = [p.name for p in self._providers if p.is_available]

        return {
            "providers": [p.name for p in self._providers],
            "available_providers": available_providers,
            "cache_size": len(self._cache),
            "offline_mode": self.config.offline_mode,
            "memory_enabled": self.config.include_memory and self.memory_manager is not None,
        }


# Convenience function
def create_research_agent(
    offline: bool = False,
    **kwargs,
) -> ResearchAgent:
    """
    Create a research agent with sensible defaults.

    Args:
        offline: Run in offline mode with mock provider
        **kwargs: Additional ResearchAgentConfig parameters

    Returns:
        Configured ResearchAgent
    """
    config = ResearchAgentConfig(
        offline_mode=offline,
        mock_external_apis=offline,
        **kwargs,
    )
    return ResearchAgent(config=config)
