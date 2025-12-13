"""
Research Agent for DroidRun.

Production-ready web research capabilities with:
- Multi-provider search (Tavily, Perplexity, Brave)
- Offline mock mode for development
- Memory-backed context retrieval
- Source merging and deduplication
"""

from .research_agent import (
    ResearchAgent,
    ResearchAgentConfig,
    ResearchResult,
)
from .providers import (
    SearchProvider,
    SearchResult,
    TavilyProvider,
    PerplexityProvider,
    BraveSearchProvider,
    MockSearchProvider,
    CompositeSearchProvider,
)

__all__ = [
    "ResearchAgent",
    "ResearchAgentConfig",
    "ResearchResult",
    "SearchResult",
    "SearchProvider",
    "TavilyProvider",
    "PerplexityProvider",
    "BraveSearchProvider",
    "MockSearchProvider",
    "CompositeSearchProvider",
]
