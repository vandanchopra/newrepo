"""
Research Agent for DroidRun.

Production-ready web research capabilities with:
- Multi-provider search (Tavily, Perplexity, Brave)
- Offline mock mode for development
- Memory-backed context retrieval
- Source merging and deduplication
"""

from droidrun.agent.research.research_agent import (
    ResearchAgent,
    ResearchAgentConfig,
    ResearchResult,
    SearchResult,
)
from droidrun.agent.research.providers import (
    SearchProvider,
    TavilyProvider,
    PerplexityProvider,
    BraveSearchProvider,
    MockSearchProvider,
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
]
