"""
Tests for Research Agent.

Production-ready tests for:
- ResearchAgent
- Search providers (Mock, Tavily, etc.)
- Offline mode
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestMockSearchProvider(unittest.TestCase):
    """Tests for MockSearchProvider."""

    def test_mock_provider_available(self):
        """Test that mock provider is always available."""
        from droidrun.agent.research.providers import MockSearchProvider

        provider = MockSearchProvider()

        self.assertTrue(provider.is_available)
        self.assertEqual(provider.name, "mock")

    def test_mock_provider_returns_results(self):
        """Test that mock provider returns results."""
        from droidrun.agent.research.providers import MockSearchProvider

        provider = MockSearchProvider(latency_ms=0)

        results = asyncio.run(provider.search("test query", max_results=5))

        self.assertEqual(len(results), 5)
        for result in results:
            self.assertIsNotNone(result.title)
            self.assertIsNotNone(result.url)
            self.assertIsNotNone(result.snippet)

    def test_mock_provider_deterministic(self):
        """Test that mock provider is deterministic."""
        from droidrun.agent.research.providers import MockSearchProvider

        provider = MockSearchProvider(latency_ms=0)

        results1 = asyncio.run(provider.search("same query", max_results=3))
        results2 = asyncio.run(provider.search("same query", max_results=3))

        for r1, r2 in zip(results1, results2):
            self.assertEqual(r1.url, r2.url)
            self.assertEqual(r1.snippet, r2.snippet)

    def test_mock_provider_different_queries(self):
        """Test that different queries produce different results."""
        from droidrun.agent.research.providers import MockSearchProvider

        provider = MockSearchProvider(latency_ms=0)

        results1 = asyncio.run(provider.search("query one", max_results=3))
        results2 = asyncio.run(provider.search("query two", max_results=3))

        self.assertNotEqual(results1[0].url, results2[0].url)


class TestSearchResult(unittest.TestCase):
    """Tests for SearchResult dataclass."""

    def test_search_result_creation(self):
        """Test SearchResult creation."""
        from droidrun.agent.research.providers import SearchResult

        result = SearchResult(
            title="Test Title",
            url="https://example.com",
            snippet="Test snippet",
            score=0.95,
            source="test",
            metadata={"key": "value"},
        )

        self.assertEqual(result.title, "Test Title")
        self.assertEqual(result.url, "https://example.com")
        self.assertEqual(result.score, 0.95)
        self.assertEqual(result.metadata["key"], "value")


class TestResearchAgent(unittest.TestCase):
    """Tests for ResearchAgent."""

    def setUp(self):
        """Set up test fixtures."""
        from droidrun.agent.research.research_agent import (
            ResearchAgent,
            ResearchAgentConfig,
        )
        from droidrun.agent.research.providers import MockSearchProvider

        self.config = ResearchAgentConfig(
            offline_mode=True,
            mock_external_apis=True,
            cache_results=False,
        )
        self.agent = ResearchAgent(
            providers=[MockSearchProvider(latency_ms=0)],
            config=self.config,
        )

    def test_research_returns_results(self):
        """Test that research returns results."""
        result = asyncio.run(self.agent.research("test query"))

        self.assertEqual(result.query, "test query")
        self.assertGreater(len(result.results), 0)
        self.assertIsNotNone(result.summary)
        self.assertGreater(result.duration_ms, 0)

    def test_research_respects_max_results(self):
        """Test that research respects max_results parameter."""
        result = asyncio.run(self.agent.research("test query", max_results=3))

        self.assertLessEqual(len(result.results), 3)

    def test_research_includes_sources(self):
        """Test that research includes source information."""
        result = asyncio.run(self.agent.research("test query"))

        self.assertGreater(len(result.sources_used), 0)
        self.assertIn("mock", result.sources_used)

    def test_research_task_convenience_method(self):
        """Test research_task convenience method."""
        result = asyncio.run(self.agent.research_task(
            task="Open Instagram app",
            additional_context="Android automation",
        ))

        self.assertIsNotNone(result.query)
        self.assertIn("Instagram", result.query)

    def test_find_similar_solutions(self):
        """Test find_similar_solutions method."""
        result = asyncio.run(self.agent.find_similar_solutions(
            problem="login failed",
            domain="mobile testing",
        ))

        self.assertIsNotNone(result.summary)

    def test_get_app_information(self):
        """Test get_app_information method."""
        result = asyncio.run(self.agent.get_app_information(
            app_name="Instagram",
            info_type="automation",
        ))

        self.assertIsNotNone(result.query)

    def test_statistics(self):
        """Test get_statistics method."""
        stats = self.agent.get_statistics()

        self.assertIn("providers", stats)
        self.assertIn("offline_mode", stats)
        self.assertTrue(stats["offline_mode"])


class TestResearchAgentCaching(unittest.TestCase):
    """Tests for ResearchAgent caching."""

    def test_cache_hit(self):
        """Test that caching works."""
        from droidrun.agent.research.research_agent import (
            ResearchAgent,
            ResearchAgentConfig,
        )
        from droidrun.agent.research.providers import MockSearchProvider

        config = ResearchAgentConfig(
            cache_results=True,
            cache_ttl_seconds=60,
        )
        agent = ResearchAgent(
            providers=[MockSearchProvider(latency_ms=0)],
            config=config,
        )

        # First request
        result1 = asyncio.run(agent.research("cached query"))
        self.assertFalse(result1.cached)

        # Second request should be cached
        result2 = asyncio.run(agent.research("cached query"))
        self.assertTrue(result2.cached)

    def test_clear_cache(self):
        """Test cache clearing."""
        from droidrun.agent.research.research_agent import (
            ResearchAgent,
            ResearchAgentConfig,
        )
        from droidrun.agent.research.providers import MockSearchProvider

        config = ResearchAgentConfig(cache_results=True)
        agent = ResearchAgent(
            providers=[MockSearchProvider(latency_ms=0)],
            config=config,
        )

        asyncio.run(agent.research("query to cache"))
        agent.clear_cache()

        result = asyncio.run(agent.research("query to cache"))
        self.assertFalse(result.cached)


class TestResearchAgentOffline(unittest.TestCase):
    """Tests for ResearchAgent in offline mode."""

    def test_offline_mode_uses_mock(self):
        """Test that offline mode uses mock provider."""
        from droidrun.agent.research.research_agent import (
            ResearchAgent,
            ResearchAgentConfig,
        )

        config = ResearchAgentConfig(
            offline_mode=True,
            mock_external_apis=True,
        )
        agent = ResearchAgent(config=config)

        result = asyncio.run(agent.research("offline test"))

        self.assertGreater(len(result.results), 0)
        self.assertIn("mock", result.sources_used)

    def test_no_api_keys_falls_back_to_mock(self):
        """Test that missing API keys fall back to mock."""
        from droidrun.agent.research.research_agent import (
            ResearchAgent,
            ResearchAgentConfig,
        )

        # Create agent without any API keys
        config = ResearchAgentConfig()
        agent = ResearchAgent(config=config)

        # Should still work
        result = asyncio.run(agent.research("test without apis"))

        self.assertGreater(len(result.results), 0)


class TestCompositeSearchProvider(unittest.TestCase):
    """Tests for CompositeSearchProvider."""

    def test_composite_combines_results(self):
        """Test that composite provider combines results."""
        from droidrun.agent.research.providers import (
            CompositeSearchProvider,
            MockSearchProvider,
        )

        # Create multiple mock providers
        providers = [
            MockSearchProvider(latency_ms=0),
            MockSearchProvider(latency_ms=0),
        ]

        composite = CompositeSearchProvider(providers)

        results = asyncio.run(composite.search("test", max_results=10))

        # Should have combined results
        self.assertGreater(len(results), 0)

    def test_composite_deduplicates(self):
        """Test that composite provider deduplicates by URL."""
        from droidrun.agent.research.providers import (
            CompositeSearchProvider,
            MockSearchProvider,
        )

        # Same provider twice would produce same URLs
        provider = MockSearchProvider(latency_ms=0)
        composite = CompositeSearchProvider([provider])

        results = asyncio.run(composite.search("dedup test", max_results=5))

        urls = [r.url for r in results]
        unique_urls = set(urls)

        self.assertEqual(len(urls), len(unique_urls))


if __name__ == "__main__":
    unittest.main()
