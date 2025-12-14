"""
Browser Integration for DroidRun.

Provides specialized actions for web browsers:
- Navigate to URLs
- Search the web
- Extract page content
- Fill forms and click elements
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

try:
    from . import register_app_handler
except ImportError:
    # Standalone testing - create dummy decorator
    def register_app_handler(package_name):
        def decorator(cls):
            return cls
        return decorator

logger = logging.getLogger("droidrun.apps.browser")

# Common browser packages
CHROME_PACKAGE = "com.android.chrome"
FIREFOX_PACKAGE = "org.mozilla.firefox"
EDGE_PACKAGE = "com.microsoft.emmx"
BRAVE_PACKAGE = "com.brave.browser"
SAMSUNG_BROWSER = "com.sec.android.app.sbrowser"

BROWSER_PACKAGES = [
    CHROME_PACKAGE,
    FIREFOX_PACKAGE,
    EDGE_PACKAGE,
    BRAVE_PACKAGE,
    SAMSUNG_BROWSER,
]


@dataclass
class PageContent:
    """Content extracted from a web page."""
    url: str
    title: str
    text_content: str
    links: List[Dict[str, str]]
    forms: List[Dict[str, Any]]
    success: bool = True
    error: Optional[str] = None


@dataclass
class SearchResult:
    """Result from a web search."""
    query: str
    results: List[Dict[str, str]]  # title, url, snippet
    success: bool = True
    error: Optional[str] = None


@register_app_handler(CHROME_PACKAGE)
class BrowserHandler:
    """
    Handler for web browsers (Chrome, Firefox, etc).

    Provides high-level actions for web navigation,
    searching, and content extraction.
    """

    def __init__(
        self,
        tools_instance: Any = None,
        preferred_browser: str = CHROME_PACKAGE,
    ):
        """
        Initialize browser handler.

        Args:
            tools_instance: AdbTools instance for device interaction
            preferred_browser: Package name of preferred browser
        """
        self.tools = tools_instance
        self.preferred_browser = preferred_browser
        self._current_url = ""
        self._navigation_history: List[str] = []

    async def ensure_browser_open(self) -> bool:
        """
        Ensure browser is open and ready.

        Returns:
            True if browser is ready, False otherwise
        """
        if not self.tools:
            logger.warning("No tools instance available")
            return False

        try:
            # Launch preferred browser
            await self.tools.launch_app(self.preferred_browser)
            await asyncio.sleep(1.5)  # Wait for browser to load

            # Verify we're in a browser
            ui_state = await self.tools.get_ui_state()
            current_package = ui_state.get("package_name", "")

            if current_package in BROWSER_PACKAGES:
                logger.info(f"✅ Browser ready: {current_package}")
                return True
            else:
                logger.warning(f"Not in browser, current package: {current_package}")
                return False

        except Exception as e:
            logger.error(f"Failed to open browser: {e}")
            return False

    async def navigate_to(self, url: str) -> bool:
        """
        Navigate to a URL.

        Args:
            url: URL to navigate to

        Returns:
            True if navigation successful
        """
        if not self.tools:
            return False

        try:
            # Ensure URL has protocol
            if not url.startswith(("http://", "https://")):
                url = f"https://{url}"

            # Use ADB to open URL directly
            await self.tools.device.shell(
                f'am start -a android.intent.action.VIEW -d "{url}"'
            )

            await asyncio.sleep(2.0)  # Wait for page to load

            self._current_url = url
            self._navigation_history.append(url)

            logger.info(f"Navigated to: {url}")
            return True

        except Exception as e:
            logger.error(f"Navigation failed: {e}")
            return False

    async def search_google(self, query: str) -> SearchResult:
        """
        Perform a Google search.

        Args:
            query: Search query

        Returns:
            SearchResult with found results
        """
        result = SearchResult(query=query, results=[])

        try:
            # Navigate to Google search
            search_url = f"https://www.google.com/search?q={quote_plus(query)}"
            if not await self.navigate_to(search_url):
                result.success = False
                result.error = "Failed to navigate to search"
                return result

            # Wait for results to load
            await asyncio.sleep(2.0)

            # Extract search results from page
            result.results = await self._extract_search_results()
            result.success = len(result.results) > 0

            logger.info(f"Search '{query}' found {len(result.results)} results")
            return result

        except Exception as e:
            result.success = False
            result.error = str(e)
            return result

    async def _extract_search_results(self) -> List[Dict[str, str]]:
        """Extract search results from current page."""
        results = []

        if not self.tools:
            return results

        try:
            # Get accessibility tree
            ui_state = await self.tools.get_ui_state()
            a11y_tree = ui_state.get("a11y_tree", [])

            # Look for clickable items with text that look like search results
            def find_results(node, results):
                if isinstance(node, dict):
                    text = node.get("text", "")
                    clickable = node.get("clickable", False)
                    content_desc = node.get("content-desc", "")

                    # Search results usually have a title and are clickable
                    if clickable and text and len(text) > 20:
                        results.append({
                            "title": text[:100],
                            "url": content_desc if "http" in content_desc else "",
                            "snippet": text[:200],
                        })

                    for child in node.get("children", []):
                        find_results(child, results)
                elif isinstance(node, list):
                    for item in node:
                        find_results(item, results)

            find_results(a11y_tree, results)

            # Limit to top 10 results
            return results[:10]

        except Exception as e:
            logger.error(f"Failed to extract search results: {e}")
            return []

    async def extract_page_content(self) -> PageContent:
        """
        Extract content from current page.

        Returns:
            PageContent with extracted information
        """
        content = PageContent(
            url=self._current_url,
            title="",
            text_content="",
            links=[],
            forms=[],
        )

        if not self.tools:
            content.success = False
            content.error = "No tools instance"
            return content

        try:
            ui_state = await self.tools.get_ui_state()
            a11y_tree = ui_state.get("a11y_tree", [])

            text_parts = []
            links = []
            forms = []

            def extract_content(node):
                if isinstance(node, dict):
                    text = node.get("text", "")
                    clickable = node.get("clickable", False)
                    content_desc = node.get("content-desc", "")
                    class_name = node.get("class", "")

                    if text:
                        text_parts.append(text)

                    # Detect links
                    if clickable and ("http" in content_desc or "link" in class_name.lower()):
                        links.append({
                            "text": text or content_desc,
                            "url": content_desc if "http" in content_desc else "",
                        })

                    # Detect form elements
                    if "EditText" in class_name or "input" in class_name.lower():
                        forms.append({
                            "type": "input",
                            "hint": node.get("hint", ""),
                            "value": text,
                        })

                    for child in node.get("children", []):
                        extract_content(child)
                elif isinstance(node, list):
                    for item in node:
                        extract_content(item)

            extract_content(a11y_tree)

            content.text_content = "\n".join(text_parts)
            content.links = links[:50]  # Limit
            content.forms = forms[:20]  # Limit

            # Try to find title
            if text_parts:
                content.title = text_parts[0][:100]

            logger.info(f"Extracted content: {len(text_parts)} text elements, {len(links)} links")
            return content

        except Exception as e:
            content.success = False
            content.error = str(e)
            return content

    async def click_element(self, text: str) -> bool:
        """
        Click an element containing specific text.

        Args:
            text: Text to find and click

        Returns:
            True if clicked successfully
        """
        if not self.tools:
            return False

        try:
            # Find element by text
            ui_state = await self.tools.get_ui_state()
            a11y_tree = ui_state.get("a11y_tree", [])

            bounds = None

            def find_element(node):
                nonlocal bounds
                if isinstance(node, dict):
                    node_text = node.get("text", "") or node.get("content-desc", "")
                    if text.lower() in node_text.lower():
                        bounds = node.get("bounds")
                        return True
                    for child in node.get("children", []):
                        if find_element(child):
                            return True
                elif isinstance(node, list):
                    for item in node:
                        if find_element(item):
                            return True
                return False

            find_element(a11y_tree)

            if bounds:
                # Calculate center of bounds and tap
                # bounds format: "[x1,y1][x2,y2]"
                match = re.match(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", bounds)
                if match:
                    x1, y1, x2, y2 = map(int, match.groups())
                    x = (x1 + x2) // 2
                    y = (y1 + y2) // 2
                    await self.tools.device.click(x, y)
                    logger.info(f"Clicked element: '{text}' at ({x}, {y})")
                    return True

            logger.warning(f"Element not found: '{text}'")
            return False

        except Exception as e:
            logger.error(f"Click failed: {e}")
            return False

    async def fill_input(self, hint_or_text: str, value: str) -> bool:
        """
        Fill an input field.

        Args:
            hint_or_text: Hint text or label to identify the input
            value: Value to enter

        Returns:
            True if filled successfully
        """
        if not self.tools:
            return False

        try:
            # First click the input
            if await self.click_element(hint_or_text):
                await asyncio.sleep(0.3)
                # Type the value
                await self.tools.input_text(value)
                logger.info(f"Filled input '{hint_or_text}' with value")
                return True

            return False

        except Exception as e:
            logger.error(f"Fill input failed: {e}")
            return False

    async def go_back(self) -> bool:
        """Press back button."""
        if not self.tools:
            return False

        try:
            await self.tools.press_key("BACK")
            return True
        except Exception as e:
            logger.error(f"Go back failed: {e}")
            return False

    async def refresh_page(self) -> bool:
        """Refresh the current page."""
        if not self.tools:
            return False

        try:
            # Swipe down from top to refresh
            await self.tools.swipe(500, 200, 500, 800)
            await asyncio.sleep(1.0)
            return True
        except Exception as e:
            logger.error(f"Refresh failed: {e}")
            return False

    def get_current_url(self) -> str:
        """Get the current URL."""
        return self._current_url

    def get_navigation_history(self) -> List[str]:
        """Get navigation history."""
        return self._navigation_history.copy()


# Register additional browsers
@register_app_handler(FIREFOX_PACKAGE)
class FirefoxHandler(BrowserHandler):
    """Firefox-specific handler."""

    def __init__(self, tools_instance: Any = None):
        super().__init__(tools_instance, preferred_browser=FIREFOX_PACKAGE)


@register_app_handler(EDGE_PACKAGE)
class EdgeHandler(BrowserHandler):
    """Edge-specific handler."""

    def __init__(self, tools_instance: Any = None):
        super().__init__(tools_instance, preferred_browser=EDGE_PACKAGE)


# Convenience function
def create_browser_handler(
    tools_instance: Any = None,
    browser: str = "chrome",
) -> BrowserHandler:
    """
    Create a browser handler instance.

    Args:
        tools_instance: AdbTools instance
        browser: Browser name (chrome, firefox, edge, brave)

    Returns:
        BrowserHandler instance
    """
    browser_map = {
        "chrome": CHROME_PACKAGE,
        "firefox": FIREFOX_PACKAGE,
        "edge": EDGE_PACKAGE,
        "brave": BRAVE_PACKAGE,
        "samsung": SAMSUNG_BROWSER,
    }

    package = browser_map.get(browser.lower(), CHROME_PACKAGE)
    return BrowserHandler(tools_instance=tools_instance, preferred_browser=package)
