import logging
import re
from html.parser import HTMLParser
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse

import requests

from adapters.base_adapter import BaseAdapter

logger = logging.getLogger(__name__)


class _HTMLTextExtractor(HTMLParser):
    """
    Extracts visible text from HTML, stripping tags and non-visible elements.
    """

    _SKIP_TAGS = {"script", "style", "head", "meta", "link", "noscript"}

    def __init__(self):
        super().__init__()
        self._pieces: List[str] = []
        self._skip_depth: int = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag.lower() in self._SKIP_TAGS:
            self._skip_depth += 1
        if tag.lower() in ("br", "p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr"):
            self._pieces.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        if tag.lower() in ("p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "table"):
            self._pieces.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._pieces.append(data)

    def get_text(self) -> str:
        raw = "".join(self._pieces)
        # Collapse multiple whitespace/newlines into readable form
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


class _HTMLLinkExtractor(HTMLParser):
    """
    Extracts href links and their anchor text from HTML.
    """

    def __init__(self, base_url: str):
        super().__init__()
        self._base_url = base_url
        self.links: List[Dict[str, str]] = []
        self._current_href: Optional[str] = None
        self._current_text_parts: List[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag.lower() == "a":
            attrs_dict = dict(attrs)
            href = attrs_dict.get("href")
            if href:
                self._current_href = urljoin(self._base_url, href)
                self._current_text_parts = []

    def handle_data(self, data: str) -> None:
        if self._current_href is not None:
            self._current_text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._current_href is not None:
            text = " ".join(self._current_text_parts).strip()
            self.links.append({"url": self._current_href, "text": text})
            self._current_href = None
            self._current_text_parts = []


class HeadlessBrowserAdapter(BaseAdapter):
    """
    Headless Browser Adapter for fetching and parsing web pages.
    Provides text extraction, link extraction, and raw HTML retrieval
    using requests and stdlib HTML parsing.
    """

    _DEFAULT_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; Geist/1.0; +https://github.com/DavidDworworetzky/Geist)"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    def __init__(
        self,
        timeout: int = 30,
        max_redirects: int = 5,
        headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        """
        Initialize the headless browser adapter.

        Args:
            timeout: Request timeout in seconds.
            max_redirects: Maximum number of HTTP redirects to follow.
            headers: Optional custom headers to merge with defaults.
        """
        self.timeout = timeout
        self.session = requests.Session()
        self.session.max_redirects = max_redirects
        self.session.headers.update(self._DEFAULT_HEADERS)
        if headers:
            self.session.headers.update(headers)

    def enumerate_actions(self) -> List[str]:
        return ["browse", "get_links", "get_page_source"]

    def _fetch(self, url: str) -> requests.Response:
        """
        Perform an HTTP GET request.

        Args:
            url: The URL to fetch.

        Returns:
            The HTTP response.

        Raises:
            requests.RequestException: On network or HTTP errors.
        """
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response

    def browse(self, url: str) -> str:
        """
        Fetch a URL and return the visible text content of the page.

        Args:
            url: The URL to browse.

        Returns:
            Extracted visible text, or an error message on failure.
        """
        try:
            response = self._fetch(url)
            content_type = response.headers.get("Content-Type", "")
            if "text/html" not in content_type and "application/xhtml" not in content_type:
                # Return raw text for non-HTML content
                return response.text

            extractor = _HTMLTextExtractor()
            extractor.feed(response.text)
            return extractor.get_text()
        except requests.RequestException as e:
            logger.warning("Failed to browse %s: %s", url, e)
            return f"Failed to retrieve content from {url}: {e}"

    def get_links(self, url: str) -> List[Dict[str, str]]:
        """
        Fetch a URL and return all hyperlinks found on the page.

        Args:
            url: The URL to extract links from.

        Returns:
            List of dicts with 'url' and 'text' keys, or empty list on failure.
        """
        try:
            response = self._fetch(url)
            extractor = _HTMLLinkExtractor(base_url=url)
            extractor.feed(response.text)
            return extractor.links
        except requests.RequestException as e:
            logger.warning("Failed to get links from %s: %s", url, e)
            return []

    def get_page_source(self, url: str) -> str:
        """
        Fetch a URL and return the raw HTML source.

        Args:
            url: The URL to fetch.

        Returns:
            Raw HTML string, or an error message on failure.
        """
        try:
            response = self._fetch(url)
            return response.text
        except requests.RequestException as e:
            logger.warning("Failed to get page source from %s: %s", url, e)
            return f"Failed to retrieve page source from {url}: {e}"
