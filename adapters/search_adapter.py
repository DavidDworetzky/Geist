"""Bounded web-search adapter used by the chat tool registry."""

from __future__ import annotations

import html
import ipaddress
import json
import socket
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests

from adapters.base_adapter import BaseAdapter


class _DuckDuckGoResultParser(HTMLParser):
    def __init__(self, limit: int):
        super().__init__(convert_charrefs=True)
        self.limit = limit
        self.results: list[dict[str, str]] = []
        self._current: dict[str, str] | None = None
        self._field: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        if tag == "a" and "result__a" in classes and len(self.results) < self.limit:
            self._current = {
                "title": "",
                "url": self._unwrap_url(attributes.get("href") or ""),
                "snippet": "",
            }
            self._field = "title"
        elif self._current is not None and "result__snippet" in classes:
            self._field = "snippet"

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._current is not None and self._field == "title":
            self._field = None
        elif tag in {"a", "div"} and self._current is not None and self._field == "snippet":
            self._finish_current()

    def handle_data(self, data: str) -> None:
        if self._current is not None and self._field:
            self._current[self._field] += data

    def close(self) -> None:
        super().close()
        self._finish_current()

    def _finish_current(self) -> None:
        if self._current is None:
            return
        normalized = {key: html.unescape(value).strip() for key, value in self._current.items()}
        if normalized["title"] and normalized["url"]:
            self.results.append(normalized)
        self._current = None
        self._field = None

    @staticmethod
    def _unwrap_url(value: str) -> str:
        parsed = urlparse(value)
        redirected = parse_qs(parsed.query).get("uddg")
        return redirected[0] if redirected else value


class SearchAdapter(BaseAdapter):
    """Search public web results without exposing arbitrary URL fetching as a tool."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 15.0,
        max_response_bytes: int = 1_000_000,
        **kwargs: Any,
    ):
        self.base_url = (base_url or "https://html.duckduckgo.com/html/").rstrip("/") + "/"
        self.timeout = timeout
        self.max_response_bytes = max_response_bytes

    def enumerate_actions(self) -> list[str]:
        return ["search"]

    def search(
        self,
        search_term: str,
        max_results: int = 5,
        recency: str | None = None,
    ) -> list[dict[str, str]]:
        query = search_term.strip()
        if not query:
            raise ValueError("search_term is required")
        if len(query) > 512:
            raise ValueError("search_term must be 512 characters or fewer")
        max_results = max(1, min(max_results, 10))

        params: dict[str, str] = {"q": query}
        recency_codes = {"day": "d", "week": "w", "month": "m", "year": "y"}
        if recency in recency_codes:
            params["df"] = recency_codes[recency]

        response = requests.get(
            self.base_url,
            params=params,
            headers={"User-Agent": "Geist/1.0 (+https://github.com/DavidDworetzky/Geist)"},
            timeout=self.timeout,
            stream=True,
        )
        try:
            response.raise_for_status()
            content = self._read_bounded(response, "Search response")
            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                payload = json.loads(content.decode(response.encoding or "utf-8"))
                raw_results = (
                    payload.get("results", payload) if isinstance(payload, dict) else payload
                )
                return [
                    {
                        "title": str(item.get("title", "")),
                        "url": str(item.get("url", "")),
                        "snippet": str(item.get("snippet", "")),
                    }
                    for item in raw_results[:max_results]
                    if isinstance(item, dict) and item.get("url")
                ]

            parser = _DuckDuckGoResultParser(max_results)
            parser.feed(content.decode(response.encoding or "utf-8", errors="replace"))
            parser.close()
            return parser.results[:max_results]
        finally:
            response.close()

    def _read_bounded(self, response: requests.Response, label: str) -> bytes:
        content = bytearray()
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            remaining = self.max_response_bytes + 1 - len(content)
            content.extend(chunk[:remaining])
            if len(content) > self.max_response_bytes:
                raise RuntimeError(f"{label} exceeded the configured size limit")
        return bytes(content)

    def get(self, url: str) -> str:
        """Retained for direct adapter users, but intentionally not registered as a chat tool."""
        self._validate_public_url(url)
        response = requests.get(
            url,
            headers={"User-Agent": "Geist/1.0 (+https://github.com/DavidDworetzky/Geist)"},
            timeout=self.timeout,
            allow_redirects=False,
            stream=True,
        )
        try:
            response.raise_for_status()
            content = self._read_bounded(response, "Response")
            return content.decode(response.encoding or "utf-8", errors="replace")
        finally:
            response.close()

    @staticmethod
    def _validate_public_url(url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Only public HTTP(S) URLs are allowed")
        try:
            addresses = socket.getaddrinfo(
                parsed.hostname, parsed.port or 443, type=socket.SOCK_STREAM
            )
        except socket.gaierror as error:
            raise ValueError("URL hostname could not be resolved") from error
        for address in addresses:
            ip = ipaddress.ip_address(address[4][0])
            if not ip.is_global:
                raise ValueError("Private, local, and reserved network targets are not allowed")
