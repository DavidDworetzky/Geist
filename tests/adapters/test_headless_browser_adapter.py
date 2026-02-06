import pytest
import requests
from unittest.mock import Mock, patch, PropertyMock
from adapters.headless_browser_adapter import (
    HeadlessBrowserAdapter,
    _HTMLTextExtractor,
    _HTMLLinkExtractor,
)


class TestHTMLTextExtractor:

    def test_basic_text_extraction(self):
        extractor = _HTMLTextExtractor()
        extractor.feed("<p>Hello world</p>")
        assert extractor.get_text() == "Hello world"

    def test_strips_script_and_style(self):
        html = "<p>Visible</p><script>var x = 1;</script><style>.a{color:red}</style><p>Also visible</p>"
        extractor = _HTMLTextExtractor()
        extractor.feed(html)
        text = extractor.get_text()
        assert "Visible" in text
        assert "Also visible" in text
        assert "var x" not in text
        assert "color:red" not in text

    def test_inserts_newlines_for_block_elements(self):
        html = "<h1>Title</h1><p>Paragraph</p><div>Div</div>"
        extractor = _HTMLTextExtractor()
        extractor.feed(html)
        text = extractor.get_text()
        assert "Title" in text
        assert "Paragraph" in text
        assert "Div" in text
        # Block elements should produce newlines between content
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        assert len(lines) >= 3

    def test_collapses_whitespace(self):
        html = "<p>  lots   of    spaces  </p>"
        extractor = _HTMLTextExtractor()
        extractor.feed(html)
        text = extractor.get_text()
        assert "  " not in text

    def test_nested_skip_tags(self):
        html = "<script><script>inner</script></script><p>visible</p>"
        extractor = _HTMLTextExtractor()
        extractor.feed(html)
        text = extractor.get_text()
        assert "inner" not in text
        assert "visible" in text

    def test_empty_html(self):
        extractor = _HTMLTextExtractor()
        extractor.feed("")
        assert extractor.get_text() == ""


class TestHTMLLinkExtractor:

    def test_extracts_absolute_links(self):
        html = '<a href="https://example.com">Example</a>'
        extractor = _HTMLLinkExtractor(base_url="https://base.com")
        extractor.feed(html)
        assert len(extractor.links) == 1
        assert extractor.links[0]["url"] == "https://example.com"
        assert extractor.links[0]["text"] == "Example"

    def test_resolves_relative_links(self):
        html = '<a href="/about">About</a>'
        extractor = _HTMLLinkExtractor(base_url="https://example.com/page")
        extractor.feed(html)
        assert len(extractor.links) == 1
        assert extractor.links[0]["url"] == "https://example.com/about"

    def test_multiple_links(self):
        html = '<a href="/a">A</a><a href="/b">B</a><a href="/c">C</a>'
        extractor = _HTMLLinkExtractor(base_url="https://example.com")
        extractor.feed(html)
        assert len(extractor.links) == 3
        texts = [link["text"] for link in extractor.links]
        assert texts == ["A", "B", "C"]

    def test_skips_anchors_without_href(self):
        html = '<a name="top">No href</a><a href="/real">Real</a>'
        extractor = _HTMLLinkExtractor(base_url="https://example.com")
        extractor.feed(html)
        assert len(extractor.links) == 1
        assert extractor.links[0]["text"] == "Real"

    def test_empty_html(self):
        extractor = _HTMLLinkExtractor(base_url="https://example.com")
        extractor.feed("")
        assert extractor.links == []


class TestHeadlessBrowserAdapter:

    def setup_method(self):
        self.adapter = HeadlessBrowserAdapter(timeout=10)

    def test_enumerate_actions(self):
        actions = self.adapter.enumerate_actions()
        assert "browse" in actions
        assert "get_links" in actions
        assert "get_page_source" in actions

    def test_init_defaults(self):
        adapter = HeadlessBrowserAdapter()
        assert adapter.timeout == 30
        assert "User-Agent" in adapter.session.headers

    def test_init_custom_headers(self):
        adapter = HeadlessBrowserAdapter(headers={"X-Custom": "value"})
        assert adapter.session.headers["X-Custom"] == "value"

    @patch("adapters.headless_browser_adapter.requests.Session")
    def test_init_max_redirects(self, mock_session_cls):
        mock_session = Mock()
        mock_session.headers = {}
        mock_session_cls.return_value = mock_session
        adapter = HeadlessBrowserAdapter(max_redirects=10)
        assert mock_session.max_redirects == 10

    @patch.object(HeadlessBrowserAdapter, "_fetch")
    def test_browse_html(self, mock_fetch):
        mock_response = Mock()
        mock_response.headers = {"Content-Type": "text/html; charset=utf-8"}
        mock_response.text = "<html><body><h1>Hello</h1><p>World</p></body></html>"
        mock_fetch.return_value = mock_response

        result = self.adapter.browse("https://example.com")
        assert "Hello" in result
        assert "World" in result

    @patch.object(HeadlessBrowserAdapter, "_fetch")
    def test_browse_non_html(self, mock_fetch):
        mock_response = Mock()
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.text = '{"key": "value"}'
        mock_fetch.return_value = mock_response

        result = self.adapter.browse("https://example.com/api")
        assert result == '{"key": "value"}'

    @patch.object(HeadlessBrowserAdapter, "_fetch")
    def test_browse_request_failure(self, mock_fetch):
        mock_fetch.side_effect = requests.ConnectionError("Connection refused")

        result = self.adapter.browse("https://unreachable.example.com")
        assert "Failed to retrieve content" in result

    @patch.object(HeadlessBrowserAdapter, "_fetch")
    def test_get_links(self, mock_fetch):
        mock_response = Mock()
        mock_response.text = (
            '<html><body>'
            '<a href="https://example.com/a">Link A</a>'
            '<a href="/b">Link B</a>'
            '</body></html>'
        )
        mock_fetch.return_value = mock_response

        links = self.adapter.get_links("https://example.com")
        assert len(links) == 2
        assert links[0]["url"] == "https://example.com/a"
        assert links[0]["text"] == "Link A"
        assert links[1]["url"] == "https://example.com/b"
        assert links[1]["text"] == "Link B"

    @patch.object(HeadlessBrowserAdapter, "_fetch")
    def test_get_links_failure(self, mock_fetch):
        mock_fetch.side_effect = requests.Timeout("Timeout")

        links = self.adapter.get_links("https://example.com")
        assert links == []

    @patch.object(HeadlessBrowserAdapter, "_fetch")
    def test_get_page_source(self, mock_fetch):
        html = "<html><body><p>Raw source</p></body></html>"
        mock_response = Mock()
        mock_response.text = html
        mock_fetch.return_value = mock_response

        result = self.adapter.get_page_source("https://example.com")
        assert result == html

    @patch.object(HeadlessBrowserAdapter, "_fetch")
    def test_get_page_source_failure(self, mock_fetch):
        mock_fetch.side_effect = requests.HTTPError("500 Server Error")

        result = self.adapter.get_page_source("https://example.com")
        assert "Failed to retrieve page source" in result

    @patch.object(HeadlessBrowserAdapter, "_fetch")
    def test_browse_strips_script_content(self, mock_fetch):
        mock_response = Mock()
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.text = (
            "<html><head><script>alert('xss')</script></head>"
            "<body><p>Safe content</p></body></html>"
        )
        mock_fetch.return_value = mock_response

        result = self.adapter.browse("https://example.com")
        assert "Safe content" in result
        assert "alert" not in result

    def test_fetch_calls_session_get(self):
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        self.adapter.session.get = Mock(return_value=mock_response)

        result = self.adapter._fetch("https://example.com")
        self.adapter.session.get.assert_called_once_with(
            "https://example.com", timeout=10
        )
        mock_response.raise_for_status.assert_called_once()
        assert result == mock_response
