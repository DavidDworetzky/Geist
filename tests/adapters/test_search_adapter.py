import json
from unittest.mock import Mock, patch

import pytest

from adapters.search_adapter import SearchAdapter


def _response(*, content: bytes, content_type: str, payload=None) -> Mock:
    response = Mock()
    response.content = content
    response.iter_content.return_value = [content]
    response.headers = {"content-type": content_type}
    response.encoding = "utf-8"
    response.json.return_value = payload
    return response


@patch("adapters.search_adapter.requests.get")
def test_search_parses_json_results_and_applies_query_bounds(mock_get):
    payload = {
        "results": [
            {
                "title": f"Result {index}",
                "url": f"https://example.com/{index}",
                "snippet": f"Snippet {index}",
            }
            for index in range(12)
        ]
    }
    mock_get.return_value = _response(
        content=json.dumps(payload).encode("utf-8"),
        content_type="application/json; charset=utf-8",
        payload=payload,
    )
    adapter = SearchAdapter(timeout=3.5)

    results = adapter.search("  bounded query  ", max_results=99, recency="week")

    assert len(results) == 10
    assert results[0] == {
        "title": "Result 0",
        "url": "https://example.com/0",
        "snippet": "Snippet 0",
    }
    mock_get.return_value.raise_for_status.assert_called_once_with()
    request_kwargs = mock_get.call_args.kwargs
    assert request_kwargs["params"] == {"q": "bounded query", "df": "w"}
    assert request_kwargs["timeout"] == 3.5
    assert request_kwargs["stream"] is True
    mock_get.return_value.close.assert_called_once_with()


@patch("adapters.search_adapter.requests.get")
def test_search_parses_duckduckgo_html_and_unwraps_result_url(mock_get):
    body = b"""
        <div class="result">
          <a class="result__a" href="/l/?uddg=https%3A%2F%2Fexample.com%2Farticle">
            First &amp; best result
          </a>
          <a class="result__snippet">A useful <strong>search</strong> snippet.</a>
        </div>
        <div class="result">
          <a class="result__a" href="https://second.example/path">Second result</a>
          <div class="result__snippet">Second snippet.</div>
        </div>
    """
    mock_get.return_value = _response(content=body, content_type="text/html")
    adapter = SearchAdapter()

    results = adapter.search("html query", max_results=1)

    assert results == [
        {
            "title": "First & best result",
            "url": "https://example.com/article",
            "snippet": "A useful search snippet.",
        }
    ]


@patch("adapters.search_adapter.requests.get")
def test_search_rejects_response_over_configured_byte_limit(mock_get):
    mock_get.return_value = _response(content=b"12345", content_type="text/html")
    adapter = SearchAdapter(max_response_bytes=4)

    with pytest.raises(RuntimeError, match="size limit"):
        adapter.search("query")


@pytest.mark.parametrize("query", ["", "   ", "x" * 513])
@patch("adapters.search_adapter.requests.get")
def test_search_rejects_invalid_query_before_network_call(mock_get, query):
    adapter = SearchAdapter()

    with pytest.raises(ValueError):
        adapter.search(query)

    mock_get.assert_not_called()
