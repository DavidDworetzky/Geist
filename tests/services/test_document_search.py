import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.services.document_search import DocumentSearchService


def _uploaded_file(**overrides):
    values = {
        "file_id": 1,
        "filename": "notes.md",
        "original_filename": "Alpha Notes.md",
        "extracted_text": "Project Alpha launch notes",
        "mime_type": "text/markdown",
        "file_size": 128,
        "upload_date": datetime.datetime(2026, 7, 14, 12, 0, 0),
        "is_processed": True,
        "user_id": 42,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@patch("app.services.document_search.SessionLocal")
def test_search_filters_database_query_by_requesting_user(mock_session_local):
    session = MagicMock()
    mock_session_local.return_value.__enter__.return_value = session
    query = session.query.return_value
    filtered_query = query.filter.return_value
    ordered_query = filtered_query.order_by.return_value
    limited_query = ordered_query.limit.return_value
    limited_query.all.return_value = [
        _uploaded_file(),
        _uploaded_file(
            file_id=2,
            filename="unrelated.md",
            original_filename="Unrelated.md",
            extracted_text="Different project",
        ),
    ]

    results = DocumentSearchService.search(user_id=42, query="alpha", limit=10)

    selected_columns = session.query.call_args.args
    assert {column.key for column in selected_columns} == {
        "file_id",
        "filename",
        "original_filename",
        "extracted_text",
        "mime_type",
        "file_size",
        "upload_date",
        "is_processed",
    }
    user_predicate = query.filter.call_args.args[0]
    assert user_predicate.left.key == "user_id"
    assert user_predicate.right.value == 42
    ordered_query.limit.assert_called_once_with(100)
    assert [result["file_id"] for result in results] == [1]
    assert results[0]["match_type"] == ["original_filename", "content"]
