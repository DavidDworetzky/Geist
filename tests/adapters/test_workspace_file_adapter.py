import os

import pytest

from adapters.workspace_file_adapter import WorkspaceFileAdapter


def test_workspace_file_lifecycle_and_search(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text(
        "first line\nneedle = True\nlast line\n",
        encoding="utf-8",
    )
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "ignored.js").write_text("needle", encoding="utf-8")
    (tmp_path / ".env").write_text("SECRET=value", encoding="utf-8")
    adapter = WorkspaceFileAdapter(str(tmp_path))

    assert adapter.list_files(pattern="*.py") == ["src/app.py"]
    excerpt = adapter.read_file("src/app.py", start_line=2, end_line=2)
    assert excerpt["content"] == "needle = True\n"
    assert excerpt["start_line"] == 2
    assert excerpt["end_line"] == 2

    matches = adapter.search_text("needle", pattern="*.py")
    assert matches == [
        {
            "path": "src/app.py",
            "line": 2,
            "column": 1,
            "text": "needle = True",
        }
    ]

    written = adapter.write_file("src/new.ts", "export const value = 1;\n")
    assert written == {"path": "src/new.ts", "characters_written": 24}
    edited = adapter.edit_file("src/new.ts", "value = 1", "value = 2")
    assert edited["replacements"] == 1
    assert (tmp_path / "src" / "new.ts").read_text(encoding="utf-8").endswith("value = 2;\n")


def test_edit_fails_closed_when_match_count_differs(tmp_path):
    target = tmp_path / "app.py"
    target.write_text("same\nsame\n", encoding="utf-8")
    adapter = WorkspaceFileAdapter(str(tmp_path))

    with pytest.raises(ValueError, match="Expected 1 matching block"):
        adapter.edit_file("app.py", "same", "changed")

    assert target.read_text(encoding="utf-8") == "same\nsame\n"


def test_workspace_paths_cannot_escape_or_read_credentials(tmp_path):
    outside = tmp_path.parent / "outside.py"
    outside.write_text("outside", encoding="utf-8")
    (tmp_path / ".env").write_text("SECRET=value", encoding="utf-8")
    adapter = WorkspaceFileAdapter(str(tmp_path))

    with pytest.raises(ValueError, match="escapes"):
        adapter.read_file("../outside.py")
    with pytest.raises(ValueError, match="credential"):
        adapter.read_file(".env")

    link = tmp_path / "linked.py"
    try:
        os.symlink(outside, link)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")
    with pytest.raises(ValueError, match="escapes"):
        adapter.read_file("linked.py")
