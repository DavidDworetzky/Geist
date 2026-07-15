import os

import pytest

from adapters.markdown_file_adapter import MarkdownFileAdapter


def test_read_and_write_reject_parent_and_absolute_path_escape(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("outside", encoding="utf-8")
    adapter = MarkdownFileAdapter(file_root=str(root))

    with pytest.raises(ValueError, match="escapes"):
        adapter.read_file("../outside.md")
    with pytest.raises(ValueError, match="relative"):
        adapter.read_file(str(outside))
    with pytest.raises(ValueError, match="escapes"):
        adapter.write_file("../outside.md", "changed")

    assert outside.read_text(encoding="utf-8") == "outside"


def test_read_rejects_symlink_that_resolves_outside_root(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.md").write_text("secret", encoding="utf-8")
    os.symlink(outside, root / "linked")
    adapter = MarkdownFileAdapter(file_root=str(root))

    with pytest.raises(ValueError, match="escapes"):
        adapter.read_file("linked/secret.md")


def test_list_rejects_search_roots_outside_configured_root(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.md").write_text("secret", encoding="utf-8")
    adapter = MarkdownFileAdapter(file_root=str(root))

    assert adapter.get_files("..") == []
    assert adapter.get_files(str(outside)) == []


@pytest.mark.parametrize("filename", [".env", "settings.py", "notes.txt"])
def test_read_and_write_reject_non_markdown_files(tmp_path, filename):
    adapter = MarkdownFileAdapter(file_root=str(tmp_path))
    (tmp_path / filename).write_text("secret", encoding="utf-8")

    with pytest.raises(ValueError, match=".md or .markdown"):
        adapter.read_file(filename)
    with pytest.raises(ValueError, match=".md or .markdown"):
        adapter.write_file(filename, "changed")

    assert (tmp_path / filename).read_text(encoding="utf-8") == "secret"
