from pathlib import Path

from v1.tui.paths import (
    BrowseKind,
    format_dir,
    list_browse_entries,
    list_files_by_type,
)


def test_list_browse_entries_skips_hidden_dirs_and_files(tmp_path: Path):
    (tmp_path / "notes.txt").write_text("hi")
    (tmp_path / "docs").mkdir()
    (tmp_path / "alpha").mkdir()
    (tmp_path / ".secret").mkdir()

    entries = list_browse_entries(tmp_path)

    assert entries[0].kind == BrowseKind.USE_CURRENT
    assert entries[0].path == tmp_path.resolve()
    assert entries[1].kind == BrowseKind.PARENT
    assert entries[1].path == tmp_path.resolve().parent
    dirs = [entry for entry in entries if entry.kind == BrowseKind.DIR]
    assert [entry.name for entry in dirs] == ["alpha", "docs"]


def test_list_files_by_type_is_case_insensitive_and_sorted(tmp_path: Path):
    (tmp_path / "b.PDF").write_bytes(b"%PDF-1.4")
    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "skip.txt").write_text("no")
    (tmp_path / "notes.md").write_text("md")
    (tmp_path / "nested").mkdir()

    assert list_files_by_type(tmp_path, "pdf") == ["a.pdf", "b.PDF"]
    assert "skip.txt" not in list_files_by_type(tmp_path, "pdf")
    assert "notes.md" not in list_files_by_type(tmp_path, "pdf")


def test_format_dir_uses_home_tilde(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    nested = tmp_path / "papers"
    nested.mkdir()

    assert format_dir(tmp_path) == "~"
    assert format_dir(nested) == "~/papers"
    assert format_dir(Path("/tmp")).startswith("/")
