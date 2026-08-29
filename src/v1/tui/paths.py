"""Folder navigation and ingestible-file listing for the ingest picker."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

FILE_TYPES = ("txt", "md")


class BrowseKind(StrEnum):
    USE_CURRENT = "use_current"
    PARENT = "parent"
    DIR = "dir"


@dataclass(frozen=True)
class BrowseEntry:
    kind: BrowseKind
    name: str
    path: Path


def format_dir(path: Path) -> str:
    try:
        resolved = path.expanduser().resolve()
    except OSError:
        resolved = path
    home = Path.home()
    try:
        relative = resolved.relative_to(home)
    except ValueError:
        return str(resolved)
    return "~" if str(relative) == "." else f"~/{relative}"


def list_browse_entries(directory: Path) -> list[BrowseEntry]:
    directory = directory.resolve()
    entries = [BrowseEntry(BrowseKind.USE_CURRENT, "use this folder", directory)]
    parent = directory.parent
    if parent != directory:
        entries.append(BrowseEntry(BrowseKind.PARENT, "..", parent))

    children = sorted(
        (
            item
            for item in directory.iterdir()
            if item.is_dir() and not item.name.startswith(".")
        ),
        key=lambda item: item.name.lower(),
    )
    entries.extend(
        BrowseEntry(BrowseKind.DIR, item.name, item) for item in children
    )
    return entries


def list_files_by_type(directory: Path, file_type: str) -> list[str]:
    want = f".{file_type.lower()}"
    names = [
        item.name
        for item in directory.resolve().iterdir()
        if item.is_file() and item.suffix.lower() == want
    ]
    names.sort(key=str.lower)
    return names
