"""Minimal LazyDocs terminal: pick ingest or chat, then that path's model."""

from __future__ import annotations

import os
import sys
import termios
import textwrap
import tty
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from rich.align import Align
from rich.box import ROUNDED
from rich.console import Console, Group
from rich.padding import Padding
from rich.panel import Panel
from rich.text import Text

from v1.tui.logo import (
    CYAN,
    DARK,
    GREEN,
    LIME,
    MUTED,
    PANEL,
    RED,
    SELECT,
    TEAL,
    render_title,
)
from v1.tui.models import (
    LocalModel,
    ModelListError,
    format_size,
    index_of,
    list_local_models,
    models_for_role,
)
from v1.tui.paths import (
    FILE_TYPE,
    BrowseEntry,
    BrowseKind,
    format_dir,
    list_browse_entries,
    list_files_by_type,
)

OLLAMA_DOCS = "https://docs.ollama.com/"


class Step(StrEnum):
    HOME = "home"
    LOADING = "loading"
    PICK_MODEL = "pick_model"
    PICK_DIR = "pick_dir"
    PICK_FILE = "pick_file"
    READY = "ready"
    INGESTING = "ingesting"
    OPENING_CHAT = "opening_chat"
    CHAT = "chat"
    THINKING = "thinking"
    ERROR = "error"
    QUIT = "quit"


class Action(StrEnum):
    INGEST = "ingest"
    CHAT = "chat"


@dataclass(frozen=True)
class ActionOption:
    action: Action
    title: str
    hint: str


ACTION_OPTIONS = (
    ActionOption(
        Action.INGEST,
        "Ingest",
        "Chunk files, embed them, and store the index.",
    ),
    ActionOption(
        Action.CHAT,
        "Chat",
        "Ask questions over documents you already ingested.",
    ),
)


@dataclass(frozen=True)
class Selection:
    action: Action
    model: str
    path: str | None = None


@dataclass(frozen=True)
class Notice:
    ok: bool
    message: str


class ChatSession(Protocol):
    """A live agent bound to one chat model, owned by the caller of `run`."""

    def ask(self, question: str) -> str: ...

    def close(self) -> None: ...


class Speaker(StrEnum):
    USER = "you"
    AGENT = "lazydocs"
    FAILURE = "failed"


@dataclass(frozen=True)
class Turn:
    speaker: Speaker
    text: str


@dataclass
class AppState:
    step: Step = Step.HOME
    action: Action | None = None
    action_cursor: int = 0
    models: list[LocalModel] = field(default_factory=list)
    model_items: list[LocalModel] = field(default_factory=list)
    model_cursor: int = 0
    error: str = ""
    error_kind: str = ""
    preferred_embed: str = ""
    preferred_chat: str = ""
    list_models: Callable[[], list[LocalModel]] = list_local_models
    browse_dir: Path = field(default_factory=Path.cwd)
    browse_entries: list[BrowseEntry] = field(default_factory=list)
    browse_cursor: int = 0
    browse_err: str = ""
    selected_type: str = FILE_TYPE
    files: list[str] = field(default_factory=list)
    files_cursor: int = 0
    files_err: str = ""
    selected_file: str = ""
    list_entries: Callable[[Path], list[BrowseEntry]] = list_browse_entries
    list_files: Callable[[Path, str], list[str]] = list_files_by_type
    ingest: Callable[[str, str], None] | None = None
    open_chat: Callable[[str], ChatSession] | None = None
    session: ChatSession | None = None
    turns: list[Turn] = field(default_factory=list)
    draft: str = ""
    pending: str = ""
    chat_scroll: int = 0
    notice: Notice | None = None

    def current_option(self) -> ActionOption:
        return ACTION_OPTIONS[self.action_cursor]

    def selected_model(self) -> LocalModel | None:
        if not self.model_items:
            return None
        return self.model_items[self.model_cursor]

    def feature_title(self) -> str:
        if self.action == Action.INGEST:
            return "Ingest"
        if self.action == Action.CHAT:
            return "Chat"
        return self.current_option().title

    def load(self) -> None:
        try:
            self.models = self.list_models()
        except ModelListError as exc:
            self.models = []
            self.model_items = []
            self.error = str(exc)
            self.error_kind = exc.kind
            self.step = Step.ERROR
            return

        self.error = ""
        self.error_kind = ""
        self._prepare_picker()
        self.step = Step.PICK_MODEL

    def open_action(self) -> None:
        self.action = self.current_option().action
        if self.models:
            self._prepare_picker()
            self.step = Step.PICK_MODEL
            return
        self.step = Step.LOADING

    def handle(self, key: str) -> None:
        # Chat comes first: every printable key belongs to the draft, so the
        # single-letter shortcuts must not swallow it.
        if self.step == Step.CHAT:
            self._handle_chat(key)
            return

        if key in {"q", "ctrl+c"}:
            self.step = Step.QUIT
            return

        if self.step == Step.HOME:
            if key in {"up", "k"}:
                self.action_cursor = (self.action_cursor - 1) % len(ACTION_OPTIONS)
            elif key in {"down", "j"}:
                self.action_cursor = (self.action_cursor + 1) % len(ACTION_OPTIONS)
            elif key == "enter":
                self.open_action()
            elif key == "esc":
                self.step = Step.QUIT
            return

        if self.step == Step.ERROR:
            if key in {"r", "enter"}:
                self.step = Step.LOADING
            elif key == "esc":
                self.step = Step.HOME
            return

        if key == "esc":
            if self.step == Step.PICK_MODEL:
                self.step = Step.HOME
            elif self.step == Step.PICK_DIR:
                self.step = Step.PICK_MODEL
            elif self.step == Step.PICK_FILE:
                self.step = Step.PICK_DIR
            elif self.step == Step.READY:
                self.step = (
                    Step.PICK_FILE if self.action == Action.INGEST else Step.PICK_MODEL
                )
            return

        if self.step == Step.PICK_MODEL:
            if not self.model_items:
                return
            if key in {"up", "k"}:
                self._move(-1)
            elif key in {"down", "j"}:
                self._move(1)
            elif key == "enter":
                if self.action == Action.INGEST:
                    self._load_browse(self.browse_dir)
                    self.step = Step.PICK_DIR
                else:
                    self.step = Step.READY
            return

        if self.step == Step.PICK_DIR:
            self._handle_browse(key)
            return

        if self.step == Step.PICK_FILE:
            self._handle_files(key)
            return

        if self.step == Step.READY and key == "enter":
            if self.action == Action.INGEST:
                self.step = Step.INGESTING
                return
            self.step = Step.OPENING_CHAT

    def _prepare_picker(self) -> None:
        role = "embed" if self.action == Action.INGEST else "chat"
        preferred = (
            self.preferred_embed
            if self.action == Action.INGEST
            else self.preferred_chat
        )
        self.model_items = models_for_role(self.models, role)
        self.model_cursor = index_of(self.model_items, preferred)

    def _move(self, delta: int) -> None:
        if not self.model_items:
            return
        self.model_cursor = (self.model_cursor + delta) % len(self.model_items)

    def _nudge(self, attr: str, count: int, delta: int) -> None:
        if count <= 0:
            return
        current = getattr(self, attr)
        setattr(self, attr, max(0, min(count - 1, current + delta)))

    def _load_browse(self, directory: Path) -> None:
        self.browse_cursor = 0
        try:
            directory = directory.resolve()
            self.browse_dir = directory
            self.browse_entries = self.list_entries(directory)
            self.browse_err = ""
        except OSError as exc:
            self.browse_dir = directory
            self.browse_entries = []
            self.browse_err = str(exc)

    def _load_files(self) -> None:
        self.files_cursor = 0
        try:
            self.files = self.list_files(self.browse_dir, self.selected_type)
            self.files_err = ""
        except OSError as exc:
            self.files = []
            self.files_err = str(exc)

    def _handle_browse(self, key: str) -> None:
        if key in {"up", "k"}:
            self._nudge("browse_cursor", len(self.browse_entries), -1)
            return
        if key in {"down", "j"}:
            self._nudge("browse_cursor", len(self.browse_entries), 1)
            return
        if key != "enter" or not self.browse_entries:
            return
        entry = self.browse_entries[self.browse_cursor]
        if entry.kind == BrowseKind.USE_CURRENT:
            self.selected_type = FILE_TYPE
            self._load_files()
            self.step = Step.PICK_FILE
            return
        self._load_browse(entry.path)

    def _handle_files(self, key: str) -> None:
        if key in {"up", "k"}:
            self._nudge("files_cursor", len(self.files), -1)
            return
        if key in {"down", "j"}:
            self._nudge("files_cursor", len(self.files), 1)
            return
        if key != "enter" or self.files_err or not self.files:
            return
        self.selected_file = str(
            (self.browse_dir / self.files[self.files_cursor]).resolve()
        )
        self.step = Step.READY

    def selection(self) -> Selection | None:
        model = self.selected_model()
        if self.action is None or model is None:
            return None
        if self.action == Action.INGEST:
            if not self.selected_file:
                return None
            return Selection(
                action=self.action,
                model=model.name,
                path=self.selected_file,
            )
        return Selection(action=self.action, model=model.name)

    def finish_ingest(self) -> None:
        selection = self.selection()
        if selection is None or not selection.path:
            self._go_home(ok=False, message="Nothing to ingest.")
            return
        if self.ingest is None:
            self._go_home(ok=False, message="Ingest is not configured.")
            return
        try:
            self.ingest(selection.path, selection.model)
        except Exception as exc:
            self._go_home(ok=False, message=str(exc) or type(exc).__name__)
            return
        self._go_home(ok=True, message=f"Ingested {Path(selection.path).name}")

    def start_chat(self) -> None:
        selection = self.selection()
        if selection is None:
            self._go_home(ok=False, message="Pick a chat model first.")
            return
        if self.open_chat is None:
            self._go_home(ok=False, message="Chat is not configured.")
            return
        try:
            self.session = self.open_chat(selection.model)
        except Exception as exc:
            self._go_home(ok=False, message=str(exc) or type(exc).__name__)
            return
        self.turns = []
        self.draft = ""
        self.pending = ""
        self.chat_scroll = 0
        self.step = Step.CHAT

    def answer(self) -> None:
        question, self.pending = self.pending, ""
        self.chat_scroll = 0
        self.step = Step.CHAT
        if self.session is None or not question:
            return
        try:
            reply = self.session.ask(question)
        except Exception as exc:
            self.turns.append(Turn(Speaker.FAILURE, str(exc) or type(exc).__name__))
            return
        self.turns.append(Turn(Speaker.AGENT, reply))

    def leave_chat(self) -> None:
        session, self.session = self.session, None
        self.turns = []
        self.draft = ""
        self.pending = ""
        self.chat_scroll = 0
        if session is not None:
            session.close()

    def _handle_chat(self, key: str) -> None:
        if key == "ctrl+c":
            self.leave_chat()
            self.step = Step.QUIT
            return
        if key == "esc":
            self.leave_chat()
            self._go_home()
            return
        if key == "up":
            self.chat_scroll += 1
            return
        if key == "down":
            self.chat_scroll = max(0, self.chat_scroll - 1)
            return
        if key == "enter":
            question = self.draft.strip()
            if not question:
                return
            self.turns.append(Turn(Speaker.USER, question))
            self.pending = question
            self.draft = ""
            self.chat_scroll = 0
            self.step = Step.THINKING
            return
        if key == "backspace":
            self.draft = self.draft[:-1]
            return
        if len(key) == 1 and key.isprintable():
            self.draft += key

    def _go_home(self, *, ok: bool | None = None, message: str = "") -> None:
        self.notice = None if ok is None else Notice(ok=ok, message=message)
        self.action = None
        self.selected_file = ""
        self.step = Step.HOME


def preferred_from_env() -> tuple[str, str]:
    return (
        os.environ.get("OLLAMA_EMBED_MODEL", "").strip(),
        os.environ.get("OLLAMA_MODEL", "").strip(),
    )


def normalize_key(raw: str) -> str:
    if raw in {"\r", "\n"}:
        return "enter"
    if raw in {"\x7f", "\x08"}:
        return "backspace"
    if raw == "\x03":
        return "ctrl+c"
    if raw == "\x1b[A":
        return "up"
    if raw == "\x1b[B":
        return "down"
    if raw == "\x1b":
        return "esc"
    return raw


def read_key() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        first = sys.stdin.read(1)
        if first != "\x1b":
            return normalize_key(first)
        settings = termios.tcgetattr(fd)
        settings[6][termios.VMIN] = 0
        settings[6][termios.VTIME] = 1
        termios.tcsetattr(fd, termios.TCSANOW, settings)
        extra = sys.stdin.read(2)
        return normalize_key(first + extra)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _chip(label: str) -> Text:
    """Compact badge for confirmed values (ready screen, exit summary)."""
    return Text(f" {label} ", style=f"bold {DARK} on {CYAN}")


def _path_badge(label: str) -> Text:
    """Current folder context — visually separate from selectable rows."""
    return Text(f" {label} ", style=f"{TEAL} on {PANEL}")


def _plain_label_style(tone: str) -> str:
    return {
        "action": f"bold {GREEN}",
        "parent": f"italic {MUTED}",
        "dir": TEAL,
        "item": TEAL,
    }.get(tone, TEAL)


def _selected_label_style(tone: str) -> str:
    return {
        "action": f"bold {LIME} underline",
        "parent": f"italic {TEAL} underline",
        "dir": f"bold {CYAN} underline",
        "item": f"bold {LIME} underline",
    }.get(tone, f"bold {LIME} underline")


def _confirm_chip(label: str, *, selected: bool) -> Text:
    bg = LIME if selected else GREEN
    style = f"bold {DARK} on {bg}"
    if selected:
        style += " underline"
    return Text(f" {label} ", style=style)


def _browse_action_row(*, selected: bool) -> Text:
    row = Text("▸ " if selected else "  ", style=f"bold {LIME}" if selected else "")
    row.append_text(_confirm_chip("✓  use this folder", selected=selected))
    return row


def _browse_parent_row(*, selected: bool) -> Text:
    row = Text("▸ " if selected else "  ", style=f"bold {LIME}" if selected else "")
    label_style = f"italic {TEAL} underline" if selected else f"italic {MUTED}"
    row.append("←  .. (up)", style=label_style)
    return row


def _browse_divider() -> Text:
    return Text("  " + "─" * 28, style=MUTED)


def _picker_row(
    label: str,
    *,
    selected: bool,
    tone: str = "item",
    detail: str = "",
) -> Text:
    if selected:
        row = Text()
        row.append("▸ ", style=f"bold {LIME}")
        row.append(label, style=_selected_label_style(tone))
        if detail:
            row.append(f"  {detail}", style=MUTED)
        return row

    row = Text("  ", style="")
    row.append(label, style=_plain_label_style(tone))
    if detail:
        row.append(f"  {detail}", style=MUTED)
    return row


def _picker_screen(
    title: str,
    hint: str,
    list_rows: list,
    *,
    folder: Path | None = None,
) -> Padding:
    parts: list = [
        Text(title, style=f"bold {CYAN}"),
        Text(hint, style=MUTED),
    ]
    if folder is not None:
        parts.extend([Text(""), _line("folder", _path_badge(format_dir(folder)))])
    if list_rows:
        parts.extend([Text(""), *list_rows])
    return Padding(
        Panel(
            Group(*parts),
            box=ROUNDED,
            border_style=MUTED,
            padding=(1, 2),
        ),
        (1, 2),
    )


def _meta_bits(model: LocalModel) -> str:
    bits = [model.role]
    if model.family:
        bits.append(model.family)
    if model.parameter_size:
        bits.append(model.parameter_size)
    size = format_size(model.size)
    if size:
        bits.append(size)
    return " · ".join(bits)


def _window(cursor: int, total: int, visible: int) -> tuple[int, int]:
    if visible <= 0 or visible >= total:
        return 0, total
    start = cursor - visible // 2
    start = max(0, min(start, total - visible))
    return start, start + visible


def render_header(width: int) -> Panel:
    title = render_title(max(0, width - 10))
    return Panel(
        Align.center(title),
        box=ROUNDED,
        border_style=GREEN,
        padding=(1, 3),
    )


def render_compact_header(feature: str) -> Padding:
    row = Text()
    row.append("LAZYDOCS", style=f"bold {LIME}")
    row.append("  ›  ", style=MUTED)  # noqa: RUF001
    row.append(feature, style=f"bold {CYAN}")
    return Padding(row, (1, 2, 0, 2))


def render(state: AppState, width: int, height: int) -> Group:
    if state.step == Step.HOME:
        return _render_home(state, width)

    parts: list = [render_compact_header(state.feature_title())]

    if state.step == Step.LOADING:
        parts.append(_section("Ollama", "Contacting the local server…"))
        parts.append(_help("q quit"))
    elif state.step == Step.INGESTING:
        parts.append(_section("Ingest", "Chunking, embedding, and storing the index…"))
        parts.append(_help("please wait"))
    elif state.step == Step.ERROR:
        parts.append(_error_body(state))
        parts.append(_help("r/enter retry  •  esc home  •  q quit"))
    elif state.step == Step.PICK_MODEL:
        parts.append(_model_picker(state, height))
        parts.append(_help("↑↓ move  •  enter select  •  esc home  •  q quit"))
    elif state.step == Step.PICK_DIR:
        parts.append(_dir_picker(state, height))
        parts.append(_help("↑↓ move  •  enter open/use  •  esc back  •  q quit"))
    elif state.step == Step.PICK_FILE:
        parts.append(_file_picker(state, height))
        parts.append(_help("↑↓ move  •  enter select  •  esc back  •  q quit"))
    elif state.step == Step.OPENING_CHAT:
        parts.append(_section("Chat", "Loading the model and the index…"))
        parts.append(_help("please wait"))
    elif state.step in {Step.CHAT, Step.THINKING}:
        parts.append(_chat_body(state, width, height))
        if state.step == Step.THINKING:
            parts.append(_help("searching your documents…"))
        else:
            parts.append(_help("↑↓ scroll  •  enter send  •  esc home  •  ctrl+c quit"))
    elif state.step == Step.READY:
        parts.append(_ready_body(state))
        if state.action == Action.INGEST:
            parts.append(_help("enter ingest  •  esc back  •  q quit"))
        else:
            parts.append(_help("enter continue  •  esc back  •  q quit"))

    return Group(*parts)


def _render_home(state: AppState, width: int) -> Group:
    card_width = min(56, max(40, width - 12))
    cards: list = []
    for index, option in enumerate(ACTION_OPTIONS):
        if cards:
            cards.append(Text(""))
        cards.append(
            _action_card(
                option, selected=index == state.action_cursor, width=card_width
            )
        )

    heading: list = []
    if state.notice is not None:
        mark = "✓  " if state.notice.ok else "✗  "
        style = f"bold {GREEN}" if state.notice.ok else f"bold {RED}"
        heading.extend(
            [
                Text(mark + state.notice.message, style=style),
                Text(""),
            ]
        )
    heading.extend(
        [
            Text("What do you want to do?", style=f"bold {CYAN}"),
            Text("Pick a path. The model comes next.", style=MUTED),
            Text(""),
        ]
    )
    menu = Group(
        *heading,
        *cards,
        Text(""),
        Text("↑↓ move  •  enter open  •  q quit", style=MUTED),
    )
    return Group(
        Align.center(render_header(width)),
        Padding(Align.center(menu), (1, 0, 1, 0)),
    )


def _action_card(option: ActionOption, *, selected: bool, width: int) -> Panel:
    title = Text()
    if selected:
        title.append("▶ ", style=f"bold {SELECT}")
        title.append(option.title, style=f"bold {SELECT}")
    else:
        title.append("  ", style=MUTED)
        title.append(option.title, style=f"bold {TEAL}")
    hint = Text("  " + option.hint, style=MUTED, no_wrap=True, overflow="ellipsis")
    return Panel(
        Group(title, hint),
        box=ROUNDED,
        border_style=SELECT if selected else MUTED,
        padding=(0, 1),
        width=width,
    )


def _section(title: str, body: str) -> Padding:
    return Padding(
        Group(
            Text(title, style=f"bold {CYAN}"),
            Text(""),
            Text(body, style=TEAL),
        ),
        (1, 2),
    )


def _line(label: str, value: Text) -> Text:
    row = Text(f"{label}: ", style=f"bold {MUTED}")
    row.append_text(value)
    return row


def _help(text: str) -> Padding:
    return Padding(Text(text, style=MUTED), (1, 2))


def _error_body(state: AppState) -> Padding:
    lines: list[Text] = [
        Text("✗  " + state.error, style=f"bold {RED}"),
        Text(""),
    ]
    if state.error_kind == "unreachable":
        lines.append(Text(f"Docs: {OLLAMA_DOCS}", style=TEAL))
    return Padding(Group(*lines), (1, 2))


def _model_picker(state: AppState, height: int) -> Padding:
    if state.action == Action.INGEST:
        title = "Embedding model"
        hint = "Used to chunk and index documents."
    else:
        title = "Chat model"
        hint = "Used to answer questions."

    items = state.model_items
    if not items:
        return _picker_screen(
            title,
            hint,
            [Text("  no models in this list", style=MUTED)],
        )

    visible = max(3, height - 14) if height else len(items)
    start, end = _window(state.model_cursor, len(items), visible)
    list_rows: list = []
    if start > 0:
        list_rows.append(Text(f"  ↑ {start} above", style=MUTED))
    for index in range(start, end):
        list_rows.append(_model_row(items[index], selected=index == state.model_cursor))
    remaining = len(items) - end
    if remaining > 0:
        list_rows.append(Text(f"  ↓ {remaining} below", style=MUTED))
    return _picker_screen(title, hint, list_rows)


def _model_row(model: LocalModel, *, selected: bool) -> Text:
    return _picker_row(
        model.name,
        selected=selected,
        tone="item",
        detail=_meta_bits(model),
    )


def _dir_picker(state: AppState, height: int) -> Padding:
    title = "Folder"
    hint = "Navigate to the folder that holds the PDFs."
    if state.browse_err:
        return _picker_screen(
            title,
            hint,
            [Text("✗  " + state.browse_err, style=f"bold {RED}")],
            folder=state.browse_dir,
        )
    items = state.browse_entries
    if not items:
        return _picker_screen(
            title,
            hint,
            [Text("  empty folder", style=MUTED)],
            folder=state.browse_dir,
        )

    visible = max(3, height - 16) if height else len(items)
    start, end = _window(state.browse_cursor, len(items), visible)
    list_rows: list = []
    if start > 0:
        list_rows.append(Text(f"  ↑ {start} above", style=MUTED))
    for index in range(start, end):
        entry = items[index]
        if (
            index > start
            and entry.kind == BrowseKind.DIR
            and items[index - 1].kind == BrowseKind.PARENT
        ):
            list_rows.append(_browse_divider())
        list_rows.append(_browse_row(entry, selected=index == state.browse_cursor))
    remaining = len(items) - end
    if remaining > 0:
        list_rows.append(Text(f"  ↓ {remaining} below", style=MUTED))
    return _picker_screen(title, hint, list_rows, folder=state.browse_dir)


def _browse_row(entry: BrowseEntry, *, selected: bool) -> Text:
    if entry.kind == BrowseKind.USE_CURRENT:
        return _browse_action_row(selected=selected)
    if entry.kind == BrowseKind.PARENT:
        return _browse_parent_row(selected=selected)
    return _picker_row(f"{entry.name}/", selected=selected, tone="dir")


def _file_picker(state: AppState, height: int) -> Padding:
    kind = state.selected_type.upper() or FILE_TYPE.upper()
    title = f"{kind} files"
    hint = "Pick the PDF to chunk and index."
    if state.files_err:
        return _picker_screen(
            title,
            hint,
            [Text("✗  " + state.files_err, style=f"bold {RED}")],
            folder=state.browse_dir,
        )
    items = state.files
    if not items:
        suffix = state.selected_type or FILE_TYPE
        return _picker_screen(
            title,
            hint,
            [Text(f"  no .{suffix} files in this folder", style=MUTED)],
            folder=state.browse_dir,
        )

    visible = max(3, height - 16) if height else len(items)
    start, end = _window(state.files_cursor, len(items), visible)
    list_rows: list = []
    if start > 0:
        list_rows.append(Text(f"  ↑ {start} above", style=MUTED))
    for index in range(start, end):
        list_rows.append(
            _choice_row(items[index], selected=index == state.files_cursor)
        )
    remaining = len(items) - end
    if remaining > 0:
        list_rows.append(Text(f"  ↓ {remaining} below", style=MUTED))
    return _picker_screen(title, hint, list_rows, folder=state.browse_dir)


def _choice_row(label: str, *, selected: bool) -> Text:
    return _picker_row(label, selected=selected, tone="item")


def _ready_body(state: AppState) -> Padding:
    model = state.selected_model()
    action_label = state.feature_title()
    model_label = "embedding" if state.action == Action.INGEST else "chat"
    hint = (
        f"{action_label} will use this file and model."
        if state.action == Action.INGEST
        else f"{action_label} will use this model."
    )
    lines = [
        Text("Ready", style=f"bold {CYAN}"),
        Text(hint, style=MUTED),
        Text(""),
        _line("action", _chip(action_label.lower())),
        _line(model_label, _chip(model.name) if model else Text("—", style=MUTED)),
    ]
    if state.action == Action.INGEST:
        file_name = Path(state.selected_file).name if state.selected_file else "—"
        lines.append(_line("file", _chip(file_name)))
    return Padding(Group(*lines), (1, 2))


def _speaker_style(speaker: Speaker) -> str:
    return {
        Speaker.USER: f"bold {CYAN}",
        Speaker.AGENT: f"bold {LIME}",
        Speaker.FAILURE: f"bold {RED}",
    }[speaker]


def _turn_rows(turn: Turn, width: int) -> list[Text]:
    body_style = RED if turn.speaker == Speaker.FAILURE else TEAL
    rows = [Text(turn.speaker, style=_speaker_style(turn.speaker))]
    for paragraph in turn.text.splitlines() or [""]:
        if not paragraph.strip():
            rows.append(Text(""))
            continue
        for line in textwrap.wrap(paragraph, width=width) or [""]:
            rows.append(Text(line, style=body_style))
    return rows


def _transcript_rows(turns: list[Turn], width: int) -> list[Text]:
    rows: list[Text] = []
    for index, turn in enumerate(turns):
        if index:
            rows.append(Text(""))
        rows.extend(_turn_rows(turn, width))
    return rows


def _visible_transcript(
    turns: list[Turn], width: int, budget: int, scroll: int
) -> tuple[list[Text], int]:
    rows = _transcript_rows(turns, width)
    total = len(rows)
    if not budget or total <= budget:
        return rows, 0

    # Overflowing content needs at least one marker line. At the top or
    # bottom that is one line; in the middle it is two.
    inner = budget - 1
    max_scroll = max(0, total - inner)
    scroll = max(0, min(scroll, max_scroll))
    end = total - scroll
    start = end - inner
    if start > 0:
        inner = budget - 2
        start = end - inner
    start = max(0, start)

    visible: list[Text] = []
    if start > 0:
        visible.append(Text(f"↑ {start} above", style=MUTED))
    visible.extend(rows[start:end])
    below = total - end
    if below > 0:
        visible.append(Text(f"↓ {below} below", style=MUTED))
    return visible, scroll


def _prompt_row(state: AppState) -> Text:
    if state.step == Step.THINKING:
        return Text("  ⋯  thinking…", style=MUTED)
    row = Text("  ›  ", style=f"bold {LIME}")  # noqa: RUF001
    row.append(state.draft, style=TEAL)
    row.append("▌", style=CYAN)
    return row


def _chat_body(state: AppState, width: int, height: int) -> Padding:
    inner = max(20, width - 12)
    budget = max(4, height - 13) if height else 0
    rows, state.chat_scroll = _visible_transcript(
        state.turns, inner, budget, state.chat_scroll
    )
    if not rows:
        rows = [Text("Ask anything about the documents you ingested.", style=MUTED)]
    return Padding(
        Group(
            Panel(
                Group(*rows),
                box=ROUNDED,
                border_style=MUTED,
                padding=(1, 2),
            ),
            _prompt_row(state),
        ),
        (1, 2),
    )


def run(
    console: Console | None = None,
    *,
    ingest: Callable[[str, str], None] | None = None,
    open_chat: Callable[[str], ChatSession] | None = None,
) -> None:
    console = console or Console()
    if not console.is_terminal or not sys.stdin.isatty():
        console.print("LazyDocs needs an interactive terminal.", style=f"bold {RED}")
        return

    preferred_embed, preferred_chat = preferred_from_env()
    state = AppState(
        preferred_embed=preferred_embed,
        preferred_chat=preferred_chat,
        ingest=ingest,
        open_chat=open_chat,
    )

    try:
        with console.screen() as screen:
            while state.step != Step.QUIT:
                screen.update(render(state, console.size.width, console.size.height))
                if state.step == Step.LOADING:
                    state.load()
                elif state.step == Step.INGESTING:
                    state.finish_ingest()
                elif state.step == Step.OPENING_CHAT:
                    state.start_chat()
                elif state.step == Step.THINKING:
                    state.answer()
                else:
                    state.handle(read_key())
    finally:
        state.leave_chat()
