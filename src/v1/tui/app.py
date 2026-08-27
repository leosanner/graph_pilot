"""Minimal LazyDocs terminal: pick a path, then pick that path's model."""

from __future__ import annotations

import os
import sys
import termios
import tty
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Callable

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

OLLAMA_DOCS = "https://docs.ollama.com/"


class Step(StrEnum):
    HOME = "home"
    LOADING = "loading"
    PICK_MODEL = "pick_model"
    READY = "ready"
    ERROR = "error"
    DONE = "done"
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
            elif self.step == Step.READY:
                self.step = Step.PICK_MODEL
            return

        if self.step == Step.PICK_MODEL:
            if not self.model_items:
                return
            if key in {"up", "k"}:
                self._move(-1)
            elif key in {"down", "j"}:
                self._move(1)
            elif key == "enter":
                self.step = Step.READY
            return

        if self.step == Step.READY and key == "enter":
            self.step = Step.DONE

    def _prepare_picker(self) -> None:
        role = "embed" if self.action == Action.INGEST else "chat"
        preferred = (
            self.preferred_embed if self.action == Action.INGEST else self.preferred_chat
        )
        self.model_items = models_for_role(self.models, role)
        self.model_cursor = index_of(self.model_items, preferred)

    def _move(self, delta: int) -> None:
        if not self.model_items:
            return
        self.model_cursor = (self.model_cursor + delta) % len(self.model_items)

    def selection(self) -> Selection | None:
        model = self.selected_model()
        if self.action is None or model is None:
            return None
        return Selection(action=self.action, model=model.name)


def preferred_from_env() -> tuple[str, str]:
    return (
        os.environ.get("OLLAMA_EMBED_MODEL", "").strip(),
        os.environ.get("OLLAMA_MODEL", "").strip(),
    )


def normalize_key(raw: str) -> str:
    if raw in {"\r", "\n"}:
        return "enter"
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


def _chip(label: str, *, selected: bool = False) -> Text:
    bg = LIME if selected else CYAN
    return Text(f" {label} ", style=f"bold {DARK} on {bg}")


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
    row.append("  ›  ", style=MUTED)
    row.append(feature, style=f"bold {CYAN}")
    return Padding(row, (1, 2, 0, 2))


def render(state: AppState, width: int, height: int) -> Group:
    if state.step == Step.HOME:
        return _render_home(state, width)

    parts: list = [render_compact_header(state.feature_title())]

    if state.step == Step.LOADING:
        parts.append(_section("Ollama", "Contacting the local server…"))
        parts.append(_help("q quit"))
    elif state.step == Step.ERROR:
        parts.append(_error_body(state))
        parts.append(_help("r/enter retry  •  esc home  •  q quit"))
    elif state.step == Step.PICK_MODEL:
        parts.append(_model_picker(state, height))
        parts.append(_help("↑↓ move  •  enter select  •  esc home  •  q quit"))
    elif state.step == Step.READY:
        parts.append(_ready_body(state))
        parts.append(_help("enter continue  •  esc back  •  q quit"))

    return Group(*parts)


def _render_home(state: AppState, width: int) -> Group:
    card_width = min(56, max(40, width - 12))
    cards: list = []
    for index, option in enumerate(ACTION_OPTIONS):
        if cards:
            cards.append(Text(""))
        cards.append(
            _action_card(option, selected=index == state.action_cursor, width=card_width)
        )

    menu = Group(
        Text("What do you want to do?", style=f"bold {CYAN}"),
        Text("Pick a path. The model comes next.", style=MUTED),
        Text(""),
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

    rows: list = [
        Text(title, style=f"bold {CYAN}"),
        Text(hint, style=MUTED),
        Text(""),
    ]
    items = state.model_items
    if not items:
        rows.append(Text("  no models in this list", style=MUTED))
        return Padding(Group(*rows), (1, 2))

    visible = max(3, height - 12) if height else len(items)
    start, end = _window(state.model_cursor, len(items), visible)
    if start > 0:
        rows.append(Text(f"  ↑ {start} above", style=MUTED))
    for index in range(start, end):
        rows.append(_model_row(items[index], selected=index == state.model_cursor))
    remaining = len(items) - end
    if remaining > 0:
        rows.append(Text(f"  ↓ {remaining} below", style=MUTED))
    return Padding(Group(*rows), (1, 2))


def _model_row(model: LocalModel, *, selected: bool) -> Text:
    marker = "▶ " if selected else "  "
    marker_style = f"bold {SELECT}" if selected else MUTED
    row = Text(marker, style=marker_style)
    row.append_text(_chip(model.name, selected=selected))
    meta = _meta_bits(model)
    if meta:
        row.append(f"  {meta}", style=MUTED)
    return row


def _ready_body(state: AppState) -> Padding:
    model = state.selected_model()
    action_label = state.feature_title()
    model_label = "embedding" if state.action == Action.INGEST else "chat"
    return Padding(
        Group(
            Text("Ready", style=f"bold {CYAN}"),
            Text(f"{action_label} will use this model.", style=MUTED),
            Text(""),
            _line("action", _chip(action_label.lower())),
            _line(model_label, _chip(model.name) if model else Text("—", style=MUTED)),
        ),
        (1, 2),
    )


def run(console: Console | None = None) -> Selection | None:
    console = console or Console()
    if not console.is_terminal or not sys.stdin.isatty():
        console.print("LazyDocs needs an interactive terminal.", style=f"bold {RED}")
        return None

    preferred_embed, preferred_chat = preferred_from_env()
    state = AppState(
        preferred_embed=preferred_embed,
        preferred_chat=preferred_chat,
    )

    with console.screen() as screen:
        while state.step not in {Step.DONE, Step.QUIT}:
            screen.update(render(state, console.size.width, console.size.height))
            if state.step == Step.LOADING:
                state.load()
                continue
            state.handle(read_key())

    if state.step != Step.DONE:
        return None

    selection = state.selection()
    if selection is None:
        return None

    console.print()
    console.print(Text("LAZYDOCS", style=f"bold {LIME}"))
    console.print(_line("action", _chip(selection.action)))
    label = "embedding" if selection.action == Action.INGEST else "chat"
    console.print(_line(label, _chip(selection.model)))
    console.print()
    return selection
