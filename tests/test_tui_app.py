from pathlib import Path

from rich.console import Console

from v1.tui.app import Action, AppState, Speaker, Step, render
from v1.tui.models import LocalModel, ModelListError
from v1.tui.paths import BrowseKind


def _models() -> list[LocalModel]:
    return [
        LocalModel(name="nomic-embed-text:latest", role="embed", family="bert"),
        LocalModel(name="bge-m3:latest", role="embed", family="bert"),
        LocalModel(
            name="llama3.1:8b", role="chat", family="llama", parameter_size="8.0B"
        ),
        LocalModel(
            name="qwen2.5:7b", role="chat", family="qwen", parameter_size="7.6B"
        ),
    ]


def _ingest_at_model(tmp_path: Path, ingest=None) -> AppState:
    state = AppState(
        preferred_embed="bge-m3",
        list_models=_models,
        browse_dir=tmp_path,
        ingest=ingest or (lambda _path, _model: None),
    )
    state.handle("enter")
    state.load()
    return state


def test_home_opens_ingest_then_picks_an_embedding_model(tmp_path: Path):
    (tmp_path / "notes.pdf").write_bytes(b"%PDF-1.4")
    state = _ingest_at_model(tmp_path)

    assert state.step == Step.PICK_MODEL
    assert state.action == Action.INGEST
    assert [m.name for m in state.model_items] == [
        "nomic-embed-text:latest",
        "bge-m3:latest",
    ]
    assert state.selected_model().name == "bge-m3:latest"

    state.handle("enter")
    assert state.step == Step.PICK_DIR
    assert state.browse_entries[0].kind == BrowseKind.USE_CURRENT

    state.handle("enter")
    assert state.step == Step.PICK_FILE
    assert state.selected_type == "pdf"
    assert state.files == ["notes.pdf"]

    state.handle("enter")
    assert state.step == Step.READY
    state.handle("enter")
    assert state.step == Step.INGESTING
    state.finish_ingest()
    assert state.step == Step.HOME
    assert state.notice is not None
    assert state.notice.ok
    assert state.notice.message == "Ingested notes.pdf"


def test_home_opens_chat_then_picks_a_chat_model():
    state = AppState(
        preferred_chat="qwen2.5:7b",
        list_models=_models,
    )

    state.handle("down")
    state.handle("enter")
    state.load()

    assert state.action == Action.CHAT
    assert [m.name for m in state.model_items] == ["llama3.1:8b", "qwen2.5:7b"]
    assert state.selected_model().name == "qwen2.5:7b"

    state.handle("enter")
    assert state.step == Step.READY
    state.handle("enter")
    selection = state.selection()
    assert selection is not None
    assert selection.action == Action.CHAT
    assert selection.model == "qwen2.5:7b"
    assert selection.path is None


class FakeSession:
    def __init__(
        self, reply: str = "42 pages, all of them dull.", error: Exception | None = None
    ):
        self.reply = reply
        self.error = error
        self.asked: list[str] = []
        self.closed = False

    def ask(self, question: str) -> str:
        self.asked.append(question)
        if self.error is not None:
            raise self.error
        return self.reply

    def close(self) -> None:
        self.closed = True


def _chat_at_prompt(session: FakeSession | None = None) -> tuple[AppState, FakeSession]:
    session = session or FakeSession()
    state = AppState(
        preferred_chat="qwen2.5:7b",
        list_models=_models,
        open_chat=lambda _model: session,
    )
    state.handle("down")
    state.handle("enter")
    state.load()
    state.handle("enter")
    state.handle("enter")
    state.start_chat()
    return state, session


def _type(state: AppState, text: str) -> None:
    for char in text:
        state.handle(char)


def test_chat_sends_a_question_and_shows_the_answer():
    state, session = _chat_at_prompt()
    assert state.step == Step.CHAT

    _type(state, "how long is the report?")
    assert state.draft == "how long is the report?"

    state.handle("enter")
    assert state.step == Step.THINKING
    assert state.draft == ""

    state.answer()
    assert state.step == Step.CHAT
    assert session.asked == ["how long is the report?"]
    assert [(turn.speaker, turn.text) for turn in state.turns] == [
        (Speaker.USER, "how long is the report?"),
        (Speaker.AGENT, "42 pages, all of them dull."),
    ]


def test_chat_typing_does_not_trigger_the_single_letter_shortcuts():
    state, _ = _chat_at_prompt()

    _type(state, "quick req")

    assert state.step == Step.CHAT
    assert state.draft == "quick req"


def test_chat_backspace_edits_the_draft():
    state, _ = _chat_at_prompt()

    _type(state, "abc")
    state.handle("backspace")
    state.handle("backspace")

    assert state.draft == "a"


def test_chat_ignores_an_empty_question():
    state, session = _chat_at_prompt()

    _type(state, "   ")
    state.handle("enter")

    assert state.step == Step.CHAT
    assert state.turns == []
    assert session.asked == []


def test_chat_keeps_going_after_the_agent_fails():
    state, session = _chat_at_prompt(FakeSession(error=RuntimeError("ollama is down")))

    _type(state, "hello")
    state.handle("enter")
    state.answer()

    assert state.step == Step.CHAT
    assert state.turns[-1].speaker == Speaker.FAILURE
    assert state.turns[-1].text == "ollama is down"

    session.error = None
    _type(state, "again")
    state.handle("enter")
    state.answer()

    assert state.turns[-1].speaker == Speaker.AGENT


def test_chat_esc_closes_the_session_and_returns_home():
    state, session = _chat_at_prompt()
    _type(state, "hi")
    state.handle("enter")
    state.answer()

    state.handle("esc")

    assert state.step == Step.HOME
    assert state.notice is None
    assert session.closed
    assert state.session is None
    assert state.turns == []


def test_chat_ctrl_c_closes_the_session_before_quitting():
    state, session = _chat_at_prompt()

    state.handle("ctrl+c")

    assert state.step == Step.QUIT
    assert session.closed


def test_chat_without_a_configured_session_returns_home_with_a_notice():
    state = AppState(preferred_chat="qwen2.5:7b", list_models=_models)
    state.handle("down")
    state.handle("enter")
    state.load()
    state.handle("enter")
    state.handle("enter")
    assert state.step == Step.OPENING_CHAT

    state.start_chat()

    assert state.step == Step.HOME
    assert state.notice is not None
    assert not state.notice.ok
    assert state.notice.message == "Chat is not configured."


def test_chat_start_failure_returns_home_with_the_error():
    def boom(_model: str):
        raise RuntimeError("no chunks ingested yet")

    state = AppState(
        preferred_chat="qwen2.5:7b",
        list_models=_models,
        open_chat=boom,
    )
    state.handle("down")
    state.handle("enter")
    state.load()
    state.handle("enter")
    state.handle("enter")
    state.start_chat()

    assert state.step == Step.HOME
    assert state.notice is not None
    assert not state.notice.ok
    assert state.notice.message == "no chunks ingested yet"


def test_chat_render_shows_the_transcript_and_the_draft():
    state, _ = _chat_at_prompt()
    _type(state, "how long is the report?")
    state.handle("enter")
    state.answer()
    _type(state, "and the author?")

    console = Console(record=True, width=80)
    console.print(render(state, 80, 24))
    text = console.export_text()

    assert "how long is the report?" in text
    assert "42 pages" in text
    assert "and the author?" in text
    assert "enter send" in text


def test_chat_render_shows_a_thinking_hint_while_the_agent_runs():
    state, _ = _chat_at_prompt()
    _type(state, "why?")
    state.handle("enter")

    console = Console(record=True, width=80)
    console.print(render(state, 80, 24))
    text = console.export_text()

    assert "thinking" in text
    assert "why?" in text
    assert text.lower().count("thinking") == 1


def _shown(state: AppState, *, width: int = 80, height: int = 24) -> str:
    console = Console(record=True, width=width)
    console.print(render(state, width, height))
    return console.export_text()


def test_chat_scrolls_a_long_answer_with_arrow_keys():
    reply = "\n".join(f"line-{i:02d}" for i in range(40))
    state, _ = _chat_at_prompt(FakeSession(reply=reply))
    _type(state, "go")
    state.handle("enter")
    state.answer()

    bottom = _shown(state)
    assert "line-39" in bottom
    assert "line-00" not in bottom
    assert "above" in bottom
    assert "↑↓ scroll" in bottom

    for _ in range(100):
        state.handle("up")
    top = _shown(state)
    assert "line-00" in top
    assert "line-39" not in top
    assert "below" in top

    for _ in range(100):
        state.handle("down")
    bottom_again = _shown(state)
    assert "line-39" in bottom_again
    assert "line-00" not in bottom_again


def test_chat_down_at_the_bottom_stays_put():
    state, _ = _chat_at_prompt(FakeSession(reply="short"))
    _type(state, "go")
    state.handle("enter")
    state.answer()

    state.handle("down")
    assert state.chat_scroll == 0
    assert "short" in _shown(state)


def test_chat_sending_a_question_resets_scroll_to_the_latest():
    reply = "\n".join(f"line-{i:02d}" for i in range(40))
    state, _ = _chat_at_prompt(FakeSession(reply=reply))
    _type(state, "go")
    state.handle("enter")
    state.answer()
    _shown(state)
    state.handle("up")
    state.handle("up")
    assert state.chat_scroll > 0

    _type(state, "next")
    state.handle("enter")
    assert state.chat_scroll == 0


def test_ingest_failure_returns_home_with_the_error(tmp_path: Path):
    (tmp_path / "notes.pdf").write_bytes(b"%PDF-1.4")
    calls: list[tuple[str, str]] = []

    def boom(path: str, model: str) -> None:
        calls.append((path, model))
        raise RuntimeError("could not embed")

    state = _ingest_at_model(tmp_path, ingest=boom)
    state.handle("enter")
    state.handle("enter")
    state.handle("enter")
    state.handle("enter")
    state.finish_ingest()

    assert calls == [
        (str((tmp_path / "notes.pdf").resolve()), "bge-m3:latest"),
    ]
    assert state.step == Step.HOME
    assert state.notice is not None
    assert not state.notice.ok
    assert state.notice.message == "could not embed"
    assert state.action is None
    assert state.selection() is None


def test_ingest_esc_walks_back_through_the_path_picker(tmp_path: Path):
    (tmp_path / "notes.pdf").write_bytes(b"%PDF-1.4")
    state = _ingest_at_model(tmp_path)
    state.handle("enter")
    state.handle("enter")
    state.handle("enter")
    assert state.step == Step.READY

    state.handle("esc")
    assert state.step == Step.PICK_FILE
    state.handle("esc")
    assert state.step == Step.PICK_DIR
    state.handle("esc")
    assert state.step == Step.PICK_MODEL


def test_ingest_navigates_into_a_nested_folder(tmp_path: Path):
    nested = tmp_path / "papers"
    nested.mkdir()
    (nested / "guide.pdf").write_bytes(b"%PDF-1.4")
    state = _ingest_at_model(tmp_path)
    state.handle("enter")

    kinds = [entry.kind for entry in state.browse_entries]
    assert BrowseKind.DIR in kinds
    dir_index = kinds.index(BrowseKind.DIR)
    state.browse_cursor = dir_index
    state.handle("enter")

    assert state.browse_dir == nested.resolve()
    state.handle("enter")
    assert state.selected_type == "pdf"
    assert state.files == ["guide.pdf"]
    state.handle("enter")
    assert state.selected_file == str((nested / "guide.pdf").resolve())


def test_ingest_does_not_select_when_the_folder_has_no_matching_files(
    tmp_path: Path,
):
    state = _ingest_at_model(tmp_path)
    state.handle("enter")
    state.handle("enter")

    assert state.step == Step.PICK_FILE
    assert state.files == []
    state.handle("enter")
    assert state.step == Step.PICK_FILE
    assert state.selection() is None


def test_ingest_lists_only_pdfs_in_the_folder(tmp_path: Path):
    (tmp_path / "keep.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "notes.txt").write_text("hello")
    (tmp_path / "guide.md").write_text("md")
    state = _ingest_at_model(tmp_path)
    state.handle("enter")
    state.handle("enter")

    assert state.step == Step.PICK_FILE
    assert state.files == ["keep.pdf"]


def test_folder_listing_error_stays_on_the_dir_picker(tmp_path: Path):
    def boom(_directory: Path):
        raise OSError("permission denied")

    state = AppState(
        preferred_embed="bge-m3",
        list_models=_models,
        browse_dir=tmp_path,
        list_entries=boom,
    )
    state.handle("enter")
    state.load()
    state.handle("enter")

    assert state.step == Step.PICK_DIR
    assert state.browse_err == "permission denied"
    state.handle("enter")
    assert state.step == Step.PICK_DIR


def test_esc_returns_to_home_from_the_model_picker():
    state = AppState(list_models=_models)
    state.handle("enter")
    state.load()
    assert state.step == Step.PICK_MODEL

    state.handle("esc")
    assert state.step == Step.HOME

    state.handle("q")
    assert state.step == Step.QUIT


def test_cached_models_skip_a_second_ollama_roundtrip():
    calls = {"n": 0}

    def once() -> list[LocalModel]:
        calls["n"] += 1
        return _models()

    state = AppState(list_models=once)
    state.handle("enter")
    state.load()
    state.handle("esc")
    state.handle("down")
    state.handle("enter")

    assert calls["n"] == 1
    assert state.step == Step.PICK_MODEL
    assert state.action == Action.CHAT


def test_load_surfaces_ollama_errors_inside_the_feature():
    def fail():
        raise ModelListError("Could not reach Ollama.", kind="unreachable")

    state = AppState(list_models=fail)
    state.handle("enter")
    state.load()

    assert state.step == Step.ERROR
    assert state.error_kind == "unreachable"

    state.handle("esc")
    assert state.step == Step.HOME
    state.handle("enter")
    assert state.step == Step.LOADING


def test_folder_picker_render_shows_use_this_folder(tmp_path: Path):
    state = _ingest_at_model(tmp_path)
    state.handle("enter")
    console = Console(record=True, width=80)
    console.print(render(state, 80, 24))
    text = console.export_text()
    assert "use this folder" in text
    assert "Folder" in text
    assert "PDFs" in text


def test_file_picker_render_lists_pdfs(tmp_path: Path):
    (tmp_path / "notes.pdf").write_bytes(b"%PDF-1.4")
    state = _ingest_at_model(tmp_path)
    state.handle("enter")
    state.handle("enter")
    console = Console(record=True, width=80)
    console.print(render(state, 80, 24))
    text = console.export_text()
    assert "PDF files" in text
    assert "notes.pdf" in text


def test_home_render_shows_ingest_result(tmp_path: Path):
    (tmp_path / "notes.pdf").write_bytes(b"%PDF-1.4")
    state = _ingest_at_model(tmp_path)
    state.handle("enter")
    state.handle("enter")
    state.handle("enter")
    state.handle("enter")
    state.finish_ingest()
    console = Console(record=True, width=80)
    console.print(render(state, 80, 24))
    text = console.export_text()
    assert "Ingested notes.pdf" in text
    assert "What do you want to do?" in text
