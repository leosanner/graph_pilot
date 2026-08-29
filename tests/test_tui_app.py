from pathlib import Path

from rich.console import Console

from v1.tui.app import Action, AppState, Step, render
from v1.tui.models import LocalModel, ModelListError
from v1.tui.paths import BrowseKind


def _models() -> list[LocalModel]:
    return [
        LocalModel(name="nomic-embed-text:latest", role="embed", family="bert"),
        LocalModel(name="bge-m3:latest", role="embed", family="bert"),
        LocalModel(name="llama3.1:8b", role="chat", family="llama", parameter_size="8.0B"),
        LocalModel(name="qwen2.5:7b", role="chat", family="qwen", parameter_size="7.6B"),
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
