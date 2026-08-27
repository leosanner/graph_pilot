from v1.tui.app import Action, AppState, Step
from v1.tui.models import LocalModel, ModelListError


def _models() -> list[LocalModel]:
    return [
        LocalModel(name="nomic-embed-text:latest", role="embed", family="bert"),
        LocalModel(name="bge-m3:latest", role="embed", family="bert"),
        LocalModel(name="llama3.1:8b", role="chat", family="llama", parameter_size="8.0B"),
        LocalModel(name="qwen2.5:7b", role="chat", family="qwen", parameter_size="7.6B"),
    ]


def test_home_opens_ingest_then_picks_an_embedding_model():
    state = AppState(
        preferred_embed="bge-m3",
        list_models=_models,
    )

    assert state.step == Step.HOME
    state.handle("enter")
    assert state.step == Step.LOADING

    state.load()
    assert state.step == Step.PICK_MODEL
    assert state.action == Action.INGEST
    assert [m.name for m in state.model_items] == [
        "nomic-embed-text:latest",
        "bge-m3:latest",
    ]
    assert state.selected_model().name == "bge-m3:latest"

    state.handle("enter")
    assert state.step == Step.READY
    state.handle("enter")
    selection = state.selection()
    assert state.step == Step.DONE
    assert selection is not None
    assert selection.action == Action.INGEST
    assert selection.model == "bge-m3:latest"


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
    state.handle("enter")
    selection = state.selection()
    assert selection is not None
    assert selection.action == Action.CHAT
    assert selection.model == "qwen2.5:7b"


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
