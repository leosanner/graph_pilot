from types import SimpleNamespace

import pytest

from v1.tui.models import (
    LocalModel,
    ModelListError,
    format_size,
    index_of,
    is_embed_model,
    list_local_models,
    models_for_role,
)


def test_is_embed_model_matches_common_names():
    assert is_embed_model("nomic-embed-text")
    assert is_embed_model("bge-m3:latest")
    assert is_embed_model("mxbai-embed-large")
    assert not is_embed_model("llama3.1:8b")
    assert not is_embed_model("qwen2.5")


def test_models_for_role_falls_back_to_all_when_empty():
    models = [
        LocalModel(name="llama3.1", role="chat"),
        LocalModel(name="qwen2.5", role="chat"),
    ]

    assert models_for_role(models, "embed") == models
    assert models_for_role(models, "chat") == models


def test_index_of_matches_base_or_tagged_name():
    models = [
        LocalModel(name="llama3.1:8b", role="chat"),
        LocalModel(name="nomic-embed-text:latest", role="embed"),
    ]

    assert index_of(models, "nomic-embed-text") == 1
    assert index_of(models, "missing") == 0


def test_format_size():
    assert format_size(None) == ""
    assert format_size(2048) == "2 KB"
    assert format_size(137 * 1024 * 1024) == "137 MB"
    assert format_size(int(2.0 * 1024**3)) == "2.0 GB"


def test_list_local_models_maps_ollama_payload(monkeypatch):
    monkeypatch.setattr(
        "v1.tui.models.ollama.list",
        lambda: SimpleNamespace(
            models=[
                SimpleNamespace(
                    model="nomic-embed-text:latest",
                    size=274_000_000,
                    details=SimpleNamespace(family="nomic-bert", parameter_size="137M"),
                ),
                SimpleNamespace(
                    model="llama3.1:8b",
                    size=4_900_000_000,
                    details=SimpleNamespace(family="llama", parameter_size="8.0B"),
                ),
            ]
        ),
    )

    models = list_local_models()

    assert [model.name for model in models] == [
        "llama3.1:8b",
        "nomic-embed-text:latest",
    ]
    assert models[0].role == "chat"
    assert models[1].role == "embed"


def test_list_local_models_raises_when_ollama_is_down(monkeypatch):
    def fail():
        raise ConnectionError("connection refused")

    monkeypatch.setattr("v1.tui.models.ollama.list", fail)

    with pytest.raises(ModelListError, match="Could not reach Ollama") as exc:
        list_local_models()

    assert exc.value.kind == "unreachable"


def test_list_local_models_raises_when_none_are_installed(monkeypatch):
    monkeypatch.setattr(
        "v1.tui.models.ollama.list",
        lambda: SimpleNamespace(models=[]),
    )

    with pytest.raises(ModelListError, match="No local models") as exc:
        list_local_models()

    assert exc.value.kind == "empty"
