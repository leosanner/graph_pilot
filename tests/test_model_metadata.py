from types import SimpleNamespace

import pytest

from v1.app.ingestion.vector.errors import ModelInfoError
from v1.app.ingestion.vector.utils.utils import load_model_information


def test_load_model_information_reads_architecture_fields(monkeypatch):
    monkeypatch.setattr(
        "v1.app.ingestion.vector.utils.utils.ollama.show",
        lambda name: SimpleNamespace(
            modelinfo={
                "general.architecture": "nomic-bert",
                "nomic-bert.context_length": 2048,
                "nomic-bert.embedding_length": 768,
            }
        ),
    )

    meta = load_model_information("nomic-embed-text")

    assert meta.model_name == "nomic-embed-text"
    assert meta.context_length == 2048
    assert meta.embedding_dim == 768


def test_load_model_information_raises_when_show_fails(monkeypatch):
    def fail(name):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(
        "v1.app.ingestion.vector.utils.utils.ollama.show",
        fail,
    )

    with pytest.raises(ModelInfoError, match="nomic-embed-text"):
        load_model_information("nomic-embed-text")


def test_load_model_information_raises_when_fields_are_missing(monkeypatch):
    monkeypatch.setattr(
        "v1.app.ingestion.vector.utils.utils.ollama.show",
        lambda name: SimpleNamespace(modelinfo={"general.architecture": "llama"}),
    )

    with pytest.raises(ModelInfoError, match="did not expose"):
        load_model_information("llama3.2:3b")
