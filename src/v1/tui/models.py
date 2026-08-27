"""Local Ollama models, split into embedding and chat roles."""

from __future__ import annotations

from dataclasses import dataclass

import ollama


class ModelListError(Exception):
    def __init__(self, message: str, *, kind: str) -> None:
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True)
class LocalModel:
    name: str
    role: str
    family: str = ""
    parameter_size: str = ""
    size: int | None = None


def is_embed_model(name: str) -> bool:
    base = name.lower().split(":", 1)[0]
    return any(token in base for token in ("embed", "bge", "nomic-embed"))


def format_size(size: int | None) -> str:
    if size is None or size <= 0:
        return ""
    if size >= 1024**3:
        return f"{size / 1024**3:.1f} GB"
    if size >= 1024**2:
        return f"{size / 1024**2:.0f} MB"
    if size >= 1024:
        return f"{size / 1024:.0f} KB"
    return f"{size} B"


def models_for_role(models: list[LocalModel], role: str) -> list[LocalModel]:
    matched = [model for model in models if model.role == role]
    return matched or list(models)


def index_of(models: list[LocalModel], name: str) -> int:
    needle = name.strip()
    if not needle:
        return 0
    for index, model in enumerate(models):
        base, _, _ = model.name.partition(":")
        if model.name == needle or base == needle:
            return index
    return 0


def list_local_models() -> list[LocalModel]:
    try:
        response = ollama.list()
    except Exception as exc:
        raise ModelListError(
            "Could not reach Ollama. Start it, then press r to retry.",
            kind="unreachable",
        ) from exc

    models: list[LocalModel] = []
    for item in response.models:
        name = (item.model or "").strip()
        if not name:
            continue
        details = item.details
        models.append(
            LocalModel(
                name=name,
                role="embed" if is_embed_model(name) else "chat",
                family=(details.family or "") if details else "",
                parameter_size=(details.parameter_size or "") if details else "",
                size=int(item.size) if item.size is not None else None,
            )
        )

    if not models:
        raise ModelListError(
            "No local models found. Pull one with: ollama pull nomic-embed-text",
            kind="empty",
        )

    models.sort(key=lambda model: (model.role, model.name))
    return models
