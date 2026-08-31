import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ollama

from v1.app.ingestion.vector.errors import ModelInfoError

DIR = Path(__file__).parent
CALIBRATION_FILE = "calibration_texts.json"


@dataclass
class ModelMetadata:
    context_length: int
    embedding_dim: int
    capabilites: list[str]


@dataclass
class ModelSpecs:
    metadata: ModelMetadata
    model_name: str
    char_per_token: float


def load_calibration_data() -> list[str]:
    with open(DIR / CALIBRATION_FILE, encoding="utf-8") as file:
        return [sample["text"] for sample in json.load(file)["samples"]]


def _int_by_suffix(model_info: Mapping[str, Any], suffix: str) -> int | None:
    for key, value in model_info.items():
        if key.endswith(suffix) and value is not None:
            return int(value)
    return None


def load_model_metadata(model_name: str) -> ModelMetadata:
    try:
        response = ollama.show(model_name)
    except Exception as e:
        raise ModelInfoError(f"Failed to load metadata for {model_name}") from e

    model_info = response.modelinfo or {}
    context_length = _int_by_suffix(model_info, ".context_length")
    embedding_dim = _int_by_suffix(model_info, ".embedding_length")
    capabilities = getattr(response, "capabilities", [])

    if context_length is None or embedding_dim is None:
        raise ModelInfoError(
            f"Model {model_name} did not expose context_length/embedding_length"
        )

    return ModelMetadata(
        context_length=context_length,
        embedding_dim=embedding_dim,
        capabilites=capabilities,
    )


def load_model_chunking_specs(model_name: str) -> float:
    char_per_token = 0
    calibration_data = load_calibration_data()

    for sample_text in calibration_data:
        response = ollama.embed(model=model_name, input=sample_text)

        char_per_token += len(sample_text) / (
            response.prompt_eval_count * len(calibration_data)
        )

    return char_per_token


def load_model_specs(model_name: str) -> ModelSpecs:
    model_metadata = load_model_metadata(model_name)
    char_per_token = load_model_chunking_specs(model_name)

    return ModelSpecs(
        metadata=model_metadata, model_name=model_name, char_per_token=char_per_token
    )


if __name__ == "__main__":
    print(load_model_specs("nomic-embed-text:latest"))
