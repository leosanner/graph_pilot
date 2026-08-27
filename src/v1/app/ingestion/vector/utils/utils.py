from dataclasses import dataclass
from typing import Any, Mapping
import ollama
from v1.app.ingestion.vector.errors import ModelInfoError

@dataclass
class ModelMetadata:
  model_name: str
  context_length: int
  embedding_dim: int
  capabilites: list[str]


def _int_by_suffix(model_info: Mapping[str, Any], suffix: str) -> int | None:
  for key, value in model_info.items():
    if key.endswith(suffix) and value is not None:
      return int(value)
  return None


def load_model_information(model_name: str) -> ModelMetadata:
  try:
    response = ollama.show(model_name)
  except Exception as e:
    raise ModelInfoError(f"Failed to load metadata for {model_name}") from e

  model_info = response.modelinfo or {}
  context_length = _int_by_suffix(model_info, ".context_length")
  embedding_dim = _int_by_suffix(model_info, ".embedding_length")

  if context_length is None or embedding_dim is None:
    raise ModelInfoError(
      f"Model {model_name} did not expose context_length/embedding_length"
    )

  return ModelMetadata(
    model_name=model_name,
    context_length=context_length,
    embedding_dim=embedding_dim,
    capabilites=response.capabilities
  )
