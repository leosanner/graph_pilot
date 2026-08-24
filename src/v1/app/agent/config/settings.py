from pydantic_settings import BaseSettings, SettingsConfigDict

class ModelSettings(BaseSettings):
  model_provider: str = "ollama"
  model_name: str = "example"
  temperature: int = 0
