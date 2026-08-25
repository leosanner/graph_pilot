from pydantic_settings import BaseSettings, SettingsConfigDict
from psycopg_pool import ConnectionPool

class PostgresConfig(BaseSettings):
  url: str

  model_config = SettingsConfigDict(
    env_prefix="DATABASE_"
  )

class Database:
  def __init__(self, config: PostgresConfig):
    self.pool = ConnectionPool(conninfo=config.url)

  def connect(self):
    return self.pool.connection()
