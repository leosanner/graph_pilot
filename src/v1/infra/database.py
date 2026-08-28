from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool
from pydantic_settings import BaseSettings, SettingsConfigDict

class PostgresSettings(BaseSettings):
  url: str
  min_connections: int = 1
  max_connections: int = 10

  model_config = SettingsConfigDict(
    env_prefix="DATABASE_"
  )

class PostgresClient:
  def __init__(self, config: PostgresSettings):
      self.pool = ConnectionPool(
         conninfo = config.url,
         min_size = config.min_connections,
         max_size = config.max_connections,
         configure= register_vector
      )

  def connection(self):
     return self.pool.connection()

  def close(self):
    self.pool.close()
