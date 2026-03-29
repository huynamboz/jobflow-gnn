from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    embedding_provider: str = "english"
    embedding_dim: int = 384

    data_dir: Path = Path("./data")
    skill_alias_path: Path = Path("../roadmap/week1/skill-alias.json")

    service_port: int = 8001

    num_cvs: int = 800
    num_jobs: int = 1500
    num_positive_pairs: int = 2000
    random_seed: int = 42

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
