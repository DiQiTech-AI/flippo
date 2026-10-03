from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RAG_", extra="ignore", populate_by_name=True)

    data_dir: Path = Path("data")
    database_path: Path = Path("data/rag.db")
    max_upload_mb: int = 50
    chunk_target_chars: int = 3000
    chunk_overlap_chars: int = 450
    max_search_rounds: int = 4
    max_tool_calls: int = 12
    run_timeout_seconds: int = 120

    agent_provider: Literal["demo", "gemini", "unsloth"] = "demo"
    model: str = "gemini-2.5-flash"
    google_api_key: str | None = Field(None, validation_alias=AliasChoices("GOOGLE_API_KEY", "RAG_GOOGLE_API_KEY"))
    unsloth_base_url: str = Field("http://127.0.0.1:8888/v1", validation_alias=AliasChoices("UNSLOTH_BASE_URL", "RAG_UNSLOTH_BASE_URL"))
    unsloth_api_key: str | None = Field(None, validation_alias=AliasChoices("UNSLOTH_API_KEY", "RAG_UNSLOTH_API_KEY"))
    unsloth_model_id: str | None = Field(None, validation_alias=AliasChoices("UNSLOTH_MODEL_ID", "RAG_UNSLOTH_MODEL_ID"))

    embedding_provider: Literal["deterministic", "gemini", "openai"] = "deterministic"
    embedding_model: str = "text-embedding-004"
    embedding_dimension: int = 256
    openai_embedding_base_url: str | None = None
    openai_embedding_api_key: str | None = None

    @property
    def trace_dir(self) -> Path:
        return self.data_dir / "traces"

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "documents"

    @property
    def runtime_ready(self) -> bool:
        if self.agent_provider == "demo":
            return True
        if self.agent_provider == "gemini":
            return bool(self.google_api_key)
        return bool(self.unsloth_api_key and self.unsloth_base_url)

    @property
    def runtime_message(self) -> str:
        if self.agent_provider == "demo":
            return "Demo/test extractive agent is active. Configure Gemini or Unsloth for the real Google ADK path."
        if self.runtime_ready:
            return f"Google ADK is configured with the {self.agent_provider} provider."
        if self.agent_provider == "gemini":
            return "GOOGLE_API_KEY is missing; document import and retrieval remain available."
        return "UNSLOTH_API_KEY or UNSLOTH_BASE_URL is missing; document import and retrieval remain available."

    def prepare(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
