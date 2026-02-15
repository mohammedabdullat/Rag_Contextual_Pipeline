"""
Central configuration management using Pydantic Settings.
All settings can be overridden via environment variables or .env file.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LLM / Embedding (swap-friendly) ---
    llm_api_key: str = Field(default="sk-placeholder", description="LLM API key")
    llm_base_url: str = Field(
        default="https://api.openai.com/v1", description="Base URL — swap for any OpenAI-compatible provider"
    )
    openai_embedding_model: str = Field(default="text-embedding-3-small")
    llm_chat_model: str = Field(default="gpt-4o-mini")

    # --- Document / Paths ---
    pdf_path: str = Field(default="rag-pipeline/data/paper.pdf")
    chroma_persist_dir: str = Field(default="data/chroma_db")
    ground_truth_path: str = Field(default="rag-pipeline/data/ground_truth.json")

    # --- Chunking ---
    chunk_size: int = Field(default=512, ge=64, le=2048)
    chunk_overlap: int = Field(default=50, ge=0, le=256)
    top_k: int = Field(default=5, ge=1, le=20)

    # --- Contextual Retrieval ---
    context_window_chars: int = Field(
        default=12000,
        description="Characters of full doc sent to LLM for context generation",
    )

    # --- App ---
    log_level: str = Field(default="INFO")
    app_title: str = "Contextual RAG Pipeline"
    app_version: str = "1.0.0"


@lru_cache()
def get_settings() -> Settings:
    """Cached singleton — call get_settings() anywhere in the app."""
    return Settings()
