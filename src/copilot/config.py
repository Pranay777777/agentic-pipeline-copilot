"""Typed application settings, loaded from the environment."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Every configurable value lives here — never read os.environ directly."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: Literal["local", "ci", "staging", "prod"] = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    database_url: str = "postgresql://app:app@localhost:5432/app"

    # Model provider (ADR-004): OpenRouter's free models by default.
    openrouter_api_key: SecretStr = SecretStr("")
    llm_model: str = "openrouter/free"
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_timeout_s: float = 180.0
    llm_max_tokens: int = 8000
    """Caps every completion, so one request can never cost more than this. Room for
    a reasoning model's hidden thinking plus the JSON answer."""
    llm_fallback_models: str = ""
    """Comma-separated models OpenRouter falls back to when LLM_MODEL is rate limited."""
    llm_max_retries: int = 3
    llm_reasoning_effort: Literal["low", "medium", "high", ""] = "low"
    """How long reasoning models may think; empty sends nothing."""

    # Agents (ADR-002: everything bounded).
    agent_max_attempts: int = 2
    """Model answers per agent before its deterministic check rejects the stage."""
    critic_max_rounds: int = 1
    """Times the Generator may revise a notebook the Critic rejected."""


@lru_cache
def get_settings() -> Settings:
    """Return the cached settings instance."""
    return Settings()
