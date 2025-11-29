"""
Configuration management for the dental clinic system.
"""

import os
from typing import Optional, Literal
from pydantic import Field
from pydantic_settings import BaseSettings

from patient_ai_service.models.enums import LLMProvider, AnthropicModel, OpenAIModel


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Application
    app_name: str = "Dental Clinic AI System"
    version: str = "2.0.0"
    environment: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"
    debug: bool = False

    # API Keys
    anthropic_api_key: Optional[str] = Field(None, alias="ANTHROPIC_API_KEY")
    openai_api_key: Optional[str] = Field(None, alias="OPENAI_API_KEY")

    # LLM Configuration
    llm_provider: LLMProvider = Field(LLMProvider.ANTHROPIC, alias="LLM_PROVIDER")
    anthropic_model: AnthropicModel = Field(
        AnthropicModel.CLAUDE_SONNET_4_5,
        alias="ANTHROPIC_MODEL"
    )
    openai_model: OpenAIModel = Field(OpenAIModel.GPT_4O_MINI, alias="OPENAI_MODEL")
    llm_max_tokens: int = 1000
    llm_temperature: float = 0.7
    llm_timeout: int = 30

    # Database (DB-Ops)
    db_ops_url: str = Field("http://db-ops:8001", alias="DB_OPS_URL")
    db_ops_user_email: Optional[str] = Field(None, alias="DB_OPS_USER_EMAIL")
    db_ops_user_password: Optional[str] = Field(None, alias="DB_OPS_USER_PASSWORD")

    # Redis (for production state management)
    redis_url: str = Field("redis://localhost:6379", alias="REDIS_URL")
    redis_enabled: bool = Field(False, alias="REDIS_ENABLED")
    session_ttl: int = 86400  # 24 hours in seconds

    # Session Management
    max_sessions: int = 10000
    max_concurrent_requests: int = 100
    response_timeout: int = 30

    # Features
    enable_translation: bool = True
    enable_classification: bool = True
    enable_medical_inquiry: bool = True
    enable_emergency_response: bool = True

    # Validation (Closed-Loop)
    enable_validation: bool = Field(
        default=True,
        description="Enable closed-loop validation of agent responses"
    )
    validation_max_retries: int = Field(
        default=1,
        description="Max retry attempts (1 to minimize latency)"
    )
    validation_confidence_threshold: float = Field(
        default=0.7,
        description="Min confidence to send response"
    )
    validation_temperature: float = Field(
        default=0.2,
        description="LLM temperature for validation calls"
    )

    # Clinic Configuration
    default_clinic_id: str = "clinic_001"
    clinic_name: str = "Bright Smile Dental Clinic"
    clinic_phone: str = "+971-XXX-XXXX"
    clinic_email: str = "info@brightsmile.clinic"

    # Security
    cors_origins: list = Field(default_factory=lambda: ["*"])
    allowed_hosts: list = Field(default_factory=lambda: ["*"])

    # Performance
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1 hour

    # Monitoring
    enable_metrics: bool = False
    metrics_port: int = 9090

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra environment variables

    def get_llm_model(self) -> str:
        """Get the current LLM model string based on provider."""
        if self.llm_provider == LLMProvider.ANTHROPIC:
            return self.anthropic_model.value
        elif self.llm_provider == LLMProvider.OPENAI:
            return self.openai_model.value
        else:
            raise ValueError(f"Unknown LLM provider: {self.llm_provider}")

    def get_llm_api_key(self) -> Optional[str]:
        """Get the API key for the current LLM provider."""
        if self.llm_provider == LLMProvider.ANTHROPIC:
            return self.anthropic_api_key
        elif self.llm_provider == LLMProvider.OPENAI:
            return self.openai_api_key
        else:
            raise ValueError(f"Unknown LLM provider: {self.llm_provider}")

    def validate_llm_config(self) -> bool:
        """Validate that LLM configuration is complete."""
        api_key = self.get_llm_api_key()
        if not api_key:
            raise ValueError(
                f"API key not found for provider {self.llm_provider}. "
                f"Please set {self.llm_provider.value.upper()}_API_KEY environment variable."
            )
        return True


# Global settings instance
settings = Settings()
