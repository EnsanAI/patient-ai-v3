"""
Configuration management for the dental clinic system.
"""

import logging
import os
from typing import Optional, Literal, List
from pydantic import Field
from pydantic_settings import BaseSettings

from patient_ai_service.models.enums import LLMProvider, AnthropicModel, OpenAIModel

logger = logging.getLogger(__name__)


# =============================================================================
# ARCHITECTURE V2 SETTINGS
# =============================================================================

class ArchitectureV2Settings(BaseSettings):
    """Settings for the enhanced architecture."""
    
    # Feature flags
    enable_situation_assessor: bool = True  # DEPRECATED: Replaced by unified_reasoning
    enable_focused_handling: bool = True  # DEPRECATED: Replaced by unified_reasoning
    enable_task_plans: bool = True
    enable_entity_caching: bool = True
    enable_script_detection: bool = True
    enable_conversational_fast_path: bool = True
    enable_translation: bool = True  # When enabled, input messages are translated to English before processing
    enable_planning: bool = False  # Enable/disable agent plan generation for A/B testing

    # Tool Result Processing
    enable_tool_result_override: bool = False  # When enabled, FATAL/USER_INPUT/SYSTEM_ERROR override agentic workflow
    
    # Humanizer
    enable_humanizer: bool = True  # When enabled, suggested_response passes through _humanize_response for natural, conversational responses

    # Situation Assessor
    assessor_confidence_threshold: float = 0.7
    assessor_temperature: float = 0.1
    assessor_max_tokens: int = 500
    
    # Focused Handling
    focused_temperature: float = 0.2
    focused_max_tokens: int = 300
    
    # Task Plans
    task_plan_enabled_agents: List[str] = Field(
        default_factory=lambda: ["appointment_manager", "registration"]
    )
    max_task_retries: int = 2
    
    # Entity Caching
    availability_cache_ttl_seconds: int = 300  # 5 minutes
    doctor_info_cache_ttl_seconds: Optional[int] = None  # Never expires
    
    # Fallback behavior
    fallback_to_comprehensive_on_error: bool = True
    
    class Config:
        env_prefix = "CAREBOT_V2_"


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

    # LLM Configuration - Global Defaults
    # DEPRECATED: These fields are kept for backward compatibility only.
    # LLM configuration now comes from config/llm_config.yaml
    # These will be removed in a future version.
    llm_provider: LLMProvider = Field(
        LLMProvider.ANTHROPIC,
        alias="LLM_PROVIDER",
        description="DEPRECATED: Use config/llm_config.yaml instead"
    )
    anthropic_model: AnthropicModel = Field(
        AnthropicModel.CLAUDE_SONNET_4_5,
        alias="ANTHROPIC_MODEL",
        description="DEPRECATED: Use config/llm_config.yaml instead"
    )
    openai_model: OpenAIModel = Field(
        OpenAIModel.GPT_4O_MINI,
        alias="OPENAI_MODEL",
        description="DEPRECATED: Use config/llm_config.yaml instead"
    )
    llm_max_tokens: int = Field(
        1000,
        description="DEPRECATED: Use config/llm_config.yaml instead"
    )
    llm_temperature: float = Field(
        0.7,
        description="DEPRECATED: Use config/llm_config.yaml instead"
    )
    llm_timeout: int = Field(
        30,
        description="DEPRECATED: Use config/llm_config.yaml instead"
    )

    # Component-Specific Model Overrides (optional, None = use global)
    reasoning_model_override: Optional[str] = Field(None, alias="REASONING_MODEL")
    translation_model_override: Optional[str] = Field(None, alias="TRANSLATION_MODEL")
    intent_router_model_override: Optional[str] = Field(None, alias="INTENT_ROUTER_MODEL")
    agent_model_override: Optional[str] = Field(None, alias="AGENT_MODEL")

    # Component-Specific Temperature Settings
    reasoning_temperature: float = Field(
        default=0.3,
        alias="REASONING_TEMPERATURE",
        description="Temperature for intent routing and reasoning (low for consistency)"
    )
    translation_temperature: float = Field(
        default=0.1,
        alias="TRANSLATION_TEMPERATURE",
        description="Temperature for translation and language detection (very low for accuracy)"
    )
    intent_router_temperature: float = Field(
        default=0.3,
        alias="INTENT_ROUTER_TEMPERATURE",
        description="Temperature for intent classification (low for consistent routing)"
    )
    agent_temperature: float = Field(
        default=0.7,
        alias="AGENT_TEMPERATURE",
        description="Temperature for agent responses and tool calling (higher for natural responses)"
    )

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
        default=False,  # DISABLED: Causes hallucinations and loops
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

    # Finalization (Two-Layer Quality Control)
    enable_finalization: bool = Field(
        default=False,  # DISABLED: Causes hallucinations and loops
        description="Enable finalization layer (final quality check before sending)"
    )
    finalization_temperature: float = Field(
        default=0.3,
        description="LLM temperature for finalization (slightly higher for natural edits)"
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

    # Observability
    enable_observability: bool = Field(
        default=True,
        description="Enable detailed observability logging"
    )
    observability_output_format: Literal["json", "structured", "detailed"] = Field(
        default="structured",
        description="Output format for observability logs"
    )
    observability_log_to_file: Optional[str] = Field(
        default=None,
        description="Optional file path for observability logs (None = stdout only)"
    )
    cost_tracking_enabled: bool = Field(
        default=True,
        description="Enable cost tracking for LLM calls"
    )
    
    # Prompt caching configuration
    prompt_cache_enabled: bool = Field(
        default=False,  # Start disabled
        env="PROMPT_CACHE_ENABLED",
        description="Enable prompt caching for Anthropic models"
    )
    
    # Architecture V2
    v2: ArchitectureV2Settings = Field(
        default_factory=ArchitectureV2Settings,
        description="Architecture V2 settings"
    )

    # Tool execution limits (PHASE 1)
    max_tool_iterations: int = Field(
        default=10,
        description="Default maximum tool iterations per turn"
    )
    max_tool_iterations_hard_limit: int = Field(
        default=25,
        description="Hard limit for tool iterations (safety ceiling)"
    )

    # Circuit breaker settings (PHASE 3)
    circuit_breaker_failure_threshold: int = Field(
        default=3,
        description="Number of consecutive failures before circuit opens"
    )
    circuit_breaker_recovery_timeout_seconds: float = Field(
        default=30.0,
        description="Seconds to wait before testing recovery"
    )

    # Multi-booking settings (PHASE 4)
    multi_booking_rollback_on_partial_failure: bool = Field(
        default=True,
        description="Automatically rollback on partial multi-booking failure"
    )
    multi_booking_min_success_ratio: float = Field(
        default=1.0,
        description="Minimum success ratio for multi-booking (1.0 = all must succeed)"
    )

    # Timeout settings
    booking_overall_timeout_seconds: float = Field(
        default=20.0,
        description="Overall timeout for booking workflow"
    )
    booking_operation_timeout_seconds: float = Field(
        default=10.0,
        description="Per-operation timeout for booking"
    )
    db_verification_max_retries: int = Field(
        default=3,
        description="Maximum retries for DB verification"
    )

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

    def get_component_model(self, component: str) -> str:
        """
        Get the model for a specific component, with fallback to global.

        Args:
            component: Component name (e.g., 'reasoning', 'translation', 'agent')

        Returns:
            Model string to use for this component
        """
        override_attr = f"{component}_model_override"
        override = getattr(self, override_attr, None)

        if override:
            return override

        # Fallback to global model
        return self.get_llm_model()

    def get_component_temperature(self, component: str) -> float:
        """
        Get the temperature for a specific component, with fallback to global.

        Args:
            component: Component name (e.g., 'reasoning', 'translation', 'agent')

        Returns:
            Temperature to use for this component
        """
        temp_attr = f"{component}_temperature"
        return getattr(self, temp_attr, self.llm_temperature)
    
    def get_api_key_for_provider(self, provider: str) -> Optional[str]:
        """
        Get API key for a specific provider.
        
        Used by LLMConfigManager to retrieve API keys for different providers.
        
        Args:
            provider: Provider name ('anthropic' or 'openai')
        
        Returns:
            API key string or None if not set
        """
        if provider.lower() == "anthropic":
            return self.anthropic_api_key
        elif provider.lower() == "openai":
            return self.openai_api_key
        return None
    
    def validate_on_startup(self):
        """
        Validate configuration on startup and warn about deprecated settings.
        
        This method should be called during application startup to detect
        deprecated environment variables and log warnings.
        """
        # Warn if old LLM_PROVIDER env var is set
        if self.llm_provider:
            logger.warning(
                "⚠️  LLM_PROVIDER in .env is DEPRECATED. "
                "Please use config/llm_config.yaml instead. "
                "See patient-ai-v3/config/llm_config.yaml for configuration."
            )
        
        # Warn about other deprecated LLM config vars (if they differ from defaults)
        if hasattr(self, 'anthropic_model') and self.anthropic_model != AnthropicModel.CLAUDE_SONNET_4_5:
            logger.warning(
                "⚠️  ANTHROPIC_MODEL in .env is DEPRECATED. "
                "Please use config/llm_config.yaml instead."
            )
        
        if hasattr(self, 'openai_model') and self.openai_model != OpenAIModel.GPT_4O_MINI:
            logger.warning(
                "⚠️  OPENAI_MODEL in .env is DEPRECATED. "
                "Please use config/llm_config.yaml instead."
            )


# Global settings instance
settings = Settings()
