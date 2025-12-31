"""
Hierarchical LLM Configuration Manager.

Provides function → agent → global fallback for LLM configuration.
Single source of truth: config/llm_config.yaml
API keys remain in .env for security.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Any
import yaml

from .config import settings
from .llm import LLMClient, LLMFactory
from patient_ai_service.models.enums import LLMProvider

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMConfig:
    """Immutable LLM configuration."""
    provider: str
    model: str
    temperature: float
    max_tokens: int
    timeout: int
    prompt_cache_enabled: bool = False  # Prompt caching (Anthropic only)

    # Extended Thinking (Anthropic only)
    extended_thinking_enabled: bool = False
    extended_thinking_budget: int = 2048  # Default budget (min 1024)

    def __post_init__(self):
        """Validate config values."""
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"Temperature must be between 0.0 and 2.0, got {self.temperature}")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if self.timeout < 1:
            raise ValueError(f"timeout must be >= 1, got {self.timeout}")

        # Validate extended thinking budget
        if self.extended_thinking_enabled and self.extended_thinking_budget < 1024:
            raise ValueError(
                f"extended_thinking_budget must be >= 1024, got {self.extended_thinking_budget}"
            )


class LLMConfigManager:
    """
    Manages hierarchical LLM configuration with client caching.
    
    Resolution order: function → agent → global
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the config manager.
        
        Args:
            config_path: Path to llm_config.yaml (defaults to config/llm_config.yaml)
        """
        self._config_path = config_path or self._find_config_file()
        self._yaml_config: Dict[str, Any] = {}
        self._client_cache: Dict[str, LLMClient] = {}
        self._load_config()
        self._validate_config()
        
        # Validate API keys if configured
        if self._yaml_config.get("validation", {}).get("validate_api_keys_on_startup", False):
            self._validate_api_keys()
        else:
            self._warn_missing_api_keys()
    
    def _find_config_file(self) -> str:
        """Find the config file relative to the project root."""
        # Try multiple possible locations
        possible_paths = [
            # From src/patient_ai_service/core/ -> config/llm_config.yaml
            Path(__file__).parent.parent.parent.parent / "config" / "llm_config.yaml",
            # From project root
            Path(__file__).parent.parent.parent.parent.parent / "config" / "llm_config.yaml",
            # Absolute fallback
            Path("/app/config/llm_config.yaml"),  # Docker container path
        ]
        
        for path in possible_paths:
            if path.exists():
                logger.info(f"✅ [LLMConfig] Found LLM config at: {path}")
                logger.info(f"📁 [LLMConfig] Config file exists: {path.exists()}, absolute: {path.absolute()}")
                return str(path)
        
        # Default to expected location
        default_path = Path(__file__).parent.parent.parent.parent / "config" / "llm_config.yaml"
        logger.warning(
            f"⚠️  [LLMConfig] Config file not found in any of the expected locations. "
            f"Using default: {default_path} (exists: {default_path.exists()})"
        )
        return str(default_path)
    
    def _load_config(self):
        """Load YAML configuration file."""
        try:
            if os.path.exists(self._config_path):
                with open(self._config_path, 'r') as f:
                    self._yaml_config = yaml.safe_load(f) or {}
                logger.info(f"✅ [LLMConfig] Loaded LLM config from {self._config_path}")
                
                # Log key config values for debugging
                if "agents" in self._yaml_config:
                    logger.info(f"📋 [LLMConfig] Found {len(self._yaml_config['agents'])} agents in config")
                    if "unified_reasoning" in self._yaml_config["agents"]:
                        ur_config = self._yaml_config["agents"]["unified_reasoning"]
                        logger.info(
                            f"🔍 [LLMConfig] unified_reasoning config: "
                            f"provider={ur_config.get('provider', 'NOT SET')}, "
                            f"model={ur_config.get('model', 'NOT SET')}"
                        )
            else:
                logger.warning(
                    f"⚠️  [LLMConfig] LLM config file not found at {self._config_path}. "
                    f"Using defaults from Settings class."
                )
                self._yaml_config = {}
        except Exception as e:
            logger.error(f"❌ [LLMConfig] Failed to load LLM config from {self._config_path}: {e}")
            logger.warning("Falling back to Settings class defaults")
            self._yaml_config = {}
    
    def _validate_config(self):
        """Validate YAML configuration structure."""
        if not self._yaml_config:
            return
        
        # Check for required top-level keys
        if "global" not in self._yaml_config:
            logger.warning("No 'global' section in YAML config, will use Settings defaults")
        
        # Validate global section if present
        if "global" in self._yaml_config:
            global_config = self._yaml_config["global"]
            if "provider" not in global_config:
                logger.warning("Global config missing 'provider', will use Settings default")
            if "model" not in global_config:
                logger.warning("Global config missing 'model', will use Settings default")
    
    def _validate_api_keys(self):
        """Validate that API keys exist for all providers in config."""
        providers = self._get_all_providers()
        missing = []
        
        for provider in providers:
            api_key = settings.get_api_key_for_provider(provider)
            if not api_key:
                missing.append(provider)
        
        if missing:
            raise ValueError(
                f"Missing API keys for providers: {missing}. "
                f"Please set {', '.join(f'{p.upper()}_API_KEY' for p in missing)} environment variables."
            )
    
    def _warn_missing_api_keys(self):
        """Warn about missing API keys without failing."""
        providers = self._get_all_providers()
        
        for provider in providers:
            api_key = settings.get_api_key_for_provider(provider)
            if not api_key:
                logger.warning(
                    f"⚠️  API key not set for provider '{provider}'. "
                    f"Set {provider.upper()}_API_KEY environment variable."
                )
    
    def _get_all_providers(self) -> set:
        """Get all providers referenced in YAML config."""
        providers = set()
        
        # Check global
        if "global" in self._yaml_config:
            if "provider" in self._yaml_config["global"]:
                providers.add(self._yaml_config["global"]["provider"])
        
        # Check agents
        if "agents" in self._yaml_config:
            for agent_name, agent_config in self._yaml_config["agents"].items():
                if isinstance(agent_config, dict) and "provider" in agent_config:
                    providers.add(agent_config["provider"])
                # Check functions
                if isinstance(agent_config, dict) and "functions" in agent_config:
                    for func_name, func_config in agent_config["functions"].items():
                        if isinstance(func_config, dict) and "provider" in func_config:
                            providers.add(func_config["provider"])
        
        return providers
    
    def get_config(
        self,
        agent_name: Optional[str] = None,
        function_name: Optional[str] = None
    ) -> LLMConfig:
        """
        Get LLM configuration with hierarchical resolution.
        
        Resolution order:
        1. Function-level config (if function_name and agent_name provided)
        2. Agent-level config (if agent_name provided)
        3. Global config
        
        Args:
            agent_name: Agent name (e.g., "appointment_manager")
            function_name: Function name (e.g., "_think")
        
        Returns:
            LLMConfig with resolved values
        """
        # Normalize agent name
        if agent_name:
            agent_name = self._normalize_agent_name(agent_name)
        
        # Start with global defaults
        global_config = self._yaml_config.get("global", {})
        provider_defaults = self._yaml_config.get("provider_defaults", {})
        
        # Get global values (with fallback to Settings)
        provider = global_config.get("provider") or settings.llm_provider.value
        model = global_config.get("model")

        # Get prompt cache configuration from global.prompt_cache.enabled
        prompt_cache_config = global_config.get("prompt_cache", {})
        prompt_cache_enabled = prompt_cache_config.get("enabled", False)

        # Get extended thinking configuration from global.extended_thinking
        extended_thinking_config = global_config.get("extended_thinking", {})
        extended_thinking_enabled = extended_thinking_config.get("enabled", False)
        extended_thinking_budget = extended_thinking_config.get("budget_tokens", 2048)

        if not model:
            # Use provider default or Settings
            if provider in provider_defaults:
                model = provider_defaults[provider].get("model")
            if not model:
                model = settings.get_llm_model()
        
        temperature = global_config.get("temperature", settings.llm_temperature)
        max_tokens = global_config.get("max_tokens", settings.llm_max_tokens)
        timeout = global_config.get("timeout", settings.llm_timeout)
        
        # Override with agent-level config if provided
        if agent_name and "agents" in self._yaml_config:
            agent_config = self._yaml_config["agents"].get(agent_name)
            if agent_config and isinstance(agent_config, dict):
                logger.debug(
                    f"🔧 [LLMConfig] Processing agent-level config for '{agent_name}': "
                    f"provider={agent_config.get('provider', 'NOT SET')}, "
                    f"model={agent_config.get('model', 'NOT SET')}"
                )
                
                if "provider" in agent_config:
                    old_provider = provider
                    provider = agent_config["provider"]
                    logger.debug(
                        f"🔧 [LLMConfig] Agent '{agent_name}' provider override: "
                        f"{old_provider} → {provider}"
                    )
                    # Reset model if provider changed
                    if provider in provider_defaults:
                        model = provider_defaults[provider].get("model")
                        logger.debug(
                            f"🔧 [LLMConfig] Reset model to provider default: {model}"
                        )
                    if not model:
                        model = settings.get_llm_model()
                        logger.debug(
                            f"🔧 [LLMConfig] Using Settings default model: {model}"
                        )
                
                if "model" in agent_config:
                    old_model = model
                    model = agent_config["model"]
                    logger.debug(
                        f"🔧 [LLMConfig] Agent '{agent_name}' model override: "
                        f"{old_model} → {model}"
                    )
                
                if "temperature" in agent_config:
                    temperature = agent_config["temperature"]
                
                if "max_tokens" in agent_config:
                    max_tokens = agent_config["max_tokens"]
                
                if "timeout" in agent_config:
                    timeout = agent_config["timeout"]

                # Agent-level extended thinking override
                if "extended_thinking" in agent_config:
                    agent_et_config = agent_config["extended_thinking"]
                    if "enabled" in agent_et_config:
                        extended_thinking_enabled = agent_et_config["enabled"]
                    if "budget_tokens" in agent_et_config:
                        extended_thinking_budget = agent_et_config["budget_tokens"]

                # Override with function-level config if provided
                if function_name and "functions" in agent_config:
                    func_config = agent_config["functions"].get(function_name)
                    if func_config and isinstance(func_config, dict):
                        logger.debug(
                            f"🔧 [LLMConfig] Processing function-level config for "
                            f"'{agent_name}.{function_name}': "
                            f"provider={func_config.get('provider', 'NOT SET')}, "
                            f"model={func_config.get('model', 'NOT SET')}"
                        )
                        
                        if "provider" in func_config:
                            old_provider = provider
                            provider = func_config["provider"]
                            logger.debug(
                                f"🔧 [LLMConfig] Function '{function_name}' provider override: "
                                f"{old_provider} → {provider}"
                            )
                            # Reset model if provider changed
                            if provider in provider_defaults:
                                model = provider_defaults[provider].get("model")
                                logger.debug(
                                    f"🔧 [LLMConfig] Reset model to provider default: {model}"
                                )
                            if not model:
                                model = settings.get_llm_model()
                                logger.debug(
                                    f"🔧 [LLMConfig] Using Settings default model: {model}"
                                )
                        
                        if "model" in func_config:
                            old_model = model
                            model = func_config["model"]
                            logger.debug(
                                f"🔧 [LLMConfig] Function '{function_name}' model override: "
                                f"{old_model} → {model}"
                            )
                        
                        if "temperature" in func_config:
                            temperature = func_config["temperature"]
                        
                        if "max_tokens" in func_config:
                            max_tokens = func_config["max_tokens"]
                        
                        if "timeout" in func_config:
                            timeout = func_config["timeout"]

                        # Function-level extended thinking override
                        if "extended_thinking" in func_config:
                            func_et_config = func_config["extended_thinking"]
                            if "enabled" in func_et_config:
                                extended_thinking_enabled = func_et_config["enabled"]
                            if "budget_tokens" in func_et_config:
                                extended_thinking_budget = func_et_config["budget_tokens"]

        resolved_config = LLMConfig(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            prompt_cache_enabled=prompt_cache_enabled,
            extended_thinking_enabled=extended_thinking_enabled,
            extended_thinking_budget=extended_thinking_budget,
        )
        
        # Debug logging for config resolution
        logger.debug(
            f"🔧 [LLMConfig] Resolved config for agent='{agent_name}', function='{function_name}': "
            f"provider={resolved_config.provider}, model={resolved_config.model}, "
            f"temperature={resolved_config.temperature}, max_tokens={resolved_config.max_tokens}"
        )
        
        # Special logging for unified_reasoning to help diagnose issues (INFO level for visibility)
        if agent_name == "unified_reasoning":
            logger.info(
                f"🔍 [LLMConfig] UnifiedReasoning config resolved for function='{function_name}': "
                f"provider={resolved_config.provider}, model={resolved_config.model}, "
                f"temperature={resolved_config.temperature}, max_tokens={resolved_config.max_tokens}"
            )
        
        return resolved_config
    
    def get_client(
        self,
        agent_name: Optional[str] = None,
        function_name: Optional[str] = None
    ) -> LLMClient:
        """
        Get or create cached LLM client for the given agent/function.
        
        Clients are cached by provider+model combination.
        
        Args:
            agent_name: Agent name (e.g., "appointment_manager")
            function_name: Function name (e.g., "_think")
        
        Returns:
            Cached LLMClient instance
        """
        config = self.get_config(agent_name, function_name)
        
        # Cache key: provider:model
        cache_key = f"{config.provider}:{config.model}"
        
        if cache_key not in self._client_cache:
            # Get API key for provider
            api_key = settings.get_api_key_for_provider(config.provider)
            if not api_key:
                raise ValueError(
                    f"API key not found for provider '{config.provider}'. "
                    f"Please set {config.provider.upper()}_API_KEY environment variable."
                )
            
            # Create client using LLMFactory
            provider_enum = LLMProvider(config.provider)
            client = LLMFactory.create_client(
                provider=provider_enum,
                api_key=api_key,
                model=config.model
            )
            
            self._client_cache[cache_key] = client
            logger.info(
                f"✅ [LLMConfig] Created NEW LLM client: {cache_key} "
                f"(agent='{agent_name}', function='{function_name}')"
            )
        else:
            logger.debug(
                f"♻️  [LLMConfig] Using CACHED LLM client: {cache_key} "
                f"(agent='{agent_name}', function='{function_name}')"
            )
        
        return self._client_cache[cache_key]
    
    @staticmethod
    def _normalize_agent_name(agent_name: str) -> str:
        """
        Normalize agent name from CamelCase to snake_case.
        
        Examples:
            "AppointmentManagerAgent" → "appointment_manager"
            "MedicalInquiryAgent" → "medical_inquiry"
            "TranslationAgent" → "translation"
            "appointment_manager" → "appointment_manager" (no change)
        """
        if not agent_name:
            return agent_name
        
        # If already snake_case, return as-is
        if '_' in agent_name and agent_name.islower():
            return agent_name
        
        # Remove common suffixes
        agent_name = re.sub(r'Agent$', '', agent_name)
        agent_name = re.sub(r'Engine$', '', agent_name)
        
        # Convert CamelCase to snake_case
        # Insert underscore before uppercase letters (except first)
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', agent_name)
        s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
        
        return s2.lower()


# Module-level singleton
_config_manager: Optional[LLMConfigManager] = None


def get_llm_config_manager() -> LLMConfigManager:
    """Get or create the global LLMConfigManager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = LLMConfigManager()
    return _config_manager


def get_llm_config(
    agent_name: Optional[str] = None,
    function_name: Optional[str] = None
) -> LLMConfig:
    """Convenience function to get LLM config."""
    return get_llm_config_manager().get_config(agent_name, function_name)


def get_hierarchical_llm_client(
    agent_name: Optional[str] = None,
    function_name: Optional[str] = None
) -> LLMClient:
    """Convenience function to get hierarchically-configured LLM client."""
    return get_llm_config_manager().get_client(agent_name, function_name)


def reset_llm_config_manager():
    """Reset the global config manager (useful for testing)."""
    global _config_manager
    _config_manager = None


