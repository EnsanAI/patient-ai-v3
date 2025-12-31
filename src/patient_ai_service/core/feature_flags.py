"""
Feature Flags for Architecture V2

Provides runtime control over new features for gradual rollout.
"""

import logging
from typing import Optional, Dict, Any
from patient_ai_service.core.config import settings

logger = logging.getLogger(__name__)


class FeatureFlags:
    """
    Runtime feature flag management.
    
    Allows enabling/disabling features without restart.
    """
    
    _overrides: Dict[str, bool] = {}
    
    @classmethod
    def is_enabled(cls, feature: str, session_id: Optional[str] = None) -> bool:
        """
        Check if a feature is enabled.
        
        Priority:
        1. Runtime overrides
        2. Session-specific flags (for A/B testing)
        3. Config settings
        """
        # Check runtime override
        if feature in cls._overrides:
            return cls._overrides[feature]
        
        # Check session-specific (could be from Redis for A/B testing)
        if session_id:
            session_flag = cls._get_session_flag(session_id, feature)
            if session_flag is not None:
                return session_flag
        
        # Fall back to config
        return cls._get_config_flag(feature)
    
    @classmethod
    def _get_config_flag(cls, feature: str) -> bool:
        """Get feature flag from config."""
        flag_map = {
            "situation_assessor": settings.v2.enable_situation_assessor,
            "focused_handling": settings.v2.enable_focused_handling,
            "task_plans": settings.v2.enable_task_plans,
            "entity_caching": settings.v2.enable_entity_caching,
            "script_detection": settings.v2.enable_script_detection,
            "conversational_fast_path": settings.v2.enable_conversational_fast_path,
            "translation": settings.v2.enable_translation,
            "planning": settings.v2.enable_planning,
            "tool_result_override": settings.v2.enable_tool_result_override,
            "humanizer": getattr(settings.v2, 'enable_humanizer', False),
            "output_translation": getattr(settings.v2, 'enable_output_translation', True),
        }
        return flag_map.get(feature, False)
    
    @classmethod
    def _get_session_flag(cls, session_id: str, feature: str) -> Optional[bool]:
        """Get session-specific flag (for A/B testing)."""
        # Could check Redis or other store here
        return None
    
    @classmethod
    def set_override(cls, feature: str, enabled: bool):
        """Set runtime override for a feature."""
        cls._overrides[feature] = enabled
        logger.info(f"Feature flag override: {feature}={enabled}")
    
    @classmethod
    def clear_override(cls, feature: str):
        """Clear runtime override."""
        if feature in cls._overrides:
            del cls._overrides[feature]
            logger.info(f"Feature flag override cleared: {feature}")
    
    @classmethod
    def clear_all_overrides(cls):
        """Clear all runtime overrides."""
        cls._overrides.clear()
        logger.info("All feature flag overrides cleared")


# Convenience functions
def is_assessor_enabled(session_id: str = None) -> bool:
    return FeatureFlags.is_enabled("situation_assessor", session_id)

def is_focused_enabled(session_id: str = None) -> bool:
    return FeatureFlags.is_enabled("focused_handling", session_id)

def is_task_plans_enabled(session_id: str = None) -> bool:
    return FeatureFlags.is_enabled("task_plans", session_id)

def is_entity_caching_enabled(session_id: str = None) -> bool:
    return FeatureFlags.is_enabled("entity_caching", session_id)

def is_script_detection_enabled(session_id: str = None) -> bool:
    return FeatureFlags.is_enabled("script_detection", session_id)

def is_conversational_fast_path_enabled(session_id: str = None) -> bool:
    return FeatureFlags.is_enabled("conversational_fast_path", session_id)

def is_planning_enabled(session_id: str = None) -> bool:
    """Check if agent planning is enabled for A/B testing."""
    return FeatureFlags.is_enabled("planning", session_id)

def is_tool_result_override_enabled(session_id: str = None) -> bool:
    """Check if tool result type override is enabled (FATAL/USER_INPUT bypass LLM)."""
    return FeatureFlags.is_enabled("tool_result_override", session_id)

def is_humanizer_enabled(session_id: str = None) -> bool:
    """Check if humanizer is enabled (suggested_response → _humanize_response)."""
    return FeatureFlags.is_enabled("humanizer", session_id)

def is_translation_enabled(session_id: str = None) -> bool:
    """Check if translation is enabled."""
    return FeatureFlags.is_enabled("translation", session_id)

def is_output_translation_enabled(session_id: str = None) -> bool:
    """Check if output translation is enabled (agent response → _translate_output)."""
    return FeatureFlags.is_enabled("output_translation", session_id)

