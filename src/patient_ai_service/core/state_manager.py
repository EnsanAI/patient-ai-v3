"""
State management for global and local agent states.

Supports both in-memory (development) and Redis (production) backends.
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from datetime import datetime

from patient_ai_service.models.state import (
    GlobalState,
    AppointmentAgentState,
    MedicalAgentState,
    EmergencyAgentState,
    RegistrationState,
    TranslationState,
    PatientProfile,
)
from .config import settings

logger = logging.getLogger(__name__)


class StateBackend(ABC):
    """Abstract backend for state storage."""

    @abstractmethod
    def get(self, key: str) -> Optional[str]:
        """Get value by key."""
        pass

    @abstractmethod
    def set(self, key: str, value: str, ttl: Optional[int] = None):
        """Set value with optional TTL."""
        pass

    @abstractmethod
    def delete(self, key: str):
        """Delete key."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass


class InMemoryBackend(StateBackend):
    """In-memory state storage for development."""

    def __init__(self):
        self._storage: Dict[str, str] = {}
        logger.info("Initialized in-memory state backend")

    def get(self, key: str) -> Optional[str]:
        return self._storage.get(key)

    def set(self, key: str, value: str, ttl: Optional[int] = None):
        self._storage[key] = value
        # TTL not implemented for in-memory (would need background cleanup)

    def delete(self, key: str):
        self._storage.pop(key, None)

    def exists(self, key: str) -> bool:
        return key in self._storage


class RedisBackend(StateBackend):
    """Redis state storage for production."""

    def __init__(self, redis_url: str):
        try:
            import redis
            self.client = redis.from_url(redis_url, decode_responses=True)
            self.client.ping()  # Test connection
            logger.info(f"Initialized Redis state backend: {redis_url}")
        except ImportError:
            raise ImportError("Redis package not installed. Install with: pip install redis")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def get(self, key: str) -> Optional[str]:
        return self.client.get(key)

    def set(self, key: str, value: str, ttl: Optional[int] = None):
        if ttl:
            self.client.setex(key, ttl, value)
        else:
            self.client.set(key, value)

    def delete(self, key: str):
        self.client.delete(key)

    def exists(self, key: str) -> bool:
        return self.client.exists(key) > 0


class StateManager:
    """
    Manages global and local agent states.

    Provides type-safe access to different state models with automatic
    serialization/deserialization and version control.
    """

    def __init__(self, backend: Optional[StateBackend] = None):
        if backend:
            self.backend = backend
        elif settings.redis_enabled:
            self.backend = RedisBackend(settings.redis_url)
        else:
            self.backend = InMemoryBackend()

        self.ttl = settings.session_ttl
        logger.info(f"StateManager initialized with {type(self.backend).__name__}")

    def _make_key(self, session_id: str, state_type: str) -> str:
        """Generate storage key."""
        return f"session:{session_id}:{state_type}"

    # Global State Management

    def get_global_state(self, session_id: str) -> GlobalState:
        """Get global state for a session."""
        key = self._make_key(session_id, "global_state")
        data = self.backend.get(key)

        if data:
            try:
                state_dict = json.loads(data)
                return GlobalState(**state_dict)
            except Exception as e:
                logger.error(f"Error deserializing global state: {e}")

        # Create new state if not exists
        logger.info(f"Creating new global state for session: {session_id}")
        return GlobalState(session_id=session_id)

    def update_global_state(self, session_id: str, **kwargs):
        """Update global state with provided fields."""
        state = self.get_global_state(session_id)

        # Update fields
        for key, value in kwargs.items():
            if hasattr(state, key):
                setattr(state, key, value)

        # Update metadata
        state.updated_at = datetime.utcnow()
        state.version += 1

        # Save
        self._save_state(session_id, "global_state", state)

    def update_patient_profile(self, session_id: str, **kwargs):
        """Update patient profile within global state."""
        state = self.get_global_state(session_id)

        # Update patient profile fields
        for key, value in kwargs.items():
            if hasattr(state.patient_profile, key):
                setattr(state.patient_profile, key, value)

        state.updated_at = datetime.utcnow()
        state.version += 1

        self._save_state(session_id, "global_state", state)

    # Appointment Agent State

    def get_appointment_state(self, session_id: str) -> AppointmentAgentState:
        """Get appointment agent state."""
        return self._get_local_state(
            session_id,
            "appointment_state",
            AppointmentAgentState
        )

    def update_appointment_state(self, session_id: str, **kwargs):
        """Update appointment agent state."""
        self._update_local_state(
            session_id,
            "appointment_state",
            AppointmentAgentState,
            **kwargs
        )

    # Medical Agent State

    def get_medical_state(self, session_id: str) -> MedicalAgentState:
        """Get medical agent state."""
        return self._get_local_state(
            session_id,
            "medical_state",
            MedicalAgentState
        )

    def update_medical_state(self, session_id: str, **kwargs):
        """Update medical agent state."""
        self._update_local_state(
            session_id,
            "medical_state",
            MedicalAgentState,
            **kwargs
        )

    # Emergency Agent State

    def get_emergency_state(self, session_id: str) -> EmergencyAgentState:
        """Get emergency agent state."""
        return self._get_local_state(
            session_id,
            "emergency_state",
            EmergencyAgentState
        )

    def update_emergency_state(self, session_id: str, **kwargs):
        """Update emergency agent state."""
        self._update_local_state(
            session_id,
            "emergency_state",
            EmergencyAgentState,
            **kwargs
        )

    # Registration Agent State

    def get_registration_state(self, session_id: str) -> RegistrationState:
        """Get registration agent state."""
        return self._get_local_state(
            session_id,
            "registration_state",
            RegistrationState
        )

    def update_registration_state(self, session_id: str, **kwargs):
        """Update registration agent state."""
        self._update_local_state(
            session_id,
            "registration_state",
            RegistrationState,
            **kwargs
        )

    # Translation State

    def get_translation_state(self, session_id: str) -> TranslationState:
        """Get translation state."""
        return self._get_local_state(
            session_id,
            "translation_state",
            TranslationState
        )

    def update_translation_state(self, session_id: str, **kwargs):
        """Update translation state."""
        self._update_local_state(
            session_id,
            "translation_state",
            TranslationState,
            **kwargs
        )

    # Execution Log Management

    def get_execution_log(self, session_id: str) -> Optional['ExecutionLog']:
        """
        Get execution log from state (for debugging/auditing).
        
        Note: Execution log is primarily in-memory during pipeline execution.
        This method retrieves persisted log if available.
        
        Args:
            session_id: Session identifier
            
        Returns:
            ExecutionLog if found, None otherwise
        """
        from patient_ai_service.models.validation import ExecutionLog
        return self._get_local_state(
            session_id,
            "execution_log",
            ExecutionLog
        )

    def save_execution_log(self, session_id: str, execution_log: 'ExecutionLog'):
        """
        Persist execution log to state (for debugging/auditing).
        
        Args:
            session_id: Session identifier
            execution_log: ExecutionLog to persist
        """
        from patient_ai_service.models.validation import ExecutionLog
        self._save_state(session_id, "execution_log", execution_log)

    def clear_execution_log(self, session_id: str):
        """
        Clear persisted execution log.
        
        Args:
            session_id: Session identifier
        """
        key = self._make_key(session_id, "execution_log")
        self.backend.delete(key)

    # Helper Methods

    def _get_local_state(self, session_id: str, state_type: str, model_class):
        """Generic method to get local state."""
        key = self._make_key(session_id, state_type)
        data = self.backend.get(key)

        if data:
            try:
                state_dict = json.loads(data)
                return model_class(**state_dict)
            except Exception as e:
                logger.error(f"Error deserializing {state_type}: {e}")

        # Return new instance
        return model_class()

    def _update_local_state(self, session_id: str, state_type: str, model_class, **kwargs):
        """Generic method to update local state."""
        state = self._get_local_state(session_id, state_type, model_class)

        # Update fields
        for key, value in kwargs.items():
            if hasattr(state, key):
                setattr(state, key, value)

        self._save_state(session_id, state_type, state)

    def _save_state(self, session_id: str, state_type: str, state_obj):
        """Save state object to backend."""
        key = self._make_key(session_id, state_type)
        data = json.dumps(state_obj.model_dump(), default=str)
        self.backend.set(key, data, ttl=self.ttl)

    # Utility Methods

    def get_agent_context(self, session_id: str, agent_name: str) -> Dict[str, Any]:
        """Get full context for an agent (global + local state)."""
        global_state = self.get_global_state(session_id)
        context = {
            "session_id": session_id,
            "patient_profile": global_state.patient_profile.model_dump(),
            "conversation_stage": global_state.conversation_stage,
            "detected_language": global_state.detected_language,
            "entities_collected": global_state.entities_collected,
        }

        # Add agent-specific state
        if agent_name == "appointment_manager":
            context["agent_state"] = self.get_appointment_state(session_id).model_dump()
        elif agent_name == "medical_inquiry":
            context["agent_state"] = self.get_medical_state(session_id).model_dump()
        elif agent_name == "emergency_response":
            context["agent_state"] = self.get_emergency_state(session_id).model_dump()
        elif agent_name == "registration":
            context["agent_state"] = self.get_registration_state(session_id).model_dump()
        elif agent_name == "translation":
            context["agent_state"] = self.get_translation_state(session_id).model_dump()

        return context

    def export_session(self, session_id: str) -> Dict[str, Any]:
        """Export all state for a session."""
        return {
            "global_state": self.get_global_state(session_id).model_dump(),
            "appointment_state": self.get_appointment_state(session_id).model_dump(),
            "medical_state": self.get_medical_state(session_id).model_dump(),
            "emergency_state": self.get_emergency_state(session_id).model_dump(),
            "registration_state": self.get_registration_state(session_id).model_dump(),
            "translation_state": self.get_translation_state(session_id).model_dump(),
        }

    def clear_session(self, session_id: str):
        """Clear all state for a session."""
        state_types = [
            "global_state",
            "appointment_state",
            "medical_state",
            "emergency_state",
            "registration_state",
            "translation_state",
        ]

        for state_type in state_types:
            key = self._make_key(session_id, state_type)
            self.backend.delete(key)

        logger.info(f"Cleared all state for session: {session_id}")


# Global state manager instance
_state_manager: Optional[StateManager] = None


def get_state_manager() -> StateManager:
    """Get or create the global state manager instance."""
    global _state_manager
    if _state_manager is None:
        _state_manager = StateManager()
    return _state_manager


def reset_state_manager():
    """Reset the global state manager (useful for testing)."""
    global _state_manager
    _state_manager = None
