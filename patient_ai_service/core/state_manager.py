"""
State management for global and local agent states.

Supports both in-memory (development) and Redis (production) backends.
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field

from patient_ai_service.models.state import (
    GlobalState,
    AppointmentAgentState,
    MedicalAgentState,
    EmergencyAgentState,
    RegistrationState,
    TranslationState,
    PatientProfile,
)
from patient_ai_service.models.agentic import CriterionState
from .config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# ENHANCED STATE MODELS
# =============================================================================

class BlockedCriterion(BaseModel):
    """
    Represents a success criterion that is blocked pending user input.
    """
    criterion_id: str
    description: str
    blocked_reason: str
    blocked_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Options to present to user
    options: List[Any] = Field(default_factory=list)
    
    # Context needed to resume
    resume_context: Dict[str, Any] = Field(default_factory=dict)
    
    # Suggested response text
    suggested_response: Optional[str] = None


class ContinuationContext(BaseModel):
    """
    Context stored between turns for resuming blocked flows.
    """
    # What we're waiting for
    awaiting: str = ""  # "time_selection", "doctor_selection", "confirmation", etc.
    
    # Options we presented
    presented_options: List[Any] = Field(default_factory=list)
    
    # What the user originally requested
    original_request: Optional[str] = None
    
    # Partial progress we've made
    resolved_entities: Dict[str, Any] = Field(default_factory=dict)
    # Example: {"doctor_id": "uuid-123", "date": "2025-12-06", "procedure": "cleaning"}
    
    # Blocked criteria
    blocked_criteria: List[str] = Field(default_factory=list)
    
    # When this context was created
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # How many turns we've been waiting
    waiting_turns: int = 0


class AgenticExecutionState(BaseModel):
    """
    Enhanced state for tracking agentic loop execution.
    
    This is the complete state model - replaces the simpler version.
    """
    # Session tracking
    session_id: Optional[str] = None
    
    # Loop control
    iteration: int = 0
    max_iterations: int = 15
    
    # Task context from reasoning
    task_context: Optional[Dict[str, Any]] = None
    
    # Criteria tracking with full state
    criteria: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    # Format: {
    #     "criterion_0": {
    #         "description": "root canal appointment booked",
    #         "state": "pending",
    #         "blocked_reason": null,
    #         "blocked_options": null,
    #         "completion_evidence": null
    #     }
    # }
    
    # Simple lists for quick access
    success_criteria: List[str] = Field(default_factory=list)
    completed_criteria: List[str] = Field(default_factory=list)
    blocked_criteria: List[str] = Field(default_factory=list)
    failed_criteria: List[str] = Field(default_factory=list)
    
    # Blocked details
    blocked_details: Dict[str, BlockedCriterion] = Field(default_factory=dict)
    
    # Continuation context
    continuation_context: ContinuationContext = Field(default_factory=ContinuationContext)
    
    # User options pending selection
    pending_user_options: List[Any] = Field(default_factory=list)
    
    # Observations history
    observations: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Decision history
    decision_history: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Status
    status: str = "pending"  # pending, in_progress, blocked, complete, failed, max_iterations
    failure_reason: Optional[str] = None
    
    # Timestamps
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    last_updated_at: Optional[datetime] = None
    blocked_at: Optional[datetime] = None
    
    # Metrics
    total_llm_calls: int = 0
    total_tool_calls: int = 0


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

    # Agentic State Management

    def get_agentic_state(self, session_id: str) -> AgenticExecutionState:
        """
        Get agentic execution state for current request.
        """
        return self._get_local_state(
            session_id,
            "agentic_state",
            AgenticExecutionState
        )

    def update_agentic_state(self, session_id: str, **kwargs):
        """
        Update agentic execution state.
        """
        state = self.get_agentic_state(session_id)

        for key, value in kwargs.items():
            if hasattr(state, key):
                setattr(state, key, value)

        state.last_updated_at = datetime.utcnow()
        self._save_state(session_id, "agentic_state", state)

    def initialize_agentic_state(
        self,
        session_id: str,
        task_context: Dict[str, Any],
        max_iterations: int = 15
    ):
        """
        Initialize agentic state at start of request processing.
        """
        success_criteria = task_context.get("success_criteria", [])

        # Build criteria dict
        criteria = {}
        for i, desc in enumerate(success_criteria):
            criteria[f"criterion_{i}"] = {
                "description": desc,
                "state": CriterionState.PENDING.value,
                "blocked_reason": None,
                "blocked_options": None,
                "completion_evidence": None
            }

        state = AgenticExecutionState(
            session_id=session_id,
            iteration=0,
            max_iterations=max_iterations,
            task_context=task_context,
            criteria=criteria,
            success_criteria=success_criteria,
            completed_criteria=[],
            blocked_criteria=[],
            failed_criteria=[],
            status="in_progress",
            started_at=datetime.utcnow(),
            last_updated_at=datetime.utcnow()
        )

        self._save_state(session_id, "agentic_state", state)
        logger.info(f"Initialized agentic state for {session_id}: {len(success_criteria)} criteria")

    def reset_agentic_state(self, session_id: str):
        """
        Reset agentic state for new request.
        """
        key = self._make_key(session_id, "agentic_state")
        self.backend.delete(key)
        logger.debug(f"Reset agentic state for session {session_id}")

    # Criteria Management

    def mark_criterion_complete(
        self,
        session_id: str,
        description: str,
        evidence: Optional[str] = None
    ):
        """
        Mark a success criterion as complete.
        """
        state = self.get_agentic_state(session_id)

        # Find and update criterion
        for crit_id, crit_data in state.criteria.items():
            if description.lower() in crit_data["description"].lower():
                crit_data["state"] = CriterionState.COMPLETE.value
                crit_data["completion_evidence"] = evidence

                # Update quick access list
                if crit_data["description"] not in state.completed_criteria:
                    state.completed_criteria.append(crit_data["description"])

                # Remove from blocked if was blocked
                if crit_data["description"] in state.blocked_criteria:
                    state.blocked_criteria.remove(crit_data["description"])

                logger.info(f"✅ Criterion complete: {crit_data['description']}")
                break

        state.last_updated_at = datetime.utcnow()
        self._save_state(session_id, "agentic_state", state)

    def mark_criterion_blocked(
        self,
        session_id: str,
        description: str,
        reason: str,
        options: List[Any] = None,
        resume_context: Dict[str, Any] = None
    ):
        """
        Mark a criterion as blocked pending user input.
        """
        state = self.get_agentic_state(session_id)

        # Find and update criterion
        for crit_id, crit_data in state.criteria.items():
            if description.lower() in crit_data["description"].lower():
                crit_data["state"] = CriterionState.BLOCKED.value
                crit_data["blocked_reason"] = reason
                crit_data["blocked_options"] = options

                # Update quick access list
                if crit_data["description"] not in state.blocked_criteria:
                    state.blocked_criteria.append(crit_data["description"])

                # Store blocked details
                state.blocked_details[crit_id] = BlockedCriterion(
                    criterion_id=crit_id,
                    description=crit_data["description"],
                    blocked_reason=reason,
                    options=options or [],
                    resume_context=resume_context or {}
                ).model_dump()

                logger.info(f"⏸️ Criterion blocked: {crit_data['description']} - {reason}")
                break

        # Update status
        state.status = "blocked"
        state.blocked_at = datetime.utcnow()
        state.last_updated_at = datetime.utcnow()

        # Store pending options
        if options:
            state.pending_user_options = options

        self._save_state(session_id, "agentic_state", state)

    def mark_criterion_failed(
        self,
        session_id: str,
        description: str,
        reason: str
    ):
        """
        Mark a criterion as failed.
        """
        state = self.get_agentic_state(session_id)

        for crit_id, crit_data in state.criteria.items():
            if description.lower() in crit_data["description"].lower():
                crit_data["state"] = CriterionState.FAILED.value
                crit_data["failed_reason"] = reason

                if crit_data["description"] not in state.failed_criteria:
                    state.failed_criteria.append(crit_data["description"])

                logger.warning(f"❌ Criterion failed: {crit_data['description']} - {reason}")
                break

        state.last_updated_at = datetime.utcnow()
        self._save_state(session_id, "agentic_state", state)

    def unblock_criterion(
        self,
        session_id: str,
        description: str
    ):
        """
        Unblock a criterion (e.g., after user provides input).
        """
        state = self.get_agentic_state(session_id)

        for crit_id, crit_data in state.criteria.items():
            if description.lower() in crit_data["description"].lower():
                crit_data["state"] = CriterionState.IN_PROGRESS.value
                crit_data["blocked_reason"] = None
                crit_data["blocked_options"] = None

                if crit_data["description"] in state.blocked_criteria:
                    state.blocked_criteria.remove(crit_data["description"])

                # Remove from blocked details
                if crit_id in state.blocked_details:
                    del state.blocked_details[crit_id]

                logger.info(f"▶️ Criterion unblocked: {crit_data['description']}")
                break

        # Update status if no more blocked criteria
        if not state.blocked_criteria:
            state.status = "in_progress"

        state.last_updated_at = datetime.utcnow()
        self._save_state(session_id, "agentic_state", state)

    # Continuation Context Management

    def set_continuation_context(
        self,
        session_id: str,
        awaiting: str,
        options: List[Any] = None,
        original_request: str = None,
        resolved_entities: Dict[str, Any] = None,
        blocked_criteria: List[str] = None
    ):
        """
        Set continuation context for resuming after user input.
        """
        state = self.get_agentic_state(session_id)

        state.continuation_context = ContinuationContext(
            awaiting=awaiting,
            presented_options=options or [],
            original_request=original_request,
            resolved_entities=resolved_entities or {},
            blocked_criteria=blocked_criteria or state.blocked_criteria.copy(),
            waiting_turns=0
        )

        # Also update pending options for easy access
        if options:
            state.pending_user_options = options

        state.last_updated_at = datetime.utcnow()
        self._save_state(session_id, "agentic_state", state)

        logger.info(f"Set continuation context: awaiting={awaiting}, options={len(options or [])}")

    def get_continuation_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get continuation context for resuming flow.
        """
        state = self.get_agentic_state(session_id)

        if state.continuation_context and state.continuation_context.awaiting:
            return state.continuation_context.model_dump()

        return None

    def has_continuation(self, session_id: str) -> bool:
        """
        Check if there's a pending continuation.
        """
        state = self.get_agentic_state(session_id)
        return bool(
            state.continuation_context and
            state.continuation_context.awaiting
        )

    def clear_continuation_context(self, session_id: str):
        """
        Clear continuation context after it's been handled.
        """
        state = self.get_agentic_state(session_id)
        state.continuation_context = ContinuationContext()
        state.pending_user_options = []
        state.last_updated_at = datetime.utcnow()
        self._save_state(session_id, "agentic_state", state)

    # Completion Checking

    def is_task_complete(self, session_id: str) -> bool:
        """
        Check if all success criteria are met.
        """
        state = self.get_agentic_state(session_id)

        if not state.success_criteria:
            return False

        # All criteria must be complete (not pending, blocked, or failed)
        for crit_data in state.criteria.values():
            if crit_data["state"] not in [
                CriterionState.COMPLETE.value,
                CriterionState.SKIPPED.value
            ]:
                return False

        return True

    def has_blocked_criteria(self, session_id: str) -> bool:
        """
        Check if any criteria are blocked.
        """
        state = self.get_agentic_state(session_id)
        return len(state.blocked_criteria) > 0

    def get_blocked_criteria(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get all blocked criteria with their details.
        """
        state = self.get_agentic_state(session_id)

        blocked = []
        for crit_id, details in state.blocked_details.items():
            blocked.append(details if isinstance(details, dict) else details.model_dump())

        return blocked

    def get_pending_criteria(self, session_id: str) -> List[str]:
        """
        Get list of pending (not complete, blocked, or failed) criteria.
        """
        state = self.get_agentic_state(session_id)

        pending = []
        for crit_data in state.criteria.values():
            if crit_data["state"] in [
                CriterionState.PENDING.value,
                CriterionState.IN_PROGRESS.value
            ]:
                pending.append(crit_data["description"])

        return pending

    def get_agentic_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Get a summary of agentic execution for logging/debugging.
        """
        state = self.get_agentic_state(session_id)

        duration_ms = None
        if state.started_at:
            end_time = state.completed_at or datetime.utcnow()
            duration_ms = (end_time - state.started_at).total_seconds() * 1000

        return {
            "session_id": session_id,
            "status": state.status,
            "iterations": state.iteration,
            "max_iterations": state.max_iterations,
            "criteria": {
                "total": len(state.success_criteria),
                "completed": len(state.completed_criteria),
                "blocked": len(state.blocked_criteria),
                "failed": len(state.failed_criteria),
                "pending": len(self.get_pending_criteria(session_id))
            },
            "has_continuation": self.has_continuation(session_id),
            "awaiting": state.continuation_context.awaiting if state.continuation_context else None,
            "observations_count": len(state.observations),
            "tool_calls": state.total_tool_calls,
            "llm_calls": state.total_llm_calls,
            "started_at": state.started_at.isoformat() if state.started_at else None,
            "duration_ms": duration_ms
        }

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
            "agentic_state": self.get_agentic_state(session_id).model_dump(),
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
            "agentic_state",
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
