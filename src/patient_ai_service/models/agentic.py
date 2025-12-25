"""
Agentic models and enums for enhanced agent execution.

This module provides:
1. Tool Result Classification (SUCCESS, PARTIAL, USER_INPUT, RECOVERABLE, FATAL, SYSTEM_ERROR)
2. Criterion States (PENDING, IN_PROGRESS, COMPLETE, BLOCKED, FAILED, SKIPPED)
3. Agent Decision types
4. Models for criteria, observations, thinking results, and completion checks
"""

from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field


# =============================================================================
# ENUMS - Result Types and States
# =============================================================================

class ToolResultType(str, Enum):
    """
    Classification of tool execution results.
    
    This tells the agent what kind of result it received and how to proceed.
    """
    SUCCESS = "success"              # Goal achieved, criterion can be marked complete
    PARTIAL = "partial"              # Progress made, more steps needed
    USER_INPUT_NEEDED = "user_input" # Cannot proceed without user decision
    RECOVERABLE = "recoverable"      # Failed but can try different approach
    FATAL = "fatal"                  # Cannot complete this request
    SYSTEM_ERROR = "system_error"    # Infrastructure failure


class CriterionState(str, Enum):
    """
    State of a success criterion.
    
    Criteria can transition through these states during execution.
    """
    PENDING = "pending"           # Not started yet
    IN_PROGRESS = "in_progress"   # Currently working on it
    COMPLETE = "complete"         # Successfully completed
    BLOCKED = "blocked"           # Waiting for user input
    FAILED = "failed"             # Cannot be completed
    SKIPPED = "skipped"           # Not needed (e.g., already registered)


class AgentDecision(str, Enum):
    """
    Decisions the agent can make during the thinking phase.
    """
    CALL_TOOL = "call_tool"                    # Execute a tool
    RESPOND = "respond"                         # Generate final response
    CLARIFY = "clarify"                         # Ask user for clarification
    RETRY = "retry"                             # Retry last action
    RESPOND_WITH_OPTIONS = "respond_options"    # Present alternatives to user
    RESPOND_COMPLETE = "respond_complete"       # Task fully completed
    RESPOND_IMPOSSIBLE = "respond_impossible"   # Task cannot be done
    EXECUTE_RECOVERY = "execute_recovery"       # Execute recovery_action automatically
    COLLECT_INFORMATION = "collect_information" # Exit agentic loop to collect information, passes through focused response generation
    REQUEST_CONFIRMATION = "request_confirmation"  # Request user confirmation before critical action


# =============================================================================
# MODELS - Criteria, Observations, and Context
# =============================================================================

class Criterion(BaseModel):
    """
    A success criterion with its current state and metadata.
    """
    id: str
    description: str
    state: CriterionState = CriterionState.PENDING
    
    # For BLOCKED state
    blocked_reason: Optional[str] = None
    blocked_options: Optional[List[Any]] = None
    blocked_at_iteration: Optional[int] = None
    
    # For COMPLETE state
    completion_evidence: Optional[str] = None
    completed_at_iteration: Optional[int] = None
    
    # For FAILED state
    failed_reason: Optional[str] = None
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Observation(BaseModel):
    """
    An observation recorded during execution (tool result, system event, etc.)
    """
    type: str  # "tool", "system", "error", "recovery_hint"
    name: str
    result: Dict[str, Any]
    result_type: Optional[ToolResultType] = None
    iteration: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    def is_success(self) -> bool:
        return self.result.get("success", False)
    
    def get_error(self) -> Optional[str]:
        return self.result.get("error") or self.result.get("error_message")


class ConfirmationSummary(BaseModel):
    """Summary of action awaiting confirmation."""
    action: str  # "book_appointment", "cancel_appointment", "reschedule_appointment"
    details: Dict[str, Any]  # doctor_name, date, time, procedure, etc.
    tool_name: str  # The tool to call after confirmation
    tool_input: Dict[str, Any]  # The parameters to pass


class AgentResponseData(BaseModel):
    """
    Structured response data filled by agent during thinking.
    
    Each decision type fills the relevant fields, then a unified
    response generator creates the final user-facing message.
    """
    
    # ═══════════════════════════════════════════════════════════════
    # COLLECT_INFORMATION - Need more info from user before proceeding
    # ═══════════════════════════════════════════════════════════════
    information_needed: Optional[str] = None  
    # Examples: "visit_reason", "preferred_time", "doctor_preference"
    # Human-readable: "What type of appointment do you need?"
    information_question: Optional[str] = None
    # The natural question to ask the user
    
    # ═══════════════════════════════════════════════════════════════
    # CLARIFY - Need clarification on ambiguous input
    # ═══════════════════════════════════════════════════════════════
    clarification_needed: Optional[str] = None
    # What's unclear: "time_ambiguous", "doctor_name_unclear"
    clarification_question: Optional[str] = None
    # The question to ask: "Did you mean 3pm or 3am?"
    
    # ═══════════════════════════════════════════════════════════════
    # RESPOND_WITH_OPTIONS - Present choices to user
    # ═══════════════════════════════════════════════════════════════
    options: Optional[List[Any]] = Field(default_factory=list)
    # The actual options: ["10:00 AM", "2:00 PM", "4:30 PM"]
    options_context: Optional[str] = None
    # What they're choosing: "available_times", "doctors", "dates"
    options_reason: Optional[str] = None
    # Why options are needed: "Requested time not available"
    
    # ═══════════════════════════════════════════════════════════════
    # RESPOND_IMPOSSIBLE - Task cannot be completed
    # ═══════════════════════════════════════════════════════════════
    failure_reason: Optional[str] = None
    # Why it failed: "Doctor not found", "No availability this week"
    failure_suggestion: Optional[str] = None
    # What user can do: "Try a different doctor", "Check next week"
    
    # ═══════════════════════════════════════════════════════════════
    # RESPOND_COMPLETE / RESPOND - Task completed successfully
    # ═══════════════════════════════════════════════════════════════
    completion_summary: Optional[str] = None
    # What was done: "Booked appointment with Dr. Sarah on Dec 15 at 2pm"
    completion_details: Optional[Dict[str, Any]] = Field(default_factory=dict)
    # Structured details: {doctor, date, time, confirmation_number}
    
    # ═══════════════════════════════════════════════════════════════
    # REQUEST_CONFIRMATION - Need user to confirm before action
    # ═══════════════════════════════════════════════════════════════
    confirmation_action: Optional[str] = None
    # What we're about to do: "book_appointment", "cancel_appointment"
    confirmation_details: Optional[Dict[str, Any]] = Field(default_factory=dict)
    # Details to confirm: {doctor, date, time, procedure}
    confirmation_question: Optional[str] = None
    # The question: "Should I book this appointment?"
    
    # ═══════════════════════════════════════════════════════════════
    # COMMON - Shared across all decision types
    # ═══════════════════════════════════════════════════════════════
    entities: Dict[str, Any] = Field(default_factory=dict)
    # What we already know: {patient_id, doctor_name, date, time}
    
    # For internal tracking
    raw_message: Optional[str] = None
    # If agent wants to provide a pre-formed message (fallback)


class PatientContext(BaseModel):
    """Patient context for response generation."""
    name: Optional[str] = None
    is_registered: bool = False
    patient_id: Optional[str] = None
    
    @classmethod
    def from_session(cls, session_id: str, state_manager, entities: Dict = None) -> 'PatientContext':
        """Build patient context from session state."""
        # Try to get from patient state (registered patient)
        try:
            patient_state = state_manager.get_global_state(session_id)
            if patient_state and patient_state.patient_profile and patient_state.patient_profile.patient_id:
                return cls(
                    name=patient_state.patient_profile.first_name,
                    is_registered=True,
                    patient_id=patient_state.patient_profile.patient_id
                )
        except:
            pass
        
        # Not registered - check entities for collected name
        if entities:
            collected_name = (
                entities.get("patient_name") or
                entities.get("first_name") or
                entities.get("name")
            )
            if collected_name:
                return cls(
                    name=collected_name,
                    is_registered=False,
                    patient_id=None
                )
        
        # No name available
        return cls(
            name=None,
            is_registered=False,
            patient_id=None
        )
    
    def get_display_name(self) -> str:
        """Get name for display in prompts."""
        if self.name:
            return self.name
        return "[Name not yet collected]"
    
    def get_prompt_context(self) -> str:
        """Get context string for LLM prompts."""
        if self.is_registered and self.name:
            return f"Patient: {self.name} (registered, ID: {self.patient_id})"
        elif self.name:
            return f"Patient: {self.name} (not registered yet - name collected during conversation)"
        else:
            return "Patient: Not registered, name not yet collected - DO NOT invent a name!"


class ThinkingResult(BaseModel):
    """
    Result of the agent's thinking phase.
    """
    # Analysis
    analysis: str = ""
    
    # Task status
    task_status: Dict[str, Any] = Field(default_factory=dict)
    
    # Decision
    decision: AgentDecision
    reasoning: str = ""
    
    # For CALL_TOOL
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    
    # For RESPOND variants
    response_text: Optional[str] = None
    is_task_complete: bool = False
    
    # NEW: Task tracking
    current_task_id: Optional[str] = None
    
    # Blocking info
    blocked_options: Optional[List[Any]] = None
    blocked_reason: Optional[str] = None
    
    # Criteria updates (for backward compatibility)
    criteria_updates: Dict[str, str] = Field(default_factory=dict)
    
    # NEW: Structured response data
    response: AgentResponseData = Field(default_factory=AgentResponseData)
    
    # Keep legacy fields for backward compatibility during transition
    clarification_question: Optional[str] = None  # Deprecated: use response.clarification_question
    awaiting_info: Optional[str] = None  # Deprecated: use response.information_needed
    entities: Optional[Dict[str, Any]] = None  # Deprecated: use response.entities
    confirmation_summary: Optional[ConfirmationSummary] = None  # Deprecated: use response.confirmation_*
    
    # Internal flags
    detected_result_type: Optional[ToolResultType] = None
    
    # Tool expansion
    request_all_tools: bool = False
    tool_expansion_reason: Optional[str] = None

    # NEW: Updated, session-scoped entities after this thinking step.
    # This is the canonical conversation memory to be persisted in GlobalState.
    updated_entities: Dict[str, Any] = Field(default_factory=dict)

    # NEW: LLM-only entity delta (from entities_to_update)
    # ISOLATED from multi-source entities - contains ONLY what LLM explicitly updated
    # Used to track and persist LLM entity updates separately from internal entities
    llm_delta: Dict[str, Any] = Field(default_factory=dict)


class CompletionCheck(BaseModel):
    """
    Result of completion verification.
    """
    is_complete: bool
    completed_criteria: List[str] = Field(default_factory=list)
    pending_criteria: List[str] = Field(default_factory=list)
    blocked_criteria: List[str] = Field(default_factory=list)
    failed_criteria: List[str] = Field(default_factory=list)
    
    has_blocked: bool = False
    has_failed: bool = False
    
    # Blocked details for response generation
    blocked_options: Dict[str, List[Any]] = Field(default_factory=dict)
    blocked_reasons: Dict[str, str] = Field(default_factory=dict)

