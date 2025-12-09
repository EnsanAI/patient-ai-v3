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
    
    # For CLARIFY
    clarification_question: Optional[str] = None
    
    # Internal flags
    is_task_complete: bool = False
    detected_result_type: Optional[ToolResultType] = None


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

