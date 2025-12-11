"""
Validation models for closed-loop validation system.

Tracks tool execution and validation results for agent response validation.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class ToolExecution(BaseModel):
    """Record of a single tool execution."""
    tool_name: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    success: bool = True
    duration_seconds: Optional[float] = None  # Changed from duration_ms to duration_seconds
    error: Optional[str] = None


class ExecutionLog(BaseModel):
    """Complete log of agent execution for validation."""
    tools_used: List[ToolExecution] = Field(default_factory=list)
    conversation_turns: int = 0


class ValidationResult(BaseModel):
    """Result of validating an agent's response."""
    is_valid: bool
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    issues: List[str] = Field(default_factory=list)
    decision: str  # "send", "retry", "redirect", "fallback", "edit"
    feedback_to_agent: str = ""
    reasoning: List[str] = Field(default_factory=list)

    # Response finalization fields (NEW - for two-layer quality control)
    rewritten_response: Optional[str] = Field(
        default=None,
        description="Corrected/edited response from finalization layer"
    )
    was_rewritten: bool = Field(
        default=False,
        description="True if finalization layer edited the response"
    )

    def should_retry(self) -> bool:
        """Check if validation result suggests retry."""
        return not self.is_valid and self.decision == "retry"

    def should_fallback(self) -> bool:
        """Check if validation result suggests fallback."""
        return not self.is_valid and self.decision == "fallback"

    def should_use_rewritten(self) -> bool:
        """Check if rewritten response should be used instead of agent's."""
        return self.rewritten_response is not None and self.was_rewritten
