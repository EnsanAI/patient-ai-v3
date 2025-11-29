"""
Validation models for closed-loop validation system.

Tracks tool execution and validation results for agent response validation.
"""

from datetime import datetime
from typing import Dict, Any, List
from pydantic import BaseModel, Field


class ToolExecution(BaseModel):
    """Record of a single tool execution."""
    tool_name: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ExecutionLog(BaseModel):
    """Complete log of agent execution for validation."""
    tools_used: List[ToolExecution] = Field(default_factory=list)
    conversation_turns: int = 0


class ValidationResult(BaseModel):
    """Result of validating an agent's response."""
    is_valid: bool
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    issues: List[str] = Field(default_factory=list)
    decision: str  # "send", "retry", "redirect", "fallback"
    feedback_to_agent: str = ""
    reasoning: List[str] = Field(default_factory=list)

    def should_retry(self) -> bool:
        """Check if validation result suggests retry."""
        return not self.is_valid and self.decision == "retry"

    def should_fallback(self) -> bool:
        """Check if validation result suggests fallback."""
        return not self.is_valid and self.decision == "fallback"
