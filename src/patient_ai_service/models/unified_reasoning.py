"""
Unified Reasoning Output Models.

Single-pass reasoning output that replaces both SituationAssessment
and ReasoningOutput for routing decisions.
"""

from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

from patient_ai_service.models.situation_assessment import SituationType


class RouteType(str, Enum):
    """Where to route the message."""
    FAST_PATH = "fast_path"  # Direct response, no agent
    AGENT = "agent"          # Route to specific agent


class PlanDecision(str, Enum):
    """What to do with the execution plan."""
    NO_PLAN = "no_plan"              # Fast-path or general_assistant
    CREATE_NEW = "create_new"        # New intent, no existing plan
    RESUME = "resume"                # Continue existing plan unchanged
    ABANDON_CREATE = "abandon_create" # Abandon old plan, create new
    COMPLETE = "complete"            # Plan is complete, clear it


class UnifiedReasoningOutput(BaseModel):
    """
    Output from unified reasoning engine.

    Replaces both SituationAssessment and ReasoningOutput for routing.
    """

    # Always present
    route_type: RouteType
    situation_type: SituationType

    # For unclear_request fast_path: why it's unclear/what needs clarity
    why_unclear: Optional[str] = Field(
        None,
        description="For unclear_request situations: explanation of why the request is unclear or what specific information is needed for clarification"
    )

    # Only for agent routing (None for fast_path)
    confidence: Optional[float] = None
    agent: Optional[str] = None
    plan_decision: Optional[PlanDecision] = None
    plan_reasoning: Optional[str] = None
    what_user_means: Optional[str] = None
    objective: Optional[str] = None
    is_continuation: bool = False
    continuation_type: Optional[str] = None
    
    # NEW: Direct routing action for special flows
    routing_action: Optional[str] = None
    # Values:
    #   - "execute_confirmed_action": Execute pending action immediately (confirmation flow)
    #   - "collect_information": Generate lightweight response for info collection (information collection flow)
    #   - None: Route to agent normally

    @classmethod
    def fast_path(cls, situation_type: SituationType, why_unclear: Optional[str] = None) -> "UnifiedReasoningOutput":
        """Factory for fast-path responses."""
        return cls(
            route_type=RouteType.FAST_PATH,
            situation_type=situation_type,
            why_unclear=why_unclear
        )

    def needs_agent(self) -> bool:
        """Check if this requires agent execution."""
        return self.route_type == RouteType.AGENT

    def is_fast_path(self) -> bool:
        """Check if this is a fast-path response."""
        return self.route_type == RouteType.FAST_PATH

