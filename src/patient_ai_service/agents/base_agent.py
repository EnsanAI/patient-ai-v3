"""

Base Agent class for all specialized agents.

Implements a ReAct (Reasoning, Action, Observation) pattern for smart,

dynamic tool execution that reads results before responding.

Key improvements:

1. Agentic loop: Think → Act → Observe → Repeat until done

2. No premature responses: Only respond after task completion verified

3. Result-aware: Reads and validates tool results before deciding next steps

4. Dynamic execution: LLM decides what tools to use, not hardcoded sequences

"""

import json
import logging
import time
import inspect
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable, Tuple, TYPE_CHECKING
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

if TYPE_CHECKING:
    from patient_ai_service.models.task_plan import TaskPlan, Task

from patient_ai_service.models.agent_plan import AgentPlan, PlanAction, PlanStatus, TaskStatus, PlanTask

from patient_ai_service.core import get_llm_client, get_state_manager
from patient_ai_service.core.llm import LLMClient
from patient_ai_service.core.state_manager import StateManager
from patient_ai_service.core.observability import get_observability_logger
from patient_ai_service.core.config import settings
from patient_ai_service.core.feature_flags import FeatureFlags, is_humanizer_enabled
from patient_ai_service.core.llm_config import (
    LLMConfig,
    LLMConfigManager,
    get_llm_config_manager,
)
from patient_ai_service.core.prompt_cache import (
    get_universal_system_content,
    get_cache_config,
)
from patient_ai_service.models.validation import ExecutionLog, ToolExecution
from patient_ai_service.models.observability import (
    LLMCall,
    ToolExecutionDetail,
    AgentExecutionDetails,
    AgentContext,
    TokenUsage,
    CostInfo
)
from patient_ai_service.models.agentic import (
    ToolResultType,
    CriterionState,
    Criterion,
    Observation,
    CompletionCheck,
    ThinkingResult,
    AgentDecision,
    ConfirmationSummary,
    AgentResponseData,
    PatientContext
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# HUMANIZER PROMPT - Natural, human-like clinic receptionist
# ═══════════════════════════════════════════════════════════════════════════════
HUMANIZE_SYSTEM_PROMPT = """You are a SUPER PROFESSIONAL, natural, human-like clinic receptionist. Your goal: patients never realize you're AI. Humanize system's SUGGESTED RESPONSE and make it relevant to conversation. 

## CORE RULES
• Match patient's exact language, dialect, and tone (English, Egyptian Arabic, Emirati Arabic, Italian, Spanish, etc.)
• Keep ALL medical details unchanged (names, times, appointments, prescriptions)
• Never use technical terms, system IDs, or tool names
• 1-2 sentences maximum - be direct and warm
• When conversation is recent read it for a contextual cntinuuos response

## TONE HANDLING
• Angry patients: Stay calm, acknowledge briefly without repetition
• Anxious patients: Be reassuring but realistic
• Confused patients: Clarify simply
• NEVER repeat similar responses multiple times
• NEVER promise any future actions.

## FORBIDDEN PHRASES
✗ "I understand" (more than once)
✗ Any technical/system language like system IDs or tool names

## NATURAL RESPONSES
✓ Speak like a real person would
✓ Use patient's communication style
✓ Be empathetic without being redundant
✓ Sound like Jarvis - intelligent, natural, helpful

## OUTPUT
Reply as if you're speaking directly to the patient - natural, conversational, matching their language, DIALECT, tone exactly."""


def format_time_ago(timestamp: datetime) -> str:
    """Format timestamp as first unit only: '2m ago', '3h ago', '1d ago'"""
    if timestamp is None:
        return ""

    try:
        now = datetime.now(timestamp.tzinfo) if timestamp.tzinfo else datetime.now()
        delta = now - timestamp

        if delta.days > 0:
            return f"{delta.days}d ago"

        hours = delta.seconds // 3600
        if hours > 0:
            return f"{hours}h ago"

        mins = delta.seconds // 60
        if mins > 0:
            return f"{mins}m ago"

        return "now"
    except Exception:
        return ""


class CircuitState(Enum):
    """Circuit breaker state."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreaker:
    """
    Circuit breaker to prevent cascading failures.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, reject requests immediately
    - HALF_OPEN: Allow one test request to check recovery
    """
    failure_threshold: int = 3          # Failures before opening
    recovery_timeout: float = 30.0      # Seconds before trying again

    _failure_count: int = field(default=0, init=False)
    _last_failure_time: Optional[float] = field(default=None, init=False)
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)

    def record_success(self) -> None:
        """Record a successful operation."""
        self._failure_count = 0
        self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        """Record a failed operation."""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            logger.warning(
                f"Circuit breaker OPENED after {self._failure_count} failures"
            )

    def can_execute(self) -> bool:
        """Check if operation should be attempted."""
        if self._state == CircuitState.CLOSED:
            return True

        if self._state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if self._last_failure_time:
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    logger.info("Circuit breaker entering HALF_OPEN state")
                    return True
            return False

        # HALF_OPEN: allow one test request
        return True

    def get_state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state

    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        self._failure_count = 0
        self._last_failure_time = None
        self._state = CircuitState.CLOSED


# AgentDecision and ThinkingResult are imported from patient_ai_service.models.agentic
# Removed duplicate definitions - using the imported Pydantic models instead


# =============================================================================
# EXECUTION CONTEXT - Tracks everything during execution
# =============================================================================

class ExecutionContext:
    """
    Tracks all state during agentic execution.
    
    This is the central hub for:
    - Observations (tool results, events)
    - Criteria and their states
    - Continuation context for blocked flows
    - Metrics and debugging info
    """
    
    def __init__(self, session_id: str, max_iterations: int = 15, user_request: str = ""):
        self.session_id = session_id
        self.max_iterations = max_iterations
        self.iteration = 0
        self.user_request = user_request  # Store user's original message
        
        # Observations
        self.observations: List[Observation] = []
        
        # Criteria tracking
        self.criteria: Dict[str, Criterion] = {}
        
        # User options when blocked
        self.pending_user_options: List[Any] = []
        self.suggested_response: Optional[str] = None
        
        # Continuation context (persisted for next turn)
        self.continuation_context: Dict[str, Any] = {}
        
        # Error tracking
        self.fatal_error: Optional[Dict[str, Any]] = None
        self.retry_count: int = 0
        self.max_retries: int = 2
        
        # Recovery tracking
        self.recovery_attempts: Dict[str, Dict[str, Any]] = {}  # Key: error_signature, Value: recovery info
        self.recovery_executed: bool = False  # Track if we're currently executing a recovery
        
        # Metrics
        self.tool_calls: int = 0
        self.llm_calls: int = 0
        self.started_at: datetime = datetime.utcnow()
        
        # Task context (from reasoning engine)
        self.task_context: Optional[Dict[str, Any]] = None
        
        # NEW: Task Plan (legacy - for backward compatibility)
        self.task_plan: Optional['TaskPlan'] = None
        self.planning_complete: bool = False
        
        # NEW: Agent Plan (Phase 4 - persistent execution plan)
        self.plan: Optional[AgentPlan] = None
    
    # -------------------------------------------------------------------------
    # Observation Management
    # -------------------------------------------------------------------------
    
    def add_observation(
        self,
        obs_type: str,
        name: str,
        result: Dict[str, Any],
        result_type: Optional[ToolResultType] = None
    ):
        """Add an observation from a tool or system event."""
        # Auto-detect result_type if not provided
        if result_type is None:
            result_type = self._infer_result_type(result)
        
        obs = Observation(
            type=obs_type,
            name=name,
            result=result,
            result_type=result_type,
            iteration=self.iteration
        )
        self.observations.append(obs)
        
        if obs_type == "tool":
            self.tool_calls += 1
        
        logger.debug(
            f"[Iteration {self.iteration}] Observation: {obs_type}/{name} "
            f"result_type={result_type} success={result.get('success')}"
        )
    
    def _infer_result_type(self, result: Dict[str, Any]) -> ToolResultType:
        """Infer result type from result content if not explicitly set."""
        # Check for explicit result_type
        if "result_type" in result:
            try:
                return ToolResultType(result["result_type"])
            except ValueError:
                pass
        
        # Infer from content
        if result.get("success") is False:
            if result.get("recovery_action"):
                return ToolResultType.RECOVERABLE
            elif result.get("should_retry"):
                return ToolResultType.SYSTEM_ERROR
            else:
                return ToolResultType.FATAL
        
        if result.get("alternatives") and not result.get("available", True):
            return ToolResultType.USER_INPUT_NEEDED
        
        if result.get("next_action") or result.get("can_proceed") is True:
            return ToolResultType.PARTIAL
        
        if result.get("success") is True:
            if result.get("satisfies_criteria") or result.get("appointment_id"):
                return ToolResultType.SUCCESS
            return ToolResultType.PARTIAL
        
        return ToolResultType.PARTIAL  # Default
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert ExecutionContext to a JSON-serializable dictionary."""
        from datetime import datetime
        
        def serialize_value(v):
            """Recursively serialize values for JSON."""
            if isinstance(v, datetime):
                return v.isoformat()
            elif isinstance(v, dict):
                return {k: serialize_value(val) for k, val in v.items()}
            elif isinstance(v, list):
                return [serialize_value(item) for item in v]
            elif hasattr(v, 'dict') or hasattr(v, 'model_dump'):  # Pydantic models
                try:
                    return v.model_dump() if hasattr(v, 'model_dump') else v.dict()
                except:
                    return str(v)
            elif hasattr(v, '__dict__'):
                return {k: serialize_value(val) for k, val in v.__dict__.items()}
            else:
                return v
        
        result = {
            "session_id": self.session_id,
            "max_iterations": self.max_iterations,
            "iteration": self.iteration,
            "user_request": self.user_request,
            "observations": [serialize_value(obs) for obs in self.observations],
            "criteria": {k: serialize_value(v) for k, v in self.criteria.items()},
            "pending_user_options": serialize_value(self.pending_user_options),
            "suggested_response": self.suggested_response,
            "continuation_context": serialize_value(self.continuation_context),
            "fatal_error": serialize_value(self.fatal_error),
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "recovery_attempts": serialize_value(self.recovery_attempts),
            "recovery_executed": self.recovery_executed,
            "tool_calls": self.tool_calls,
            "llm_calls": self.llm_calls,
            "started_at": serialize_value(self.started_at),
            "task_context": serialize_value(self.task_context),
            "plan": serialize_value(self.plan) if self.plan else None,
        }
        return result
    
    def get_observations_summary(self) -> str:
        """Get a formatted summary of all observations for the thinking prompt."""
        if not self.observations:
            return "No observations yet."
        
        lines = []
        for obs in self.observations:
            status = "✅" if obs.is_success() else "❌"
            result_type_str = f"[{obs.result_type.value}]" if obs.result_type else ""
            
            # Format result (truncate if too long - 1000 chars to preserve important fields)
            result_str = json.dumps(obs.result, indent=2, default=str)
            if len(result_str) > 1000:
                result_str = result_str[:1000] + "..."
            
            lines.append(
                f"[Iteration {obs.iteration}] {status} {obs.type}: {obs.name} {result_type_str}\n"
                f"Result: {result_str}"
            )
        
        return "\n\n".join(lines)
    
    def get_last_observation(self) -> Optional[Observation]:
        """Get the most recent observation."""
        return self.observations[-1] if self.observations else None
    
    def get_successful_tools(self) -> List[str]:
        """Get names of tools that succeeded."""
        return [
            obs.name for obs in self.observations
            if obs.type == "tool" and obs.is_success()
        ]
    
    def get_failed_tools(self) -> List[str]:
        """Get names of tools that failed."""
        return [
            obs.name for obs in self.observations
            if obs.type == "tool" and not obs.is_success()
        ]
    
    # -------------------------------------------------------------------------
    # Criteria Management
    # -------------------------------------------------------------------------
    
    # NOTE: initialize_criteria() removed in Phase 6 - plans replace criteria-based execution
    
    def add_criterion(self, description: str, required: bool = True) -> str:
        """Add a new criterion discovered during execution."""
        criterion_id = f"criterion_{len(self.criteria)}"
        self.criteria[criterion_id] = Criterion(
            id=criterion_id,
            description=description,
            state=CriterionState.PENDING
        )
        logger.info(f"Added new criterion: {description}")
        return criterion_id
    
    def mark_criterion_complete(
        self,
        description_or_id: str,
        evidence: Optional[str] = None
    ):
        """Mark a criterion as complete."""
        criterion = self._find_criterion(description_or_id)
        if criterion:
            criterion.state = CriterionState.COMPLETE
            criterion.completion_evidence = evidence
            criterion.completed_at_iteration = self.iteration
            criterion.updated_at = datetime.utcnow()
            logger.info(f"✅ Criterion COMPLETE: {criterion.description}")
    
    def mark_criterion_blocked(
        self,
        description_or_id: str,
        reason: str,
        options: List[Any] = None
    ):
        """Mark a criterion as blocked pending user input."""
        criterion = self._find_criterion(description_or_id)
        if criterion:
            criterion.state = CriterionState.BLOCKED
            criterion.blocked_reason = reason
            criterion.blocked_options = options
            criterion.blocked_at_iteration = self.iteration
            criterion.updated_at = datetime.utcnow()
            logger.info(f"⏸️ Criterion BLOCKED: {criterion.description} - {reason}")
    
    def mark_criterion_failed(self, description_or_id: str, reason: str):
        """Mark a criterion as failed."""
        criterion = self._find_criterion(description_or_id)
        if criterion:
            criterion.state = CriterionState.FAILED
            criterion.failed_reason = reason
            criterion.updated_at = datetime.utcnow()
            logger.warning(f"❌ Criterion FAILED: {criterion.description} - {reason}")
    
    def _find_criterion(self, description_or_id: str) -> Optional[Criterion]:
        """Find criterion by ID or description match."""
        # Try direct ID match
        if description_or_id in self.criteria:
            return self.criteria[description_or_id]
        
        # Try description match (partial)
        description_lower = description_or_id.lower()
        for criterion in self.criteria.values():
            if description_lower in criterion.description.lower():
                return criterion
        
        return None
    
    # -------------------------------------------------------------------------
    # Completion Checking
    # -------------------------------------------------------------------------
    
    def check_completion(self) -> CompletionCheck:
        """Check if all criteria are met or if we're blocked.
        
        Phase 6 update: Also checks plan tasks when criteria are not being used.
        """
        logger.info(f"✓ [ExecutionContext] Checking completion status...")
        result = CompletionCheck(is_complete=False)
        
        # Check criteria first (legacy path)
        for criterion in self.criteria.values():
            if criterion.state == CriterionState.COMPLETE:
                result.completed_criteria.append(criterion.description)
                logger.info(f"✓ [ExecutionContext]   ✅ COMPLETE: {criterion.description}")
            
            elif criterion.state == CriterionState.BLOCKED:
                result.blocked_criteria.append(criterion.description)
                result.has_blocked = True
                if criterion.blocked_options:
                    result.blocked_options[criterion.description] = criterion.blocked_options
                if criterion.blocked_reason:
                    result.blocked_reasons[criterion.description] = criterion.blocked_reason
                logger.info(f"✓ [ExecutionContext]   ⏸️  BLOCKED: {criterion.description} (reason: {criterion.blocked_reason})")
            
            elif criterion.state == CriterionState.FAILED:
                result.failed_criteria.append(criterion.description)
                result.has_failed = True
                logger.info(f"✓ [ExecutionContext]   ❌ FAILED: {criterion.description} (reason: {criterion.failed_reason})")
            
            else:  # PENDING or IN_PROGRESS
                result.pending_criteria.append(criterion.description)
                logger.info(f"✓ [ExecutionContext]   ○ PENDING: {criterion.description}")
        
        # Phase 6: If no criteria, check plan tasks instead
        if not self.plan:
            logger.info(f"✓ [ExecutionContext] Using plan-based completion check (no criteria)")
            from patient_ai_service.models.agent_plan import TaskStatus
            
            for task in self.plan.tasks:
                if task.status == TaskStatus.COMPLETE:
                    result.completed_criteria.append(f"Task: {task.description}")
                    logger.info(f"✓ [ExecutionContext]   ✅ TASK COMPLETE: {task.id} - {task.description}")
                
                elif task.status == TaskStatus.BLOCKED:
                    result.blocked_criteria.append(f"Task: {task.description}")
                    result.has_blocked = True
                    if task.blocked_options:
                        result.blocked_options[task.description] = task.blocked_options
                    if task.blocked_reason:
                        result.blocked_reasons[task.description] = task.blocked_reason
                    logger.info(f"✓ [ExecutionContext]   ⏸️  TASK BLOCKED: {task.id} - {task.description}")
                
                elif task.status == TaskStatus.FAILED:
                    result.failed_criteria.append(f"Task: {task.description}")
                    result.has_failed = True
                    logger.info(f"✓ [ExecutionContext]   ❌ TASK FAILED: {task.id} - {task.description} (reason: {task.failed_reason})")
                
                else:  # PENDING or IN_PROGRESS
                    result.pending_criteria.append(f"Task: {task.description}")
                    logger.info(f"✓ [ExecutionContext]   ○ TASK PENDING: {task.id} - {task.description}")
        
        # Complete if all criteria/tasks are complete (none pending, blocked, or failed)
        result.is_complete = (
            len(result.pending_criteria) == 0 and
            len(result.blocked_criteria) == 0 and
            len(result.failed_criteria) == 0 and
            len(result.completed_criteria) > 0
        )
        
        logger.info(f"✓ [ExecutionContext] Summary:")
        logger.info(f"✓ [ExecutionContext]   → Completed: {len(result.completed_criteria)}")
        logger.info(f"✓ [ExecutionContext]   → Pending: {len(result.pending_criteria)}")
        logger.info(f"✓ [ExecutionContext]   → Blocked: {len(result.blocked_criteria)}")
        logger.info(f"✓ [ExecutionContext]   → Failed: {len(result.failed_criteria)}")
        logger.info(f"✓ [ExecutionContext]   → Is Complete: {result.is_complete}")
        
        return result
    
    def has_blocked_criteria(self) -> bool:
        """Check if any criteria are blocked."""
        return any(c.state == CriterionState.BLOCKED for c in self.criteria.values())
    
    def get_blocked_criteria(self) -> List[Criterion]:
        """Get all blocked criteria."""
        return [c for c in self.criteria.values() if c.state == CriterionState.BLOCKED]
    
    # NOTE: get_criteria_display() removed in Phase 6 - replaced with plan-based execution
    
    # NEW: Task Plan Methods
    
    def set_task_plan(self, plan: 'TaskPlan'):
        """Set the task plan for this execution."""
        from patient_ai_service.models.task_plan import TaskPlan
        self.task_plan = plan
        self.planning_complete = True
        logger.info(f"Task plan set with {len(plan.tasks)} tasks")
    
    def get_next_task(self) -> Optional['Task']:
        """Get the next executable task from the plan."""
        if not self.task_plan:
            return None
        return self.task_plan.get_next_executable_task()
    
    def mark_task_complete(self, task_id: str, result: Dict[str, Any]):
        """Mark a task as complete."""
        if not self.task_plan:
            return
        
        task = self.task_plan.get_task(task_id)
        if task:
            task.mark_complete(result, self.iteration)
            self.task_plan.update_metrics()
            logger.info("📋 [TaskPlan] ════════════════════════════════════════════════════════")
            logger.info(f"📋 [TaskPlan] ✅ Task completed: {task_id} ({task.action or task.description})")
    
    def mark_task_blocked(
        self,
        task_id: str,
        reason: str,
        options: List[Any] = None,
        suggested_response: str = None
    ):
        """Mark a task as blocked."""
        if not self.task_plan:
            return
        
        task = self.task_plan.get_task(task_id)
        if task:
            task.mark_blocked(reason, options, suggested_response)
            self.task_plan.update_metrics()
            logger.info("📋 [TaskPlan] ════════════════════════════════════════════════════════")
            logger.info(f"📋 [TaskPlan] ⚠️  Task blocked: {task_id} ({task.action or task.description})")
            logger.info(f"📋 [TaskPlan]   Reason: {reason}")
            if options:
                logger.info(f"📋 [TaskPlan]   Options: {len(options)}")
    
    def mark_task_failed(self, task_id: str, reason: str):
        """Mark a task as failed."""
        if not self.task_plan:
            return
        
        task = self.task_plan.get_task(task_id)
        if task:
            task.mark_failed(reason)
            self.task_plan.update_metrics()
            logger.warning("📋 [TaskPlan] ════════════════════════════════════════════════════════")
            logger.warning(f"📋 [TaskPlan] ❌ Task failed: {task_id} ({task.action or task.description})")
            logger.warning(f"📋 [TaskPlan]   Reason: {reason}")
    
    def is_plan_complete(self) -> bool:
        """Check if all tasks in the plan are complete."""
        if not self.task_plan:
            return False
        return self.task_plan.all_tasks_complete()
    
    def is_plan_blocked(self) -> bool:
        """Check if the plan is blocked."""
        if not self.task_plan:
            return False
        return self.task_plan.has_blocked_tasks()
    
    def get_task_plan_display(self) -> str:
        """Get task plan display for prompts."""
        if not self.task_plan:
            return "No task plan generated yet."
        return self.task_plan.get_status_display()
    
    def get_plan_continuation_context(self) -> Dict[str, Any]:
        """Get context for resuming a blocked plan."""
        if not self.task_plan:
            return {}
        return self.task_plan.get_continuation_context()
    
    # -------------------------------------------------------------------------
    # Criteria Display (existing method)
    # -------------------------------------------------------------------------
    
    def get_criteria_display(self) -> str:
        """Get formatted criteria display for thinking prompt."""
        if not self.criteria:
            return "No success criteria defined."
        
        lines = []
        for criterion in self.criteria.values():
            if criterion.state == CriterionState.COMPLETE:
                icon = "✅"
                extra = f" (evidence: {criterion.completion_evidence})" if criterion.completion_evidence else ""
            elif criterion.state == CriterionState.BLOCKED:
                icon = "⏸️"
                extra = f" (blocked: {criterion.blocked_reason})"
            elif criterion.state == CriterionState.FAILED:
                icon = "❌"
                extra = f" (failed: {criterion.failed_reason})"
            elif criterion.state == CriterionState.IN_PROGRESS:
                icon = "🔄"
                extra = ""
            else:
                icon = "○"
                extra = ""
            
            lines.append(f"{icon} {criterion.description}{extra}")
        
        return "\n".join(lines)
    
    # -------------------------------------------------------------------------
    # Continuation Context
    # -------------------------------------------------------------------------
    
    def set_continuation_context(self, **kwargs):
        """Store context for resuming after user input."""
        self.continuation_context.update(kwargs)
    
    def get_continuation_context(self) -> Dict[str, Any]:
        """Get stored continuation context."""
        return self.continuation_context.copy()
    
    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize context for state persistence."""
        return {
            "session_id": self.session_id,
            "iteration": self.iteration,
            "criteria": {k: v.model_dump() for k, v in self.criteria.items()},
            "observations_count": len(self.observations),
            "continuation_context": self.continuation_context,
            "pending_user_options": self.pending_user_options,
            "suggested_response": self.suggested_response,
            "tool_calls": self.tool_calls,
            "llm_calls": self.llm_calls,
            "started_at": self.started_at.isoformat()
        }


class BaseAgent(ABC):
    """
    Abstract base class for all agents implementing ReAct pattern.

    The agent operates in a loop:

    1. THINK: Analyze current state, what's been done, what's needed

    2. DECIDE: Call tool, respond to user, retry, or clarify

    3. ACT: Execute the decision

    4. OBSERVE: Record results

    5. REPEAT: Until task is complete or max iterations reached

    """

    # Maximum iterations to prevent infinite loops
    DEFAULT_MAX_ITERATIONS = 15

    def __init__(
        self,
        agent_name: str,
        llm_client: Optional[LLMClient] = None,
        state_manager: Optional[StateManager] = None,
        max_iterations: Optional[int] = None
    ):
        self.agent_name = agent_name
        self.llm_client = llm_client or get_llm_client()
        self.state_manager = state_manager or get_state_manager()
        self.max_iterations = max_iterations or self.DEFAULT_MAX_ITERATIONS

        # Hierarchical LLM config support
        self._config_agent_name = self._normalize_agent_name(agent_name)
        self._llm_config_manager: Optional[LLMConfigManager] = None  # Lazy loaded

        # Conversation history per session
        self.conversation_history: Dict[str, List[Dict[str, str]]] = {}

        # Minimal context from reasoning engine (per session)
        self._context: Dict[str, Dict[str, Any]] = {}

        # Execution log storage (passed from orchestrator)
        self._execution_log: Dict[str, ExecutionLog] = {}

        # Tool registry
        self._tools: Dict[str, Callable] = {}
        self._tool_schemas: List[Dict[str, Any]] = []
        
        # NEW: Cached static content
        self._cached_tools_description: Optional[str] = None
        self._cached_react_instructions: Optional[str] = None
        self._cached_result_type_guide: Optional[str] = None
        self._cached_decision_guide: Optional[str] = None

        # Register agent-specific tools
        self._register_tools()
        
        # Build caches after tools registered
        self._build_static_caches()

        logger.info(f"Initialized {self.agent_name} agent with ReAct pattern (max_iterations={self.max_iterations}) (static content cached)")

    # ==================== ABSTRACT METHODS ====================

    @abstractmethod
    def _get_system_prompt(self, session_id: str) -> str:
        """Generate system prompt with current context."""
        pass

    @abstractmethod
    def _register_tools(self):
        """Register agent-specific tools."""
        pass

    # ==================== HOOKS ====================

    async def on_activated(self, session_id: str, reasoning: Any):
        """Called when agent is selected for a session."""
        pass

    def set_context(self, session_id: str, context: Dict[str, Any]):
        """Set minimal context for this session."""
        self._context[session_id] = context

    # ==================== HIERARCHICAL LLM CONFIG HELPERS ====================

    @staticmethod
    def _normalize_agent_name(agent_name: str) -> str:
        """
        Convert CamelCase agent name to snake_case for config lookup.
        
        Examples:
            "AppointmentManagerAgent" → "appointment_manager"
            "MedicalInquiryAgent" → "medical_inquiry"
            "TranslationAgent" → "translation"
            "appointment_manager" → "appointment_manager" (no change)
        
        Args:
            agent_name: Agent name in any format
        
        Returns:
            Normalized snake_case agent name
        """
        if not agent_name:
            return agent_name
        
        # If already snake_case, return as-is
        if '_' in agent_name and agent_name.islower():
            return agent_name
        
        # Use LLMConfigManager's normalization method
        return LLMConfigManager._normalize_agent_name(agent_name)
    
    def _get_llm_config_manager(self) -> LLMConfigManager:
        """Get or create LLMConfigManager instance (lazy loading)."""
        if self._llm_config_manager is None:
            self._llm_config_manager = get_llm_config_manager()
        return self._llm_config_manager
    
    def _get_llm_config_for_function(self, function_name: Optional[str] = None) -> LLMConfig:
        """
        Get hierarchically-resolved LLM config for a specific function.
        
        Resolution order: function → agent → global
        
        Args:
            function_name: Function name (e.g., "_think", "_verify_task_completion")
        
        Returns:
            LLMConfig with resolved values
        """
        config_manager = self._get_llm_config_manager()
        return config_manager.get_config(
            agent_name=self._config_agent_name,
            function_name=function_name
        )
    
    def _get_llm_client_for_function(self, function_name: Optional[str] = None) -> LLMClient:
        """
        Get hierarchically-configured LLM client for a specific function.
        
        Clients are cached by provider+model combination.
        
        Args:
            function_name: Function name (e.g., "_think", "_verify_task_completion")
        
        Returns:
            Cached LLMClient instance configured for this function
        """
        config_manager = self._get_llm_config_manager()
        return config_manager.get_client(
            agent_name=self._config_agent_name,
            function_name=function_name
        )
        logger.debug(f"Set context for {self.agent_name} session {session_id}: {context}")


    # ==================== TOOL REGISTRATION ====================

    def register_tool(
        self,
        name: str,
        function: Callable,
        description: str,
        parameters: Dict[str, Any]
    ):
        """Register a tool/action for this agent."""
        self._tools[name] = function
        # Identify required parameters (those explicitly marked as required or without defaults)
        # Default to optional (False) unless explicitly marked as required
        required = [
            param_name for param_name, param_schema in parameters.items()
            if param_schema.get("required", False) is True and "default" not in param_schema
        ]
        schema = {
            "name": name,
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": parameters,
                "required": required  # Only include explicitly required parameters
            }
        }
        self._tool_schemas.append(schema)
        
        # IMPORTANT: Invalidate tools cache when tools change
        self._cached_tools_description = None
        
        logger.debug(f"Registered tool '{name}' for {self.agent_name}")

    # ==================== PLAN GENERATION (Phase 4) ====================
    
    async def _plan(
        self,
        session_id: str,
        objective: str,
        entities: Dict[str, Any],
        constraints: Optional[List[str]] = None
    ) -> AgentPlan:
        """
        Generate execution plan for the objective.
        
        Override in specialized agents for domain-specific planning.
        Default: Use LLM to generate plan based on role + tools.
        
        Args:
            session_id: Session identifier
            objective: WHAT to achieve (from reasoning engine)
            entities: Known entities (from reasoning engine)
            constraints: Any constraints (from reasoning engine)
            
        Returns:
            AgentPlan with tasks to execute
        """
        logger.info(f"📋 [{self.agent_name}] Creating new plan...")
        logger.info(f"📋 [{self.agent_name}]   Objective: {objective}")
        logger.info(f"📋 [{self.agent_name}]   Entities: {list(entities.keys())}")
        
        plan = AgentPlan(
            session_id=session_id,
            agent_name=self.agent_name,
            objective=objective,
            original_message=entities.get("_original_message", ""),
            initial_entities=entities,
            constraints=constraints or []
        )
        
        # Default: Let LLM generate the plan
        await self._generate_plan_with_llm(plan)
        
        logger.info(f"📋 [{self.agent_name}] ✅ Generated plan: {len(plan.tasks)} tasks")
        for task in plan.tasks:
            deps = f" (depends on: {task.depends_on})" if task.depends_on else ""
            logger.info(f"📋 [{self.agent_name}]   • {task.id}: {task.tool or 'info'} - {task.description}{deps}")
        
        plan.status = PlanStatus.EXECUTING
        logger.info(f"🧠 [{self.agent_name}] Plan status: {plan.status.value}")
        return plan
    
    async def _generate_plan_with_llm(self, plan: AgentPlan):
        """Use LLM to generate tasks for the plan."""

        tools_desc = self._get_tools_description()

        # Get agent-specific instructions (only add section if instructions exist)
        agent_instructions = self._get_agent_instructions()
        instructions_section = f"""
AGENT-SPECIFIC INSTRUCTIONS (FOLLOW STRICTLY):
{agent_instructions}

""" if agent_instructions else ""

        prompt = f"""You are the {self.agent_name} agent. Generate an execution plan.

OBJECTIVE: {plan.objective}

KNOWN ENTITIES:
{json.dumps(plan.initial_entities, indent=2, default=str)}

{instructions_section}AVAILABLE TOOLS:
{tools_desc}

Generate a plan as JSON:
{{
    "tasks": [
        {{
            "id": "task_1",
            "description": "What this task does",
            "tool": "tool_name or null for info gathering",
            "params": {{}},
            "depends_on": []
        }}
    ]
}}

RULES:
1. Only use tools listed above
2. Order tasks by dependencies
3. Keep it minimal - only necessary tasks
4. If info is missing, first task should be to collect it

Output ONLY valid JSON."""

        # Get observability logger
        obs_logger = get_observability_logger(plan.session_id) if settings.enable_observability else None
        
        # Log the built prompts
        system_prompt = "You are a task planner. Output only valid JSON."
        logger.info("=" * 80)
        logger.info(f"🗺️ [{self.agent_name}] BUILT PLAN PROMPTS:")
        logger.info("=" * 80)
        logger.info(f"🗺️ [{self.agent_name}] System Prompt:\n{system_prompt}")
        logger.info("-" * 80)
        logger.info(f"🗺️ [{self.agent_name}] User Prompt:\n{prompt}")
        logger.info("=" * 80)
        
        # Get hierarchical LLM config for plan generation
        # Note: plan generation doesn't have a specific function name, so use agent-level config
        # Pass None as function_name to get agent-level config (not function-level override)
        llm_config = self._get_llm_config_for_function(None)  # Use agent-level config, not _think override
        plan_temperature = llm_config.temperature
        plan_max_tokens = llm_config.max_tokens
        llm_client = self._get_llm_client_for_function(None)  # Use agent-level config, not _think override
        
        # Make LLM call with token tracking
        llm_start_time = time.time()
        
        if hasattr(llm_client, 'create_message_with_usage'):
            response, tokens = llm_client.create_message_with_usage(
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
                temperature=plan_temperature,
                max_tokens=plan_max_tokens
            )
        else:
            response = llm_client.create_message(
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
                temperature=plan_temperature,
                max_tokens=plan_max_tokens
            )
            tokens = TokenUsage()
        
        llm_duration_seconds = time.time() - llm_start_time
        
        # Log the raw LLM response
        logger.info("=" * 80)
        logger.info(f"🗺️ [{self.agent_name}] RAW LLM RESPONSE:")
        logger.info("=" * 80)
        logger.info(f"🗺️ [{self.agent_name}] Response Content:\n{response}")
        logger.info("=" * 80)
        
        # Record LLM call
        if obs_logger:
            llm_call = obs_logger.record_llm_call(
                component=f"agent.{self.agent_name}.plan",
                provider=llm_config.provider,
                model=llm_config.model,
                tokens=tokens,
                duration_seconds=llm_duration_seconds,
                system_prompt_length=len(system_prompt),
                messages_count=1,
                temperature=plan_temperature,
                max_tokens=plan_max_tokens,
                function_name="_create_task_plan"
            )
            
            # Full observability logging for token calculation
            logger.info(f"📋 [{self.agent_name}] 📊 Plan Generation Token Usage Details:")
            logger.info(f"📋 [{self.agent_name}]   Input tokens: {tokens.input_tokens}")
            logger.info(f"📋 [{self.agent_name}]   Output tokens: {tokens.output_tokens}")
            logger.info(f"📋 [{self.agent_name}]   Total tokens: {tokens.total_tokens}")
            logger.info(f"📋 [{self.agent_name}]   LLM Duration: {llm_duration_seconds * 1000:.2f}ms ({llm_duration_seconds:.3f}s)")
            logger.info(f"📋 [{self.agent_name}]   Model: {llm_config.model}")
            logger.info(f"📋 [{self.agent_name}]   Provider: {llm_config.provider}")
            logger.info(f"📋 [{self.agent_name}]   Temperature: {plan_temperature}")
            logger.info(f"📋 [{self.agent_name}]   Max tokens: {plan_max_tokens}")
            logger.info(f"📋 [{self.agent_name}]   System prompt length: {len(system_prompt)} chars")
            logger.info(f"📋 [{self.agent_name}]   User prompt length: {len(prompt)} chars")
            
            # Log cost if available
            if llm_call and llm_call.cost and settings.cost_tracking_enabled:
                logger.info(f"📋 [{self.agent_name}]   Cost: ${llm_call.cost.total_cost_usd:.6f}")
                logger.info(f"📋 [{self.agent_name}]     - Input cost: ${llm_call.cost.input_cost_usd:.6f}")
                logger.info(f"📋 [{self.agent_name}]     - Output cost: ${llm_call.cost.output_cost_usd:.6f}")
            
            # Record tokens in token tracker for component-level tracking
            obs_logger.token_tracker.record_tokens(
                component=f"agent.{self.agent_name}.plan",
                input_tokens=tokens.input_tokens,
                output_tokens=tokens.output_tokens
            )
        
        # Parse and add tasks
        try:
            # Extract JSON from response
            json_str = self._extract_json(response)
            data = json.loads(json_str)
            for task_data in data.get("tasks", []):
                plan.add_task(
                    description=task_data.get("description", ""),
                    tool=task_data.get("tool"),
                    params=task_data.get("params", {}),
                    depends_on=task_data.get("depends_on", [])
                )
            
            # Extract required tools from plan tasks
            plan.required_tools = plan.extract_required_tools()

            if plan.required_tools:
                logger.info(
                    f"📋 [{self.agent_name}] ✅ Extracted {len(plan.required_tools)} "
                    f"required tools: {plan.required_tools}"
                )
            else:
                logger.info(
                    f"📋 [{self.agent_name}] ℹ️  No tools required "
                    f"(all tasks conversational) - will use full catalog"
                )

            # Initialize tool scoping state
            plan.all_tools_enabled = False
            plan.tool_expansion_requested_at = None
        except Exception as e:
            logger.error(f"📋 [{self.agent_name}] Failed to parse plan: {e}")
            # Fallback: single generic task
            plan.add_task(
                description="Complete the objective",
                tool=None,
                params={}
            )
            # Initialize tool scoping state even for fallback
            plan.required_tools = []
            plan.all_tools_enabled = False
            plan.tool_expansion_requested_at = None
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from LLM response (handles markdown code blocks)."""
        import re
        # Try to find JSON in code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            return json_match.group(1)
        # Try to find JSON object directly
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json_match.group(0)
        # Return as-is if no JSON found
        return text

    # ==================== MAIN ENTRY POINT ====================

    async def process_message_with_log(
        self,
        session_id: str,
        user_message: str,
        execution_log: ExecutionLog
    ) -> Tuple[str, ExecutionLog]:
        """Process message and return response with execution log."""
        self._execution_log[session_id] = execution_log
        # Call the new agentic version
        response, execution_log = await self.process_message(session_id, user_message, execution_log)
        return response, execution_log

    def _log_execution_summary(self, exec_context: ExecutionContext):
        """Log execution summary as JSON."""
        exec_summary = exec_context.to_dict()
        exec_summary_json = json.dumps(exec_summary, indent=2, default=str)
        logger.info(f"🧱 [{self.agent_name}] EXECUTION SUMMARY:\n{exec_summary_json}")
    
    async def process_message(
        self,
        session_id: str,
        message: str,
        execution_log: Optional[ExecutionLog] = None
    ) -> Tuple[str, ExecutionLog]:
        """
        Process a message with full execution logging using the new agentic loop.
        
        This is the main agentic loop:
        1. Initialize context and criteria
        2. Think → Act → Observe → Repeat
        3. Handle result types appropriately
        4. Generate response when appropriate
        """
        # Initialize execution log
        if execution_log is None:
            execution_log = ExecutionLog(tools_used=[])
        
        # Get context (assessor already ran, routing decided)
        context = self._context.get(session_id, {})
        routing_action = context.get('routing_action', '')
        continuation_context = context.get('continuation_context', {})
        
        # ═══════════════════════════════════════════════════════════════════════
        # NON-BRAINER PRE-EXECUTION
        # ═══════════════════════════════════════════════════════════════════════
        # When user confirms with "yes", we don't need to think - just do it.
        # Execute the pending tool immediately, then let normal flow review.
        # ═══════════════════════════════════════════════════════════════════════
        
        pre_executed_observation = None  # Will hold (type, name, result) if we pre-execute
        
        if routing_action == 'execute_confirmed_action':
            # Extract pending action from continuation context
            pending_action = continuation_context.get('pending_action', {})
            pending_tool = (
                pending_action.get('tool') or 
                continuation_context.get('entities', {}).get('pending_tool')
            )
            pending_tool_input = (
                pending_action.get('tool_input') or 
                continuation_context.get('entities', {}).get('pending_tool_input', {})
            )
            
            if pending_tool:
                logger.info(f"[{self.agent_name}] ⚡ NON-BRAINER: Executing confirmed action: {pending_tool}")
                logger.info(f"[{self.agent_name}] ⚡ Input: {json.dumps(pending_tool_input, default=str) if pending_tool_input else '{}'}")
                
                # Ensure session_id in tool input
                if "session_id" not in pending_tool_input:
                    pending_tool_input["session_id"] = session_id
                
                # Execute immediately - no thinking required
                tool_result = await self._execute_tool(pending_tool, pending_tool_input, execution_log)

                logger.info(f"[{self.agent_name}] ⚡ Result: success={tool_result.get('success')}")

                # ═══════════════════════════════════════════════════════════════
                # FAST PATH: If tool succeeded, skip _think() entirely
                # ═══════════════════════════════════════════════════════════════
                if tool_result.get('success'):
                    logger.info(f"[{self.agent_name}] ⚡ FAST PATH: Tool succeeded, generating response directly")

                    # Clear continuation context - action has been taken
                    self.state_manager.clear_continuation_context(session_id)

                    # Build AgentResponseData from tool result
                    from patient_ai_service.models.agentic import AgentResponseData
                    response_data = AgentResponseData(
                        completion_summary=tool_result.get('message', f"{pending_tool} completed successfully"),
                        completion_details=tool_result.get('data', tool_result),
                        entities=pending_tool_input  # Pass the entities used
                    )

                    # Create minimal exec_context for response generation
                    exec_context = ExecutionContext(session_id, 1, user_request=message)
                    exec_context.add_observation("tool", pending_tool, tool_result)

                    # Generate response directly (skip planning, skip _think)
                    response = await self._generate_response(
                        session_id,
                        AgentDecision.RESPOND_COMPLETE,
                        response_data,
                        exec_context
                    )

                    return response, execution_log

                # Tool failed - fall through to normal agentic flow for error handling
                logger.info(f"[{self.agent_name}] ⚡ Tool failed, falling through to normal agentic flow")

                # Store for injection into exec_context later
                pre_executed_observation = ("tool", pending_tool, tool_result)

                # Clear continuation context - action has been taken (even on failure)
                self.state_manager.clear_continuation_context(session_id)
            else:
                logger.warning(f"[{self.agent_name}] execute_confirmed_action but no pending_tool found!")
        
        # NOTE: Rejection and modification are NOT special-cased here
        # They go through normal planning and thinking loop
        # UnifiedReasoning will set:
        #   - situation_type: "rejection" or "modification"
        #   - continuation_type: "rejection" or "modification"
        #   - plan_decision: "resume"
        # The agent's _think() will see pending_action context and decide appropriately
        
        # ═══════════════════════════════════════════════════════════════
        # PLAN LIFECYCLE MANAGEMENT (Phase 4)
        # ═══════════════════════════════════════════════════════════════

        # Check if planning is enabled for this session (A/B testing support)
        planning_enabled = FeatureFlags.is_enabled("planning", session_id)

        plan = None

        if not planning_enabled:
            # Planning disabled - skip plan creation entirely
            logger.info(f"📋 [{self.agent_name}] Planning DISABLED (feature flag). Proceeding without plan.")
            plan = None
            # NOTE: exec_context.task_context will still be set below to preserve
            # objective, entities, constraints for _think()
        else:
            # Original plan lifecycle logic (planning enabled)
            plan_action_str = context.get("plan_action", "create_new")
            existing_plan_data = context.get("existing_plan")

            if plan_action_str == "no_plan":
                # Skip planning entirely (fast-path or general assistant)
                logger.info(f"📋 [{self.agent_name}] Skipping plan creation (no_plan action)")
                plan = None

            elif plan_action_str == "create_new" or plan_action_str == "abandon_create":
                # Generate new plan
                logger.info(f"📋 [{self.agent_name}] Creating new plan...")
                plan = await self._plan(
                    session_id=session_id,
                    objective=context.get("objective", ""),
                    entities=context.get("entities", {}),
                    constraints=context.get("constraints", [])
                )
                # NEW: Initialize plan with llm_entities from context
                if context.get("llm_entities"):
                    plan.llm_entities = context["llm_entities"].copy()
                    logger.info(f"📋 [{self.agent_name}] Initialized plan with {len(plan.llm_entities)} LLM entities")
                self.state_manager.store_agent_plan(session_id, plan)

            elif plan_action_str in ["resume", "update_resume"]:
                # Load existing plan
                if existing_plan_data:
                    plan = AgentPlan(**existing_plan_data)
                    logger.info(f"📋 [{self.agent_name}] Resuming plan: {plan.get_summary()}")

                    if plan_action_str == "update_resume":
                        # Add new entities from this turn
                        new_entities = context.get("entities", {})
                        plan.unblock_with_info(new_entities)
                        logger.info(f"📋 [{self.agent_name}] Updated plan with: {list(new_entities.keys())}")
                        logger.info(f"🧠 [{self.agent_name}] Plan status: {plan.status.value}")

                        # NEW: Restore llm_entities from context if resuming
                        if context.get("llm_entities"):
                            plan.update_llm_entities_with_fifo(context["llm_entities"])
                            logger.info(f"📋 [{self.agent_name}] Restored {len(context['llm_entities'])} LLM entities to plan")

                        self.state_manager.store_agent_plan(session_id, plan)
                else:
                    # Fallback: create new if no existing plan data
                    logger.warning(f"📋 [{self.agent_name}] No existing plan data for {plan_action_str}, creating new")
                    plan = await self._plan(
                        session_id=session_id,
                        objective=context.get("objective", context.get("user_intent", "")),
                        entities=context.get("entities", {})
                    )
                    # NEW: Initialize plan with llm_entities from context
                    if context.get("llm_entities"):
                        plan.llm_entities = context["llm_entities"].copy()
                        logger.info(f"📋 [{self.agent_name}] Initialized plan with {len(plan.llm_entities)} LLM entities")
                    self.state_manager.store_agent_plan(session_id, plan)
            else:
                # Fallback: create new
                logger.warning(f"📋 [{self.agent_name}] Unknown plan_action: {plan_action_str}, creating new")
                plan = await self._plan(
                    session_id=session_id,
                    objective=context.get("objective", context.get("user_intent", "")),
                    entities=context.get("entities", {})
                )
                # NEW: Initialize plan with llm_entities from context
                if context.get("llm_entities"):
                    plan.llm_entities = context["llm_entities"].copy()
                    logger.info(f"📋 [{self.agent_name}] Initialized plan with {len(plan.llm_entities)} LLM entities")
                self.state_manager.store_agent_plan(session_id, plan)
        
        # Initialize execution context
        exec_context = ExecutionContext(session_id, self.max_iterations, user_request=message)
        exec_context.plan = plan  # NEW: Set plan in execution context
        
        # NEW: Set task_context from orchestrator context (includes objective, entities, etc.)
        exec_context.task_context = {
            "user_intent": context.get("user_intent", ""),
            "objective": context.get("objective", ""),
            "entities": context.get("entities", {}),
            "success_criteria": context.get("success_criteria", []),
            "constraints": context.get("constraints", []),
            "prior_context": context.get("prior_context"),
            "is_continuation": context.get("is_continuation", False),
            "continuation_type": context.get("continuation_type"),
            "selected_option": context.get("selected_option"),
            "action": context.get("action", context.get("routing_action", "")),
            "planning_enabled": planning_enabled,  # Track for debugging
        }
        
        # Load continuation context if resuming
        continuation = context.get("continuation_context", {})
        if continuation:
            exec_context.continuation_context = continuation
            logger.info(f"Resuming with continuation context: {list(continuation.keys())}")
        
        # ═══════════════════════════════════════════════════════════════════════
        # INJECT PRE-EXECUTED OBSERVATION (if we did a non-brainer execution)
        # ═══════════════════════════════════════════════════════════════════════
        if pre_executed_observation:
            obs_type, obs_name, obs_result = pre_executed_observation
            exec_context.add_observation(obs_type, obs_name, obs_result)
            logger.info(f"[{self.agent_name}] 💉 Injected pre-executed observation: {obs_name}")
            
            # Also mark the corresponding task as complete if it exists in plan
            if plan and obs_result.get('success'):
                from patient_ai_service.models.agent_plan import TaskStatus
                # Find and mark matching task
                for task in plan.tasks:
                    if task.tool == obs_name and task.status == TaskStatus.PENDING:
                        plan.mark_task_complete(task.id, obs_result)
                        logger.info(f"[{self.agent_name}] 📋 Marked task {task.id} complete from pre-execution")
                        break
        
        logger.info(f"[{self.agent_name}] Starting agentic loop for session {session_id}")
        logger.info(f"[{self.agent_name}] Message: {message[:100]}...")
        if exec_context.plan:
            logger.info(f"[{self.agent_name}] Plan: {exec_context.plan.objective} ({len(exec_context.plan.tasks)} tasks)")
        else:
            logger.warning(f"[{self.agent_name}] No plan in execution context!")
        
        # Log execution context as JSON
        exec_context_json = json.dumps(exec_context.to_dict(), indent=2, default=str)
        logger.info(f"🧱 [{self.agent_name}] EXECUTION CONTEXT:\n{exec_context_json}")
        
        # =====================================================================
        # MAIN AGENTIC LOOP
        # =====================================================================
        
        while exec_context.iteration < self.max_iterations:
            exec_context.iteration += 1
            exec_context.llm_calls += 1
            iteration = exec_context.iteration
            
            logger.info(f"\n{'='*60}")
            logger.info(f"[{self.agent_name}] ITERATION {iteration}/{self.max_iterations}")
            logger.info(f"{'='*60}")
            
            # -----------------------------------------------------------------
            # THINK: Analyze situation and decide action
            # -----------------------------------------------------------------
            
            thinking = await self._think(session_id, message, exec_context)
            
            logger.info(f"[{self.agent_name}] Decision: {thinking.decision}")
            logger.info(f"[{self.agent_name}] Reasoning: {thinking.reasoning[:100]}...")

            # NOTE: entities are persisted through ContinuationContext
            # (set via set_continuation_context when CLARIFY/COLLECT_INFO decisions
            # pause execution), NOT through GlobalState. See state.py for details.
            if getattr(thinking, "updated_entities", None):
                logger.info(
                    f"✍️ [{self.agent_name}]   → Resolved entities available: "
                    f"{list(thinking.updated_entities.keys())} "
                    f"(will persist via ContinuationContext if needed)"
                )

                # ✅ NEW: Persist updated entities back to plan for next iteration
                # This ensures delta updates accumulate across iterations
                if exec_context.plan:
                    exec_context.plan.update_entities_with_fifo(thinking.updated_entities)
                    logger.info(
                        f"✍️ [{self.agent_name}]   → Updated plan.entities (FIFO): "
                        f"{list(exec_context.plan.entities.keys())}"
                    )

                    # Update LLM-only entities in plan (if exists)
                    if thinking.llm_delta:
                        # Update plan (for current execution context)
                        exec_context.plan.update_llm_entities_with_fifo(thinking.llm_delta)
                        logger.info(
                            f"📝 [{self.agent_name}]   → Updated plan.llm_entities: +{len(thinking.llm_delta)} "
                            f"(Plan: {len(exec_context.plan.llm_entities)})"
                        )

            # MOVED OUTSIDE plan block: Persist LLM entities to GlobalState regardless of plan existence
            # This ensures entities persist even when planning is disabled
            if thinking.llm_delta:
                self.state_manager.update_llm_entities(session_id, thinking.llm_delta)
                logger.info(
                    f"📝 [{self.agent_name}]   → Persisted llm_entities to GlobalState: +{len(thinking.llm_delta)} "
                    f"(Global: {len(self.state_manager.get_llm_entities(session_id))})"
                )

            # -----------------------------------------------------------------
            # HANDLE TOOL EXPANSION REQUEST
            # -----------------------------------------------------------------
            
            if thinking.request_all_tools and exec_context.plan and not exec_context.plan.all_tools_enabled:
                logger.info(f"🔧 [{self.agent_name}] Tool expansion requested!")
                logger.info(f"🔧 [{self.agent_name}]   Reason: {thinking.tool_expansion_reason or 'Not specified'}")
                
                # Enable all tools for this plan
                exec_context.plan.all_tools_enabled = True
                exec_context.plan.tool_expansion_requested_at = exec_context.iteration
                
                # Store updated plan
                self.state_manager.store_agent_plan(session_id, exec_context.plan)
                
                logger.info(f"🔧 [{self.agent_name}]   ✅ Full tool catalog enabled")
                logger.info(f"🔧 [{self.agent_name}]   → Re-calling _think() with ALL tools...")
                
                # Record expansion event
                if settings.enable_observability:
                    obs_logger = get_observability_logger(session_id)
                    if obs_logger:
                        try:
                            obs_logger.record_custom_event(
                                event_type="tool_expansion",
                                data={
                                    "iteration": exec_context.iteration,
                                    "reason": thinking.tool_expansion_reason,
                                    "original_tool_count": len(exec_context.plan.required_tools),
                                    "total_tool_count": len(self._tool_schemas)
                                }
                            )
                        except Exception as obs_error:
                            logger.warning(f"🔧 [{self.agent_name}] Error in observability logging (non-critical): {obs_error}")
                
                # Re-run _think() with expanded tools (don't increment iteration)
                exec_context.iteration -= 1  # Compensate for loop increment
                continue  # Jump to next iteration
            
            # -----------------------------------------------------------------
            # ACT: Execute based on decision
            # -----------------------------------------------------------------
            
            if thinking.decision == AgentDecision.CALL_TOOL:
                # Execute tool
                tool_name = thinking.tool_name
                tool_input = thinking.tool_input or {}
                
                # Resolve any template placeholders in tool_input (e.g., {{task_1.doctor_id}})
                tool_input = self._resolve_tool_input_templates(session_id, tool_input, exec_context)
                
                # Inject session_id if not present (tools require it as first parameter)
                if "session_id" not in tool_input:
                    tool_input["session_id"] = session_id
                
                logger.info(f"[{self.agent_name}] Calling tool: {tool_name}")
                logger.info(f"[{self.agent_name}] Input: {json.dumps(tool_input, indent=2, default=str)}")
                
                tool_result = await self._execute_tool(tool_name, tool_input, execution_log)
                
                logger.info(f"[{self.agent_name}] Result: {json.dumps(tool_result, indent=2, default=str)[:500]}")
                
                # Process result - may override next action
                override = await self._process_tool_result(
                    session_id, tool_name, tool_result, exec_context
                )
                
                # UPDATE PLAN STATE (Phase 4)
                if exec_context.plan and thinking.current_task_id:
                    task_id = thinking.current_task_id
                    if tool_result.get("success"):
                        exec_context.plan.mark_task_complete(task_id, tool_result)
                        logger.info(f"📋 [{self.agent_name}] Task {task_id} marked complete")
                    elif tool_result.get("result_type") == "user_input_needed":
                        exec_context.plan.mark_task_blocked(
                            task_id,
                            reason=tool_result.get("message", "Need more info"),
                            awaiting=tool_result.get("missing_field", "information"),
                            options=tool_result.get("alternatives")
                        )
                        logger.info(f"📋 [{self.agent_name}] Task {task_id} marked blocked")
                    else:
                        # Handle other result types (failed, etc.)
                        task = exec_context.plan.get_task(task_id)
                        if task:
                            task.status = TaskStatus.FAILED
                            task.failed_reason = tool_result.get("message", "Task failed")
                            logger.warning(f"📋 [{self.agent_name}] Task {task_id} marked failed")
                    
                    # Save updated plan
                    self.state_manager.store_agent_plan(session_id, exec_context.plan)
                
                if override:
                    # Handle EXECUTE_RECOVERY immediately (don't wait for next iteration)
                    if override == AgentDecision.EXECUTE_RECOVERY:
                        # Find the most recent recovery attempt
                        recovery_info = None
                        for sig, info in exec_context.recovery_attempts.items():
                            if info.get("attempted"):
                                if not recovery_info or info.get("iteration", 0) > recovery_info.get("iteration", 0):
                                    recovery_info = info
                        
                        if recovery_info and recovery_info.get("recovery_action"):
                            recovery_tool = recovery_info["recovery_action"]
                            recovery_message = recovery_info.get("recovery_message")
                            
                            logger.info(
                                f"[{self.agent_name}] 🔧 Executing recovery action: {recovery_tool}"
                                + (f" ({recovery_message})" if recovery_message else "")
                            )
                            
                            # Execute recovery tool
                            recovery_input = {"session_id": session_id}
                            recovery_result = await self._execute_tool(recovery_tool, recovery_input, execution_log)
                            
                            logger.info(
                                f"[{self.agent_name}] Recovery result: "
                                f"success={recovery_result.get('success')}, "
                                f"result_type={recovery_result.get('result_type')}"
                            )
                            
                            # Process recovery result - may override again
                            recovery_override = await self._process_tool_result(
                                session_id, recovery_tool, recovery_result, exec_context
                            )
                            
                            if recovery_override:
                                response = await self._handle_override(session_id, recovery_override, exec_context)
                                if response:
                                    return response, execution_log

                            # Recovery executed, continue loop
                            exec_context.recovery_executed = False
                            continue

                    # Handle other overrides normally
                    response = await self._handle_override(session_id, override, exec_context)
                    if response:
                        return response, execution_log
            
            elif thinking.decision == AgentDecision.EXECUTE_RECOVERY:
                # Execute recovery action automatically
                # Find the most recent recovery attempt
                recovery_info = None
                for sig, info in exec_context.recovery_attempts.items():
                    if info.get("attempted"):
                        if not recovery_info or info.get("iteration", 0) > recovery_info.get("iteration", 0):
                            recovery_info = info
                
                if recovery_info and recovery_info.get("recovery_action"):
                    recovery_tool = recovery_info["recovery_action"]
                    recovery_message = recovery_info.get("recovery_message")
                    
                    logger.info(
                        f"[{self.agent_name}] 🔧 Executing recovery action: {recovery_tool}"
                        + (f" ({recovery_message})" if recovery_message else "")
                    )
                    
                    # Execute recovery tool with session_id
                    recovery_input = {"session_id": session_id}
                    # Could pass original error context if needed in future
                    
                    recovery_result = await self._execute_tool(recovery_tool, recovery_input, execution_log)
                    
                    logger.info(
                        f"[{self.agent_name}] Recovery result: "
                        f"success={recovery_result.get('success')}, "
                        f"result_type={recovery_result.get('result_type')}"
                    )
                    
                    # Process recovery result - may override next action
                    override = await self._process_tool_result(
                        session_id, recovery_tool, recovery_result, exec_context
                    )
                    
                    if override:
                        response = await self._handle_override(session_id, override, exec_context)
                        if response:
                            return response, execution_log

                    # Recovery executed, continue loop to see if it helped
                    exec_context.recovery_executed = False  # Reset flag
                    continue
                else:
                    # No recovery info found - shouldn't happen, but fallback to continue
                    logger.warning(f"[{self.agent_name}] EXECUTE_RECOVERY but no recovery_info found")
                    continue
            
            elif thinking.decision == AgentDecision.RESPOND:
                # Agent wants to respond - verify completion first
                completion = exec_context.check_completion()
                
                if completion.is_complete:
                    logger.info(f"[{self.agent_name}] ✅ Task complete! Generating response.")
                    
                    # Extract suggested_response
                    suggested = (
                        thinking.response.suggested_response if thinking.response and thinking.response.suggested_response else None
                    ) or exec_context.suggested_response
                    
                    if suggested and is_humanizer_enabled(session_id):
                        # NEW PATH: Use humanizer
                        # Store suggested_response (pre-humanization) in execution_log for ConversationMemory
                        execution_log.suggested_response = suggested
                        user_intent = self._get_user_intent(session_id, exec_context)
                        response = await self._humanize_response(session_id, suggested, user_intent)
                        self._log_execution_summary(exec_context)
                        return response, execution_log
                    elif suggested:
                        # ROLLBACK: Return suggested_response as-is (no humanization)
                        self._log_execution_summary(exec_context)
                        return suggested, execution_log
                    else:
                        # FALLBACK: Use existing _generate_response
                        response = await self._generate_response(
                            session_id,
                            AgentDecision.RESPOND_COMPLETE,
                            thinking.response,
                            exec_context
                        )
                        return response, execution_log
                
                elif completion.has_blocked:
                    logger.info(f"[{self.agent_name}] ⏸️ Criteria blocked - presenting options")
                    # Update response data with options from completion check
                    if not thinking.response.options:
                        thinking.response.options = []
                        for criterion_id, options in completion.blocked_options.items():
                            thinking.response.options.extend(options)
                    if not thinking.response.options_context:
                        thinking.response.options_context = "alternatives"
                    
                    # Extract suggested_response
                    suggested = (
                        thinking.response.suggested_response if thinking.response and thinking.response.suggested_response else None
                    ) or exec_context.suggested_response
                    
                    if suggested and is_humanizer_enabled(session_id):
                        # NEW PATH: Use humanizer
                        # Store suggested_response (pre-humanization) in execution_log for ConversationMemory
                        execution_log.suggested_response = suggested
                        user_intent = self._get_user_intent(session_id, exec_context)
                        response = await self._humanize_response(session_id, suggested, user_intent)
                        self._log_execution_summary(exec_context)
                        return response, execution_log
                    elif suggested:
                        # ROLLBACK: Return suggested_response as-is (no humanization)
                        self._log_execution_summary(exec_context)
                        return suggested, execution_log
                    else:
                        # FALLBACK: Use existing _generate_response
                        response = await self._generate_response(
                            session_id,
                            AgentDecision.RESPOND_WITH_OPTIONS,
                            thinking.response,
                            exec_context
                        )
                        return response, execution_log
                
                elif completion.has_failed:
                    logger.info(f"[{self.agent_name}] ❌ Criteria failed - explaining")
                    # Update response data with failure info
                    if not thinking.response.failure_reason:
                        thinking.response.failure_reason = "Task could not be completed"
                    
                    # Extract suggested_response
                    suggested = (
                        thinking.response.suggested_response if thinking.response and thinking.response.suggested_response else None
                    ) or exec_context.suggested_response
                    
                    if suggested and is_humanizer_enabled(session_id):
                        # NEW PATH: Use humanizer
                        # Store suggested_response (pre-humanization) in execution_log for ConversationMemory
                        execution_log.suggested_response = suggested
                        user_intent = self._get_user_intent(session_id, exec_context)
                        response = await self._humanize_response(session_id, suggested, user_intent)
                        self._log_execution_summary(exec_context)
                        return response, execution_log
                    elif suggested:
                        # ROLLBACK: Return suggested_response as-is (no humanization)
                        self._log_execution_summary(exec_context)
                        return suggested, execution_log
                    else:
                        # FALLBACK: Use existing _generate_response
                        response = await self._generate_response(
                            session_id,
                            AgentDecision.RESPOND_IMPOSSIBLE,
                            thinking.response,
                            exec_context
                        )
                        return response, execution_log
                
                else:
                    # Not complete but agent thinks it is
                    if thinking.is_task_complete:
                        # Agent explicitly marked complete - trust it and generate response
                        logger.info(f"[{self.agent_name}] Agent marked complete - generating response")
                        
                        # Extract suggested_response
                        suggested = (
                            thinking.response.suggested_response if thinking.response and thinking.response.suggested_response else None
                        ) or exec_context.suggested_response
                        
                        if suggested and is_humanizer_enabled(session_id):
                            # NEW PATH: Use humanizer
                            user_intent = self._get_user_intent(session_id, exec_context)
                            response = await self._humanize_response(session_id, suggested, user_intent)
                            self._log_execution_summary(exec_context)
                            return response, execution_log
                        elif suggested:
                            # ROLLBACK: Return suggested_response as-is (no humanization)
                            self._log_execution_summary(exec_context)
                            return suggested, execution_log
                        else:
                            # FALLBACK: Use existing _generate_response
                            response = await self._generate_response(
                                session_id,
                                AgentDecision.RESPOND_COMPLETE,
                                thinking.response,
                                exec_context
                            )
                            return response, execution_log
                    else:
                        # Force continue
                        logger.warning(
                            f"[{self.agent_name}] Agent tried to respond but task incomplete. "
                            f"Pending: {completion.pending_criteria}"
                        )
                        exec_context.add_observation(
                            "system", "completion_check",
                            {
                                "message": "Task not complete yet",
                                "pending": completion.pending_criteria
                            }
                        )
                        continue
            
            elif thinking.decision == AgentDecision.RESPOND_WITH_OPTIONS:
                logger.info(f"[{self.agent_name}] Responding with options")
                
                # Extract suggested_response
                suggested = (
                    thinking.response.suggested_response if thinking.response and thinking.response.suggested_response else None
                ) or exec_context.suggested_response
                
                if suggested and is_humanizer_enabled(session_id):
                    # NEW PATH: Use humanizer
                    user_intent = self._get_user_intent(session_id, exec_context)
                    response = await self._humanize_response(session_id, suggested, user_intent)
                    self._log_execution_summary(exec_context)
                    return response, execution_log
                elif suggested:
                    # ROLLBACK: Return suggested_response as-is (no humanization)
                    self._log_execution_summary(exec_context)
                    return suggested, execution_log
                else:
                    # FALLBACK: Use existing _generate_response
                    response = await self._generate_response(
                        session_id,
                        AgentDecision.RESPOND_WITH_OPTIONS,
                        thinking.response,
                        exec_context
                    )
                    self._log_execution_summary(exec_context)
                    return response, execution_log
            
            elif thinking.decision == AgentDecision.RESPOND_COMPLETE:
                logger.info(f"[{self.agent_name}] Task complete - generating response")
                
                # Extract suggested_response
                suggested = (
                    thinking.response.suggested_response if thinking.response and thinking.response.suggested_response else None
                ) or exec_context.suggested_response
                
                if suggested and is_humanizer_enabled(session_id):
                    # NEW PATH: Use humanizer
                    user_intent = self._get_user_intent(session_id, exec_context)
                    response = await self._humanize_response(session_id, suggested, user_intent)
                    self._log_execution_summary(exec_context)
                    return response, execution_log
                elif suggested:
                    # ROLLBACK: Return suggested_response as-is (no humanization)
                    self._log_execution_summary(exec_context)
                    return suggested, execution_log
                else:
                    # FALLBACK: Use existing _generate_response
                    response = await self._generate_response(
                        session_id,
                        AgentDecision.RESPOND_COMPLETE,
                        thinking.response,
                        exec_context
                    )
                    self._log_execution_summary(exec_context)
                    return response, execution_log
            
            elif thinking.decision == AgentDecision.RESPOND_IMPOSSIBLE:
                logger.info(f"[{self.agent_name}] Task impossible")
                
                # Extract suggested_response
                suggested = (
                    thinking.response.suggested_response if thinking.response and thinking.response.suggested_response else None
                ) or exec_context.suggested_response
                
                if suggested and is_humanizer_enabled(session_id):
                    # NEW PATH: Use humanizer
                    user_intent = self._get_user_intent(session_id, exec_context)
                    response = await self._humanize_response(session_id, suggested, user_intent)
                    self._log_execution_summary(exec_context)
                    return response, execution_log
                elif suggested:
                    # ROLLBACK: Return suggested_response as-is (no humanization)
                    self._log_execution_summary(exec_context)
                    return suggested, execution_log
                else:
                    # FALLBACK: Use existing _generate_response
                    response = await self._generate_response(
                        session_id,
                        AgentDecision.RESPOND_IMPOSSIBLE,
                        thinking.response,
                        exec_context
                    )
                    return response, execution_log
            
            elif thinking.decision == AgentDecision.CLARIFY:
                logger.info(f"[{self.agent_name}] Asking for clarification - saving state")

                # Save continuation context using response data
                awaiting = thinking.response.clarification_needed or thinking.awaiting_info or "clarification"
                # Use updated_entities as the canonical view if available
                resolved = thinking.updated_entities or \
                    thinking.response.entities or \
                    thinking.entities or \
                    self._extract_entities(exec_context)

                # Get LLM entities from plan
                llm_entities = exec_context.plan.llm_entities.copy() if exec_context.plan and exec_context.plan.llm_entities else {}

                self.state_manager.set_continuation_context(
                    session_id,
                    awaiting=awaiting,
                    options=[],
                    original_request=exec_context.user_request,
                    entities=resolved,
                    llm_entities=llm_entities,  # NEW
                    blocked_criteria=[]
                )
                
                self.state_manager.update_agentic_state(
                    session_id,
                    status="blocked",
                    active_agent=self.agent_name
                )

                # Mark plan as BLOCKED and persist for next turn (Fix 9)
                self._mark_plan_blocked_and_store(session_id, exec_context, awaiting)

                logger.info(f"[{self.agent_name}] Saved continuation: awaiting={awaiting}")

                # Extract suggested_response
                suggested = (
                    thinking.response.suggested_response if thinking.response and thinking.response.suggested_response else None
                ) or exec_context.suggested_response
                
                if suggested and is_humanizer_enabled(session_id):
                    # NEW PATH: Use humanizer
                    user_intent = self._get_user_intent(session_id, exec_context)
                    response = await self._humanize_response(session_id, suggested, user_intent)
                    return response, execution_log
                elif suggested:
                    # ROLLBACK: Return suggested_response as-is (no humanization)
                    return suggested, execution_log
                else:
                    # FALLBACK: Use existing _generate_response
                    response = await self._generate_response(
                        session_id,
                        AgentDecision.CLARIFY,
                        thinking.response,
                        exec_context
                    )
                    return response, execution_log
            
            elif thinking.decision == AgentDecision.COLLECT_INFORMATION:
                logger.info(f"[{self.agent_name}] Collecting information - saving state and generating response")

                # Save continuation context using response data
                information_needed = thinking.response.information_needed or thinking.awaiting_info or "user_information"
                # Use updated_entities as the canonical view if available
                resolved = thinking.updated_entities or \
                    thinking.response.entities or \
                    thinking.entities or \
                    self._extract_entities(exec_context)

                # NEW: Build information collection context
                awaiting_context = f"Collecting: {information_needed}"
                information_collection = {
                    "information_needed": information_needed,
                    "information_question": thinking.response.information_question or "",
                    "context": self.agent_name
                }

                # Get LLM entities from plan
                llm_entities = exec_context.plan.llm_entities.copy() if exec_context.plan and exec_context.plan.llm_entities else {}

                self.state_manager.set_continuation_context(
                    session_id,
                    awaiting="information",  # Standardized trigger
                    awaiting_context=awaiting_context,  # NEW
                    information_collection=information_collection,  # NEW
                    options=[],  # No options for free-form input
                    original_request=exec_context.user_request,
                    entities=resolved,
                    llm_entities=llm_entities,  # NEW
                    blocked_criteria=[c.description for c in exec_context.get_blocked_criteria()]
                )

                # ADD: Verification that state was saved correctly
                verification = self.state_manager.get_continuation_context(session_id)
                if not verification:
                    logger.error(f"[{self.agent_name}] CRITICAL: set_continuation_context did not persist!")
                    logger.error(f"[{self.agent_name}] get_continuation_context returned None after set")
                    raise RuntimeError("Failed to set continuation context - verification failed")

                if verification.get('awaiting') != 'information':
                    logger.error(f"[{self.agent_name}] CRITICAL: awaiting field not persisted correctly!")
                    logger.error(f"[{self.agent_name}] Expected: 'information', Got: {verification.get('awaiting')}")
                    raise RuntimeError("Failed to set continuation context - awaiting field lost")

                verified_info_collection = verification.get('information_collection', {})
                if 'information_needed' not in verified_info_collection:
                    logger.error(f"[{self.agent_name}] CRITICAL: information_needed not in persisted context!")
                    logger.error(f"[{self.agent_name}] Expected: {information_collection}")
                    logger.error(f"[{self.agent_name}] Got: {verified_info_collection}")
                    raise RuntimeError("information_needed not persisted correctly")

                logger.info(f"[{self.agent_name}] ✅ Verified continuation context persisted:")
                logger.info(f"[{self.agent_name}]   awaiting='information'")
                logger.info(f"[{self.agent_name}]   information_needed={information_needed}")
                logger.info(f"[{self.agent_name}]   entities={list(resolved.keys())}")
                
                # Update agentic state to blocked
                self.state_manager.update_agentic_state(
                    session_id,
                    status="blocked",
                    active_agent=self.agent_name
                )

                # Mark plan as BLOCKED and persist for next turn (Fix 9)
                self._mark_plan_blocked_and_store(session_id, exec_context, "information")

                logger.info(f"[{self.agent_name}] Saved continuation: awaiting=information, information_needed={information_needed}, resolved={list(resolved.keys())}")

                # Extract suggested_response
                suggested = (
                    thinking.response.suggested_response if thinking.response and thinking.response.suggested_response else None
                ) or exec_context.suggested_response
                
                if suggested and is_humanizer_enabled(session_id):
                    # NEW PATH: Use humanizer
                    user_intent = self._get_user_intent(session_id, exec_context)
                    response = await self._humanize_response(session_id, suggested, user_intent)
                    return response, execution_log
                elif suggested:
                    # ROLLBACK: Return suggested_response as-is (no humanization)
                    return suggested, execution_log
                else:
                    # FALLBACK: Use existing _generate_response
                    response = await self._generate_response(
                        session_id,
                        AgentDecision.COLLECT_INFORMATION,
                        thinking.response,
                        exec_context
                    )
                    return response, execution_log

            elif thinking.decision == AgentDecision.REQUEST_CONFIRMATION:
                logger.info(f"[{self.agent_name}] Requesting confirmation before critical action")
                
                # Check for confirmation data in response or legacy field
                if not thinking.response.confirmation_action and not thinking.confirmation_summary:
                    logger.warning(f"[{self.agent_name}] REQUEST_CONFIRMATION but no confirmation data!")
                    # Extract suggested_response for fallback
                    suggested = (
                        thinking.response.suggested_response if thinking.response and thinking.response.suggested_response else None
                    ) or exec_context.suggested_response
                    
                    if suggested and is_humanizer_enabled(session_id):
                        user_intent = self._get_user_intent(session_id, exec_context)
                        response = await self._humanize_response(session_id, suggested, user_intent)
                        return response, execution_log
                    elif suggested:
                        return suggested, execution_log
                    else:
                        # Fallback to just responding
                        response = await self._generate_response(
                            session_id,
                            AgentDecision.RESPOND,
                            thinking.response,
                            exec_context
                        )
                        return response, execution_log
                
                # Build confirmation details from response or legacy summary
                if thinking.response.confirmation_action:
                    confirmation_details = thinking.response.confirmation_details or {}
                    confirmation_action = thinking.response.confirmation_action
                    # Infer tool name from action (e.g., "book_appointment" -> "book_appointment")
                    pending_tool = confirmation_action if "_" in confirmation_action else None
                    # Build tool input from confirmation details
                    pending_tool_input = confirmation_details.copy()
                else:
                    # Legacy: use confirmation_summary
                    summary = thinking.confirmation_summary
                    if summary:
                        confirmation_details = summary.details
                        confirmation_action = summary.action
                        pending_tool = summary.tool_name
                        pending_tool_input = summary.tool_input
                    else:
                        # Fallback if both are missing
                        confirmation_details = {}
                        confirmation_action = "unknown_action"
                        pending_tool = None
                        pending_tool_input = {}
                
                # NEW: Build human-readable summary for UnifiedReasoning
                awaiting_context = self._build_confirmation_summary(
                    confirmation_action,
                    confirmation_details,
                    pending_tool
                )
                
                # NEW: Build structured pending_action dict
                pending_action = {
                    "tool": pending_tool,
                    "tool_input": pending_tool_input,
                    "action_type": self._get_action_type(confirmation_action),
                    "summary": awaiting_context
                }

                # Get LLM entities from plan
                llm_entities = exec_context.plan.llm_entities.copy() if exec_context.plan and exec_context.plan.llm_entities else {}

                # Save the pending action so we can execute it after confirmation
                self.state_manager.set_continuation_context(
                    session_id,
                    awaiting="confirmation",
                    awaiting_context=awaiting_context,  # NEW
                    pending_action=pending_action,       # NEW
                    options=["yes", "no", "confirm", "cancel"],
                    original_request=exec_context.user_request,
                    entities={
                        # Keep for backward compatibility
                        "pending_action": confirmation_action,
                        "pending_tool": pending_tool,
                        "pending_tool_input": pending_tool_input,
                        **confirmation_details
                    },
                    llm_entities=llm_entities,  # NEW
                    blocked_criteria=[]
                )
                
                self.state_manager.update_agentic_state(
                    session_id,
                    status="awaiting_confirmation",
                    active_agent=self.agent_name
                )

                # Mark plan as BLOCKED and persist for next turn (Fix 9)
                self._mark_plan_blocked_and_store(
                    session_id,
                    exec_context,
                    "confirmation",
                    options=["yes", "no", "confirm", "cancel"]
                )

                # Extract suggested_response
                suggested = (
                    thinking.response.suggested_response if thinking.response and thinking.response.suggested_response else None
                ) or exec_context.suggested_response
                
                if suggested and is_humanizer_enabled(session_id):
                    # NEW PATH: Use humanizer
                    user_intent = self._get_user_intent(session_id, exec_context)
                    response = await self._humanize_response(session_id, suggested, user_intent)
                    return response, execution_log
                elif suggested:
                    # ROLLBACK: Return suggested_response as-is (no humanization)
                    return suggested, execution_log
                else:
                    # FALLBACK: Use existing _generate_response
                    response = await self._generate_response(
                        session_id,
                        AgentDecision.REQUEST_CONFIRMATION,
                        thinking.response,
                        exec_context
                    )
                    return response, execution_log

            elif thinking.decision == AgentDecision.RETRY:
                logger.info(f"[{self.agent_name}] Retrying last action")
                # Don't increment if we just executed recovery (recovery has its own tracking)
                if not exec_context.recovery_executed:
                    exec_context.retry_count += 1
                exec_context.recovery_executed = False  # Reset flag
                continue
        
        # =====================================================================
        # MAX ITERATIONS REACHED
        # =====================================================================
        
        logger.warning(f"[{self.agent_name}] Max iterations ({self.max_iterations}) reached")
        response = self._generate_max_iterations_response(exec_context)
        
        # Update state manager
        self.state_manager.update_agentic_state(
            session_id,
            status="max_iterations",
            iteration=exec_context.iteration
        )
        
        # Log execution summary as JSON
        self._log_execution_summary(exec_context)
        
        return response, execution_log

    async def process_message_legacy(self, session_id: str, user_message: str) -> str:
        """
        [LEGACY] Process a user message using the ReAct agentic loop.

        This is the old implementation. Use process_message() instead.

        This is the main entry point. It implements:

        1. Initialize conversation state

        2. Run agentic loop until task complete or max iterations

        3. Generate final response only after all tools executed

        """
        agent_start_time = time.time()
        obs_logger = get_observability_logger(session_id) if settings.enable_observability else None

        try:
            # Initialize conversation history
            if session_id not in self.conversation_history:
                self.conversation_history[session_id] = []

            # Add user message to history
            self.conversation_history[session_id].append({
                "role": "user",
                "content": user_message
            })

            # Build the execution context that tracks tool results
            execution_context = SimpleExecutionContext(
                user_request=user_message,
                session_id=session_id
            )

            # ==================== AGENTIC LOOP ====================

            iteration = 0
            final_response = None
            while iteration < self.max_iterations:
                iteration += 1
                logger.info(f"[{self.agent_name}] Iteration {iteration}/{self.max_iterations}")

                # STEP 1: THINK - Analyze current state and decide next action
                thinking = await self._think(session_id, execution_context)

                logger.info(
                    f"[{self.agent_name}] Decision: {thinking.decision.value} | "
                    f"Reasoning: {thinking.reasoning[:100]}..."
                )

                # STEP 2: ACT based on decision
                if thinking.decision == AgentDecision.RESPOND:
                    # Task is complete, generate response
                    if thinking.is_task_complete:
                        logger.info(f"[{self.agent_name}] Task complete. Generating response.")
                        final_response = thinking.response_text
                        break
                    else:
                        # Agent wants to respond but hasn't validated completion
                        # Force a completion check
                        logger.warning(
                            f"[{self.agent_name}] Agent wants to respond but task not validated. "
                            f"Forcing completion check."
                        )
                        completion_check = await self._verify_task_completion(
                            session_id, execution_context
                        )
                        if completion_check.is_complete:
                            final_response = completion_check.response
                            break
                        else:
                            # Continue loop with guidance
                            execution_context.add_observation(
                                "system",
                                "completion_check",
                                {"status": "incomplete", "missing": completion_check.missing_items}
                            )
                            continue

                elif thinking.decision == AgentDecision.CALL_TOOL:
                    # Execute the tool
                    if not thinking.tool_name:
                        logger.error(f"[{self.agent_name}] CALL_TOOL decision but no tool_name")
                        execution_context.add_observation(
                            "error",
                            "missing_tool_name",
                            {"error": "No tool specified for CALL_TOOL decision"}
                        )
                        continue

                    tool_result = await self._execute_tool(
                        session_id,
                        thinking.tool_name,
                        thinking.tool_input or {}
                    )

                    # STEP 3: OBSERVE - Record the result
                    execution_context.add_observation(
                        "tool",
                        thinking.tool_name,
                        tool_result
                    )

                    logger.info(
                        f"[{self.agent_name}] Tool '{thinking.tool_name}' executed. "
                        f"Success: {tool_result.get('success', 'error' not in tool_result)}"
                    )

                elif thinking.decision == AgentDecision.RETRY:
                    # Retry with different approach - add guidance to context
                    execution_context.add_observation(
                        "retry",
                        "retry_attempt",
                        {"reason": thinking.reasoning, "attempt": iteration}
                    )
                    logger.info(f"[{self.agent_name}] Retrying with different approach")

                elif thinking.decision == AgentDecision.CLARIFY:
                    # Need clarification from user
                    final_response = thinking.response_text
                    logger.info(f"[{self.agent_name}] Asking for clarification")
                    break

            # ==================== POST-LOOP ====================

            if final_response is None:
                if iteration >= self.max_iterations:
                    logger.warning(
                        f"[{self.agent_name}] Reached max iterations ({self.max_iterations}). "
                        f"Generating best-effort response."
                    )
                # Generate response from whatever we have
                final_response = await self._generate_final_response(
                    session_id, execution_context
                )

            # Add response to history
            self.conversation_history[session_id].append({
                "role": "assistant",
                "content": final_response
            })

            # Trim history
            if len(self.conversation_history[session_id]) > 20:
                self.conversation_history[session_id] = \
                    self.conversation_history[session_id][-20:]

            logger.info(
                f"[{self.agent_name}] Completed in {iteration} iterations, "
                f"{len(execution_context.observations)} observations"
            )

            return final_response

        except Exception as e:
            logger.error(f"Error in {self.agent_name}.process_message: {e}", exc_info=True)
            return self._get_error_response(str(e))

    # ==================== THINKING (REASONING) ====================
    # NOTE: Old duplicate _think methods removed in Phase 4
    # The _think method at line ~3456 is the one to use (modified for plan navigation)
    
    def _build_task_params(
        self,
        task: 'Task',
        plan: 'TaskPlan',
        session_id: str
    ) -> Dict[str, Any]:
        """
        Build tool parameters for a task.
        
        Uses:
        1. Params defined in task
        2. Results from dependent tasks
        3. Entity state (derived entities)
        """
        params = dict(task.params)  # Start with defined params
        
        # Get derived entities for cached values
        entity_state = self.state_manager.get_entity_state(session_id)
        
        # Build params based on tool type
        if task.tool == "check_availability":
            # Get doctor_uuid from previous task or derived entities
            doctor_uuid = params.get("doctor_id")
            if not doctor_uuid:
                # Try from dependent task
                for dep_id in task.depends_on:
                    dep_task = plan.get_task(dep_id)
                    if dep_task and dep_task.result:
                        if dep_task.tool in ["find_doctor_by_name", "list_doctors"]:
                            doctor_data = dep_task.result.get("doctor", {})
                            doctor_uuid = doctor_data.get("id")
                            break
            
            if not doctor_uuid:
                # Try from derived entities
                doctor_entity = entity_state.derived.get_entity("doctor_uuid")
                if doctor_entity:
                    doctor_uuid = doctor_entity.value
            
            params["doctor_id"] = doctor_uuid
            
            # Get date/time from entities
            if not params.get("date"):
                params["date"] = plan.entities.get("date_preference")
            if not params.get("time"):
                params["time"] = plan.entities.get("time_preference")
        
        elif task.tool == "book_appointment":
            # Build from previous tasks and entities
            for dep_id in task.depends_on:
                dep_task = plan.get_task(dep_id)
                if not dep_task or not dep_task.result:
                    continue
                
                if dep_task.tool == "check_availability":
                    avail = dep_task.result
                    if avail.get("available_at_requested_time"):
                        params["time"] = avail.get("confirmed_time") or plan.entities.get("time_preference")
                        params["date"] = avail.get("date") or plan.entities.get("date_preference")
                
                if dep_task.tool in ["find_doctor_by_name", "list_doctors"]:
                    doctor_data = dep_task.result.get("doctor", {})
                    params["doctor_id"] = doctor_data.get("id")
            
            # Get patient_id from entity state
            patient_entity = entity_state.derived.get_entity("patient_id")
            if patient_entity:
                params["patient_id"] = patient_entity.value
            
            # Add procedure if not set
            if not params.get("reason"):
                params["reason"] = plan.entities.get("procedure_preference", "dental appointment")
        
        return params
    
    # NOTE: _generate_task_plan() and _generate_task_plan_llm() removed in Phase 6
    # Replaced with _plan() method that uses AgentPlan instead of TaskPlan
    
    async def _resume_task_plan(
        self,
        session_id: str,
        execution_context: 'ExecutionContext',
        user_selection: Any
    ) -> bool:
        """
        Resume a blocked task plan after user provides input.
        
        Returns:
            True if plan was resumed, False otherwise
        """
        from patient_ai_service.models.task_plan import TaskStatus
        
        # Get saved plan
        plan = self.state_manager.get_task_plan(session_id)
        if not plan:
            return False
        
        # Find blocked task
        blocked = plan.get_blocked_tasks()
        if not blocked:
            return False
        
        blocked_task = blocked[0]
        
        # Update task based on user selection
        # This depends on what kind of selection was needed
        if blocked_task.tool == "check_availability":
            # User selected a time from alternatives
            blocked_task.params["time"] = user_selection
            blocked_task.status = TaskStatus.PENDING
            blocked_task.blocked_reason = None
            blocked_task.blocked_options = None
        
        elif blocked_task.tool == "book_appointment":
            # User confirmed or selected
            blocked_task.status = TaskStatus.PENDING
        
        # Restore plan to execution context
        execution_context.set_task_plan(plan)
        plan.update_metrics()
        
        return True

    def _get_thinking_prompt(
        self,
        session_id: str,
        execution_context: 'ExecutionContext'
    ) -> str:
        """
        DEPRECATED: This method is no longer used.
        
        The thinking prompt is now built using a 3-layer approach:
        - Layer 1 (cached): get_universal_system_content()
        - Layer 2 (not cached): _get_thinking_system_prompt(exec_context)
        - Layer 3 (dynamic): _build_thinking_prompt(message, context, exec_context)
        
        This method is kept for backward compatibility but should not be called.
        Use _get_thinking_system_prompt() and _build_thinking_prompt() instead.
        """
        import warnings
        warnings.warn(
            "_get_thinking_prompt() is deprecated. Use _get_thinking_system_prompt() "
            "and _build_thinking_prompt() instead. This method will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Build thinking prompt using cached static content (legacy implementation).
        
        # STATIC: Agent personality (cacheable per-agent, only patient context changes)
        base_prompt = self._get_system_prompt(session_id)
        
        # STATIC: Use cached instructions (never changes)
        react_instructions = self._cached_react_instructions
        result_type_guide = self._cached_result_type_guide
        decision_guide = self._cached_decision_guide
        tools_description = self._cached_tools_description
        
        # DYNAMIC: Only these change per-iteration
        observations_summary = self._format_observations(execution_context)
        
        # NEW: Get entity status display
        entity_display = ""
        try:
            if session_id:
                entity_display = self.state_manager.get_entity_display(session_id)
        except Exception as e:
            logger.warning(f"Failed to get entity display: {e}")
        
        # NEW: Get plan display (Phase 6 - replaced criteria with plan)
        plan_display = ""
        if execution_context.plan:
            plan = execution_context.plan
            completed = len(plan.get_completed_task_ids())
            total = len(plan.tasks)
            plan_display = f"""Plan: {plan.objective}
Status: {plan.status.value}
Progress: {completed}/{total} tasks complete

Tasks:
"""
            for task in plan.tasks:
                status_icon = "✅" if task.status == TaskStatus.COMPLETE else "⏸️" if task.status == TaskStatus.BLOCKED else "🔄" if task.status == TaskStatus.IN_PROGRESS else "○"
                plan_display += f"  {status_icon} {task.id}: {task.description}\n"
        else:
            plan_display = "No execution plan yet."
        
        # Build prompt with clear static/dynamic separation
        # Static content should be at the BEGINNING for better caching
        full_prompt = f"""
{react_instructions}

{result_type_guide}

{decision_guide}

═══════════════════════════════════════════════════════════════
🛠️ AVAILABLE TOOLS (Use exact parameter names)
═══════════════════════════════════════════════════════════════

{tools_description}

═══════════════════════════════════════════════════════════════
AGENT CONTEXT
═══════════════════════════════════════════════════════════════

{base_prompt}

═══════════════════════════════════════════════════════════════
📦 ENTITY STATUS (Use cached data when valid - DON'T re-fetch!)
═══════════════════════════════════════════════════════════════

{entity_display if entity_display else "No entities resolved yet."}

═══════════════════════════════════════════════════════════════
📋 EXECUTION PLAN (Phase 6 - Plan-based execution)
═══════════════════════════════════════════════════════════════

{plan_display}

═══════════════════════════════════════════════════════════════
📊 EXECUTION HISTORY (Iteration {execution_context.iteration})
═══════════════════════════════════════════════════════════════

{observations_summary if observations_summary else "No actions taken yet."}
"""
        
        return full_prompt

    def _format_observations(self, execution_context: 'ExecutionContext') -> str:
        """Format execution observations for the prompt."""
        if not execution_context.observations:
            return ""

        lines = []
        for i, obs in enumerate(execution_context.observations, 1):
            lines.append(f"\n--- Observation {i}: {obs['type'].upper()} - {obs['name']} ---")
            lines.append(f"Result: {json.dumps(obs['result'], indent=2, default=str)}")

            # Add success/failure indicator
            if obs['type'] == 'tool':
                if obs['result'].get('success'):
                    lines.append("✅ SUCCESS")
                elif 'error' in obs['result']:
                    lines.append(f"❌ FAILED: {obs['result'].get('error')}")

        return "\n".join(lines)

    def _format_tools_for_prompt(self) -> str:
        """Format available tools with explicit parameter requirements."""
        if not self._tool_schemas:
            return "No tools available."

        lines = [
            "⚠️ IMPORTANT: Use EXACT parameter names shown below.",
            "Do NOT invent parameter names or use synonyms.",
            ""
        ]
        
        for tool in self._tool_schemas:
            lines.append(f"═══ {tool['name']} ═══")
            lines.append(f"Description: {tool['description']}")
            
            params = tool['input_schema'].get('properties', {})
            required = set(tool['input_schema'].get('required', []))
            
            if params:
                lines.append("Parameters (use these EXACT names):")
                for name, schema in params.items():
                    req_str = "✓ REQUIRED" if name in required else "○ optional"
                    desc = schema.get('description', 'No description')
                    lines.append(f"  • \"{name}\" [{req_str}]: {desc}")
            else:
                lines.append("Parameters: none")
            lines.append("")

        return "\n".join(lines)
    
    def _build_static_caches(self):
        """Build cached static content that doesn't change per-iteration."""
        
        # Cache tools description
        self._cached_tools_description = self._format_tools_for_prompt()
        
        # Cache ReAct instructions
        self._cached_react_instructions = """
═══════════════════════════════════════════════════════════════
AGENTIC REASONING PROTOCOL (ReAct Pattern)
═══════════════════════════════════════════════════════════════
You are operating in a THINK → ACT → OBSERVE loop. Your job is to:

1. ANALYZE what has been done (review observations)
2. EVALUATE if the user's request is fulfilled
3. DECIDE your next action

CRITICAL RULES:
1. NEVER claim success before verifying tool results
2. READ TOOL RESULTS CAREFULLY - check for success/error
3. COMPLETE THE FULL TASK before responding
4. VERIFY BEFORE RESPONDING - Did all required actions succeed?
"""
        
        # Cache result type guide
        self._cached_result_type_guide = self._get_result_type_guide()
        
        # Cache decision guide
        self._cached_decision_guide = self._get_decision_guide()
        
        logger.debug(f"[{self.agent_name}] Built static caches: tools={len(self._cached_tools_description)} chars")
    
    def _check_derived_entity_cache(
        self,
        session_id: str,
        tool_name: str,
        tool_input: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Check if we can use a cached derived entity instead of calling the tool.
        
        Returns:
            Cached result if available and valid, None if tool should be called
        """
        # Tools that can use cached entities
        cacheable_tools = {
            "find_doctor_by_name": "doctor_uuid",
            "list_doctors": "doctors_list",
        }
        
        if tool_name not in cacheable_tools:
            return None
        
        entity_key = cacheable_tools[tool_name]
        cached_value = self.state_manager.get_valid_derived_entity(session_id, entity_key)
        
        if cached_value is None:
            return None
        
        # Build result from cached entity
        if tool_name == "find_doctor_by_name":
            # Also need doctor_info for full result
            doctor_info = self.state_manager.get_valid_derived_entity(session_id, "doctor_info")
            if doctor_info:
                return {
                    "success": True,
                    "result_type": "success",
                    "doctor": doctor_info,
                    "from_cache": True
                }
        
        elif tool_name == "list_doctors":
            # Don't use cache if it's an empty list - allow tool to be called to check for new doctors
            if isinstance(cached_value, list) and len(cached_value) == 0:
                logger.info(f"[{self.agent_name}] Cached doctors list is empty, calling tool to refresh")
                return None  # Force tool call instead of using empty cache
            
            return {
                "success": True,
                "result_type": "partial",
                "doctors": cached_value,
                "count": len(cached_value),
                "from_cache": True
            }
        
        return None

    def _store_derived_entities(
        self,
        session_id: str,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_result: Dict[str, Any]
    ):
        """
        Store derived entities from a tool result.
        
        This is called after every successful tool execution.
        """
        # Validation
        if not self.agent_name:
            logger.error(f"Agent name not set when storing derived entities for tool {tool_name}")
            return

        try:
            stored = self.state_manager.store_tool_result(
                session_id=session_id,
                tool_name=tool_name,
                tool_params=tool_input,
                tool_result=tool_result,
                agent_name=self.agent_name
            )
            
            if stored:
                logger.info(
                    f"Agent {self.agent_name} stored {len(stored)} derived entities "
                    f"from {tool_name}: {stored}"
                )
        
        except Exception as e:
            # Don't fail the tool call if entity storage fails
            logger.error(f"Error storing derived entities: {e}", exc_info=True)

    def _resolve_tool_input_templates(
        self,
        session_id: str,
        tool_input: Dict[str, Any],
        exec_context: ExecutionContext
    ) -> Dict[str, Any]:
        """
        Resolve template placeholders in tool_input using:
        1. Previous task results from observations
        2. Derived entities from state_manager
        3. Plan's entities
        
        Templates like {{task_1.doctor_id}} or {{doctor_uuid}} are resolved to actual values.
        """
        import re
        
        resolved_input = {}
        
        # Build a resolution context from all available sources
        resolution_context = {}
        
        # 1. From plan's entities
        if exec_context.plan and exec_context.plan.entities:
            resolution_context.update(exec_context.plan.entities)
        
        # 2. From observations (tool results)
        for obs in exec_context.observations:
            if obs.type == "tool" and obs.is_success():
                result = obs.result
                # Common entity patterns to extract
                if 'doctor_id' in result:
                    resolution_context['doctor_id'] = result['doctor_id']
                    resolution_context['doctor_uuid'] = result['doctor_id']
                if 'doctors' in result and result['doctors']:
                    resolution_context['doctors_list'] = result['doctors']
                if 'appointment_id' in result:
                    resolution_context['appointment_id'] = result['appointment_id']
        
        # 3. From derived entities in state_manager
        try:
            entity_state = self.state_manager.get_entity_state(session_id)
            # Extract derived entities from EntityState
            if hasattr(entity_state, 'derived') and entity_state.derived:
                # Get all valid derived entities
                if hasattr(entity_state.derived, 'entities'):
                    for key, entity in entity_state.derived.entities.items():
                        if hasattr(entity, 'value') and not entity.is_stale():
                            resolution_context[key] = entity.value
        except Exception as e:
            logger.warning(f"Could not load entity state for template resolution: {e}")
        
        # 4. From initial context entities
        context = self._context.get(session_id, {})
        entities = context.get('entities', {})
        resolution_context.update(entities)
        
        # Now resolve each value in tool_input
        for key, value in tool_input.items():
            if isinstance(value, str):
                resolved_value = value
                
                # Find all template patterns like {{something}}
                template_pattern = r'\{\{([^}]+)\}\}'
                matches = re.findall(template_pattern, value)
                
                for match in matches:
                    template_key = match.strip()
                    resolved = None
                    
                    # Try direct lookup
                    if template_key in resolution_context:
                        resolved = resolution_context[template_key]
                    
                    # Try task_N.field pattern (e.g., task_1.doctor_id)
                    elif '.' in template_key:
                        parts = template_key.split('.', 1)
                        task_ref = parts[0]  # e.g., "task_1"
                        field_ref = parts[1]  # e.g., "doctor_id_for_mohammed_atef"
                        
                        # Look in completed task results
                        if exec_context.plan:
                            task = exec_context.plan.get_task(task_ref)
                            if task and task.result:
                                # Try to find the field in the result
                                result = task.result
                                
                                # Check if field_ref contains a doctor name hint
                                if 'doctor' in field_ref.lower() and 'doctors' in result:
                                    # Extract doctor name from field reference
                                    name_match = re.search(r'for_(.+)', field_ref)
                                    if name_match:
                                        doctor_name_search = name_match.group(1).replace('_', ' ').lower()
                                        for doctor in result.get('doctors', []):
                                            if doctor_name_search in doctor.get('name', '').lower():
                                                resolved = doctor.get('id')
                                                break
                                
                                # Direct field lookup in result
                                if not resolved and field_ref in result:
                                    resolved = result[field_ref]
                    
                    # Also try common derived entity keys
                    if not resolved and 'doctor' in template_key.lower():
                        resolved = resolution_context.get('doctor_uuid') or resolution_context.get('doctor_id')
                    
                    # Replace the template if resolved
                    if resolved is not None:
                        template_str = '{{' + match + '}}'
                        resolved_value = resolved_value.replace(template_str, str(resolved))
                        logger.info(f"🔗 [{self.agent_name}] Resolved template {template_str} → {resolved}")
                    else:
                        logger.warning(f"🔗 [{self.agent_name}] Could not resolve template: {{{{{match}}}}}")
                
                resolved_input[key] = resolved_value
            else:
                # Non-string values pass through unchanged
                resolved_input[key] = value
        
        return resolved_input

    def _build_thinking_messages(
        self,
        session_id: str,
        execution_context: 'ExecutionContext'
    ) -> List[Dict[str, str]]:
        """Build messages for the thinking LLM call."""
        messages = []

        # Include recent conversation history (last 6 turns)
        history = self.conversation_history.get(session_id, [])
        recent_history = history[-6:] if len(history) > 6 else history

        for msg in recent_history:
            messages.append(msg)

        # Add thinking prompt as final user message
        thinking_prompt = f"""

Based on the execution history in the system prompt, decide what to do next.

USER'S REQUEST: {execution_context.user_request}

CONTEXT: {json.dumps(self._context.get(session_id, {}), default=str)}

Respond with your analysis and decision in the JSON format specified.

"""

        messages.append({"role": "user", "content": thinking_prompt})

        return messages

    def _parse_thinking_response(
        self,
        response: str,
        execution_context: 'ExecutionContext'
    ) -> ThinkingResult:
        """Parse LLM thinking response into structured result."""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                # Fallback: try to interpret the response
                logger.warning("No JSON found in thinking response, attempting interpretation")
                return self._interpret_unstructured_response(response, execution_context)

            data = json.loads(json_match.group())

            decision_str = data.get('decision', 'RESPOND').upper()
            decision = AgentDecision[decision_str] if decision_str in AgentDecision.__members__ else AgentDecision.RESPOND

            # Extract task status
            task_status = data.get('task_status', {})
            is_complete = task_status.get('is_complete', False)

            # Extract tool call info if present
            tool_call = data.get('tool_call', {})
            tool_name = tool_call.get('name') if decision == AgentDecision.CALL_TOOL else None
            tool_input = tool_call.get('input', {}) if decision == AgentDecision.CALL_TOOL else None

            return ThinkingResult(
                decision=decision,
                reasoning=data.get('reasoning', data.get('analysis', '')),
                tool_name=tool_name,
                tool_input=tool_input,
                response_text=data.get('response'),
                is_task_complete=is_complete,
                validation_notes=json.dumps(task_status.get('completed', []), default=str)
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse thinking response as JSON: {e}")
            return self._interpret_unstructured_response(response, execution_context)

    def _interpret_unstructured_response(
        self,
        response: str,
        execution_context: 'ExecutionContext'
    ) -> ThinkingResult:
        """Fallback interpretation of unstructured response."""
        response_lower = response.lower()

        # Check for tool call indicators
        for tool in self._tool_schemas:
            if tool['name'] in response_lower:
                return ThinkingResult(
                    decision=AgentDecision.CALL_TOOL,
                    reasoning=f"Detected tool reference: {tool['name']}",
                    tool_name=tool['name'],
                    tool_input={}
                )

        # Check for completion indicators
        completion_words = ['complete', 'done', 'finished', 'confirmed', 'booked', 'success']
        if any(word in response_lower for word in completion_words):
            return ThinkingResult(
                decision=AgentDecision.RESPOND,
                reasoning="Response indicates completion",
                response_text=response,
                is_task_complete=True
            )

        # Default: respond with what we have
        return ThinkingResult(
            decision=AgentDecision.RESPOND,
            reasoning="Unable to parse structured response, generating response",
            response_text=response,
            is_task_complete=False
        )

    # ==================== TASK COMPLETION VERIFICATION ====================

    async def _verify_task_completion(
        self,
        session_id: str,
        execution_context: 'ExecutionContext'
    ) -> 'CompletionCheckResult':
        """
        Verify that the user's task has been completed.

        This is a safety check before generating a final response.

        """
        system_prompt = f"""You are a task completion validator.

USER'S REQUEST: {execution_context.user_request}

EXECUTION HISTORY:

{self._format_observations(execution_context)}

TASK: Verify if the user's request has been fulfilled.

Check:

1. What did the user ask for?

2. What actions were taken?

3. Did those actions succeed (check for "success": true in results)?

4. Is anything missing?

Respond with JSON:

{{
    "is_complete": true/false,
    "completed_items": ["List of successfully completed actions"],
    "missing_items": ["List of unfulfilled requirements"],
    "recommended_response": "Suggested response to user if complete"
}}

"""

        # Get observability logger
        obs_logger = get_observability_logger(session_id) if settings.enable_observability else None
        llm_start_time = time.time()
        
        # Get hierarchical LLM config for _verify_task_completion function
        llm_config = self._get_llm_config_for_function("_verify_task_completion")
        verification_temperature = llm_config.temperature
        llm_client = self._get_llm_client_for_function("_verify_task_completion")

        try:
            # Try to get token usage if available
            if hasattr(llm_client, 'create_message_with_usage'):
                response, tokens = llm_client.create_message_with_usage(
                    system=system_prompt,
                    messages=[{"role": "user", "content": "Verify task completion."}],
                    temperature=verification_temperature,
                    max_tokens=llm_config.max_tokens
                )
            else:
                response = llm_client.create_message(
                    system=system_prompt,
                    messages=[{"role": "user", "content": "Verify task completion."}],
                    temperature=verification_temperature,
                    max_tokens=llm_config.max_tokens
                )
                tokens = TokenUsage()

            llm_duration_seconds = time.time() - llm_start_time

            # Record LLM call
            if obs_logger:
                obs_logger.record_llm_call(
                    component=f"agent.{self.agent_name}.verify_completion",
                    provider=llm_config.provider,
                    model=llm_config.model,
                    tokens=tokens,
                    duration_seconds=llm_duration_seconds,
                    system_prompt_length=len(system_prompt),
                    messages_count=1,
                    temperature=verification_temperature,
                    max_tokens=llm_config.max_tokens,
                    function_name="_verify_task_completion",
                    error=None
                )

            # Parse response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return CompletionCheckResult(
                    is_complete=data.get('is_complete', False),
                    completed_items=data.get('completed_items', []),
                    missing_items=data.get('missing_items', []),
                    response=data.get('recommended_response', '')
                )

        except Exception as e:
            llm_duration_seconds = time.time() - llm_start_time
            
            # Record failed LLM call
            if obs_logger:
                obs_logger.record_llm_call(
                    component=f"agent.{self.agent_name}.verify_completion",
                    provider=llm_config.provider,
                    model=llm_config.model,
                    tokens=TokenUsage(),
                    duration_seconds=llm_duration_seconds,
                    system_prompt_length=len(system_prompt),
                    messages_count=1,
                    temperature=verification_temperature,
                    max_tokens=llm_config.max_tokens,
                    function_name="_verify_task_completion",
                    error=str(e)
                )
            
            logger.error(f"Error in completion verification: {e}")

        # Default: assume incomplete
        return CompletionCheckResult(
            is_complete=False,
            completed_items=[],
            missing_items=["Unable to verify completion"],
            response=""
        )

    # ==================== FINAL RESPONSE GENERATION ====================

    async def _generate_final_response(
        self,
        session_id: str,
        execution_context: 'ExecutionContext'
    ) -> str:
        """
        [DEPRECATED] Generate the final response to the user based on execution results.

        This method is only used by process_message_legacy() for backward compatibility.
        New code should use _generate_focused_response() instead, which includes
        what_user_means and tone from the reasoning engine.

        This is called ONLY after all tools have been executed.

        """
        system_prompt = f"""{self._get_system_prompt(session_id)}

═══════════════════════════════════════════════════════════════
FINAL RESPONSE GENERATION
═══════════════════════════════════════════════════════════════

You have completed executing tools. Now generate a response to the user.

EXECUTION RESULTS:

{self._format_observations(execution_context)}

RULES:

1. ONLY report what actually happened (check tool results)

2. If a tool succeeded, confirm what was done

3. If a tool failed, explain the issue and suggest alternatives

4. Be concise and friendly

5. Do NOT include raw JSON or technical details

6. Use natural language

FORBIDDEN:

- Including any  technical outputs, tools, JSON, UUIDs, or internal data. Only provide user-friendly text.

- Claiming actions succeeded if tool returned error

- Including tool result JSON in response

- Saying "I'll do X" - you already did it (or didn't)

"""

        # Get observability logger
        obs_logger = get_observability_logger(session_id) if settings.enable_observability else None
        llm_start_time = time.time()
        
        # Get hierarchical LLM config for _generate_response function
        llm_config = self._get_llm_config_for_function("_generate_response")
        response_temperature = llm_config.temperature
        llm_client = self._get_llm_client_for_function("_generate_response")

        try:
            user_message = f"Generate a response for: {execution_context.user_request}"
            
            # Try to get token usage if available
            if hasattr(llm_client, 'create_message_with_usage'):
                response, tokens = llm_client.create_message_with_usage(
                    system=system_prompt,
                    messages=[{
                        "role": "user",
                        "content": user_message
                    }],
                    temperature=response_temperature,
                    max_tokens=llm_config.max_tokens
                )
            else:
                response = llm_client.create_message(
                    system=system_prompt,
                    messages=[{
                        "role": "user",
                        "content": user_message
                    }],
                    temperature=response_temperature,
                    max_tokens=llm_config.max_tokens
                )
                tokens = TokenUsage()

            llm_duration_seconds = time.time() - llm_start_time

            # Record LLM call
            if obs_logger:
                obs_logger.record_llm_call(
                    component=f"agent.{self.agent_name}.generate_response",
                    provider=llm_config.provider,
                    model=llm_config.model,
                    tokens=tokens,
                    duration_seconds=llm_duration_seconds,
                    system_prompt_length=len(system_prompt),
                    messages_count=1,
                    temperature=response_temperature,
                    max_tokens=llm_config.max_tokens,
                    function_name="_generate_response",
                    error=None
                )

            return response

        except Exception as e:
            llm_duration_seconds = time.time() - llm_start_time
            
            # Record failed LLM call
            if obs_logger:
                obs_logger.record_llm_call(
                    component=f"agent.{self.agent_name}.generate_response",
                    provider=llm_config.provider,
                    model=llm_config.model,
                    tokens=TokenUsage(),
                    duration_seconds=llm_duration_seconds,
                    system_prompt_length=len(system_prompt),
                    messages_count=1,
                    temperature=response_temperature,
                    max_tokens=llm_config.max_tokens,
                    function_name="_generate_response",
                    error=str(e)
                )
            
            logger.error(f"Error generating final response: {e}")
            return "I apologize, but I encountered an issue processing your request. Please try again."

    # ==================== TOOL EXECUTION ====================

    async def _execute_tool(
        self,
        session_id: str,
        tool_name: str,
        tool_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a tool and return the result."""
        tool_start_time = time.time()
        obs_logger = get_observability_logger(session_id) if settings.enable_observability else None

        if tool_name not in self._tools:
            logger.error(f"Unknown tool: {tool_name}")
            error_result = {"error": f"Unknown tool: {tool_name}", "success": False}
            return error_result

        try:
            tool_function = self._tools[tool_name]
            tool_input['session_id'] = session_id

            # Execute tool (handle both sync and async)
            import asyncio
            if asyncio.iscoroutinefunction(tool_function):
                result = await tool_function(**tool_input)
            else:
                result = tool_function(**tool_input)

            logger.info(f"Tool '{tool_name}' executed successfully")
            tool_duration_seconds = time.time() - tool_start_time
            result_dict = result if isinstance(result, dict) else {"result": result}

            # Log to observability
            if obs_logger:
                obs_logger.record_tool_execution(
                    tool_name=tool_name,
                    inputs=tool_input,
                    outputs=result_dict,
                    duration_seconds=tool_duration_seconds,
                    success=True
                )

            # Log to execution log
            if session_id in self._execution_log:
                tool_execution = ToolExecution(
                    tool_name=tool_name,
                    inputs=tool_input,
                    outputs=result_dict,
                    timestamp=datetime.utcnow()
                )
                self._execution_log[session_id].tools_used.append(tool_execution)

            return result_dict

        except Exception as e:
            logger.error(f"Error executing tool '{tool_name}': {e}", exc_info=True)
            error_result = {"error": str(e), "success": False}

            if obs_logger:
                obs_logger.record_tool_execution(
                    tool_name=tool_name,
                    inputs=tool_input,
                    outputs=error_result,
                    duration_seconds=time.time() - tool_start_time,
                    success=False,
                    error=str(e)
                )

            if session_id in self._execution_log:
                tool_execution = ToolExecution(
                    tool_name=tool_name,
                    inputs=tool_input,
                    outputs=error_result,
                    timestamp=datetime.utcnow()
                )
                self._execution_log[session_id].tools_used.append(tool_execution)

            return error_result

    # ==================== UTILITY METHODS ====================

    def _get_error_response(self, error: str) -> str:
        """Generate user-friendly error response."""
        return (
            "I'm sorry, I encountered an error while processing your request. "
            "Please try again or contact support if the issue persists."
        )

    def _get_context_note(self, session_id: str) -> str:
        """Generate a brief context note for the system prompt."""
        context = self._context.get(session_id, {})
        if not context:
            return ""

        parts = []

        if "what_user_means" in context:
            parts.append(f"What user means: {context['what_user_means']}")
        elif "user_intent" in context:
            parts.append(f"User intent: {context['user_intent']}")

        if "action" in context:
            parts.append(f"Suggested action: {context['action']}")

        if "prior_context" in context:
            parts.append(f"Context: {context['prior_context']}")

        # Language context
        current_language = context.get("current_language")
        if current_language and current_language != "en":
            dialect = context.get("current_dialect", "")
            lang_display = f"{current_language}-{dialect}" if dialect else current_language
            parts.append(f"User's language: {lang_display}")

        if not parts:
            return ""

        return "\n[CONVERSATION CONTEXT]\n" + "\n".join(parts) + "\n"

    def clear_history(self, session_id: str):
        """Clear conversation history for a session."""
        if session_id in self.conversation_history:
            del self.conversation_history[session_id]
            logger.info(f"Cleared history for {self.agent_name}, session: {session_id}")

    def get_history_length(self, session_id: str) -> int:
        """Get conversation history length."""
        return len(self.conversation_history.get(session_id, []))

    # ==================== NEW AGENTIC HELPER METHODS ====================

    def _get_agent_instructions(self) -> str:
        """
        Get agent-specific instructions for planning and thinking.

        Override in specialized agents to add custom behavioral rules like:
        - "Always confirm details before executing"
        - "Never provide medical advice"
        - "Ask for missing information before proceeding"

        Returns:
            String with agent-specific instructions (empty for default behavior)
        """
        return ""  # Default: no special instructions

    def _get_thinking_system_prompt(self, exec_context: Optional['ExecutionContext'] = None) -> str:
        """
        Get the AGENT-SPECIFIC system prompt (Layer 2 - NOT cached).

        This returns the agent identity, instructions, and tools.
        The universal guides are in Layer 1 (cached).
        """
        # Get tools description (scoped to plan if available)
        logger.info(f"📝 [{self.agent_name}] Step 1: Getting tools description...")
        
        # Determine which tools to include
        filter_tools = None  # Default: all tools
        scoping_enabled = False
        
        if exec_context and exec_context.plan and exec_context.plan.should_use_scoped_tools():
            filter_tools = exec_context.plan.required_tools
            scoping_enabled = True
            logger.info(
                f"📝 [{self.agent_name}]   → Using scoped tools: {filter_tools} "
                f"(FULL descriptions per tool)"
            )
        else:
            logger.info(f"📝 [{self.agent_name}]   → Using full tool catalog")
            if exec_context and exec_context.plan:
                if exec_context.plan.all_tools_enabled:
                    logger.info(
                        f"📝 [{self.agent_name}]      Reason: Expansion requested "
                        f"at iteration {exec_context.plan.tool_expansion_requested_at}"
                    )
                elif not exec_context.plan.required_tools:
                    logger.info(
                        f"📝 [{self.agent_name}]      Reason: Plan has no required tools"
                    )
            else:
                logger.info(f"📝 [{self.agent_name}]      Reason: No plan exists")
        
        # Get tool descriptions (filtered or full, but ALWAYS with complete details per tool)
        tools_desc = self._get_tools_description(filter_tools=filter_tools)
        
        logger.info(
            f"📝 [{self.agent_name}]   ✅ Tools description length: {len(tools_desc)} chars"
        )
        
        # Calculate and log token savings
        if scoping_enabled:
            all_tools_desc = self._get_tools_description()
            saved_chars = len(all_tools_desc) - len(tools_desc)
            estimated_tokens_saved = saved_chars // 4
            logger.info(
                f"📝 [{self.agent_name}]   💰 Estimated tokens saved: ~{estimated_tokens_saved}"
            )
        
        logger.debug(f"📝 [{self.agent_name}]   → Tools description:\n{tools_desc}")
        
        # Build scoping warning text if scoping is enabled
        scoping_warning = ""
        if scoping_enabled:
            scoping_warning = """
[!] TOOL SCOPING ACTIVE
================================================================================

Only tools required by your plan are shown above. Each tool has its FULL
description with all parameters and details.

If you discover you need a tool NOT listed (e.g., user changed their mind,
tool failed and you need alternative):

  1. Set "request_all_tools": true in your JSON response
  2. Set "tool_expansion_reason": "Brief explanation why"
  3. You will be called again with the complete tool catalog

"""
        
        tools_header_suffix = " (SCOPED TO PLAN)" if scoping_enabled else ""
        
        return f"""
═══════════════════════════════════════════════════════════════════════════════
YOU ARE THE THINKING MODULE
═══════════════════════════════════════════════════════════════════════════════

You are the thinking module for the {self.agent_name} agent.

Your job is to:
1. Analyze the current situation based on observations
2. Decide the best next action to take
3. Know when the task is complete

═══════════════════════════════════════════════════════════════════════════════
AGENT-SPECIFIC INSTRUCTIONS
═══════════════════════════════════════════════════════════════════════════════

{self._get_agent_instructions() or 'No special instructions for this agent.'}

═══════════════════════════════════════════════════════════════════════════════
🛠️  AVAILABLE TOOLS{tools_header_suffix}
═══════════════════════════════════════════════════════════════════════════════

{tools_desc}

{scoping_warning}═══════════════════════════════════════════════════════════════════════════════
CRITICAL: Always respond with valid JSON in the exact format specified.
═══════════════════════════════════════════════════════════════════════════════
"""

    def _get_result_type_guide(self) -> str:
        """Guide for understanding tool result types."""
        return """
═══════════════════════════════════════════════════════════════════════════════
UNDERSTANDING TOOL RESULTS
═══════════════════════════════════════════════════════════════════════════════

After each tool call, the result has a result_type that tells you what to do:

┌─────────────────┬────────────────────────────────────────────────────────────┐
│ result_type     │ What it means & What to do                                 │
├─────────────────┼────────────────────────────────────────────────────────────┤
│ SUCCESS         │ ✅ Goal achieved! Mark criterion complete.                  │
│                 │    Look for: success=true, appointment_id, confirmation    │
│                 │    Action: Mark relevant criterion COMPLETE, continue      │
├─────────────────┼────────────────────────────────────────────────────────────┤
│ PARTIAL         │ ⏳ Progress made, more steps needed.                        │
│                 │    Look for: data returned but more actions needed         │
│                 │    Action: Continue to next logical step                   │
├─────────────────┼────────────────────────────────────────────────────────────┤
│ USER_INPUT      │ 🔄 STOP! Cannot proceed without user decision.             │
│                 │    Look for: alternatives array, available=false           │
│                 │    Action: RESPOND_WITH_OPTIONS - present choices to user  │
│                 │    DO NOT keep trying tools - user must choose!            │
├─────────────────┼────────────────────────────────────────────────────────────┤
│ RECOVERABLE     │ 🔧 Try a different approach.                               │
│                 │    Look for: recovery_action field                         │
│                 │    Action: Try suggested recovery action or alternative    │
├─────────────────┼────────────────────────────────────────────────────────────┤
│ FATAL           │ ❌ Cannot complete this request.                           │
│                 │    Look for: error with no recovery path                   │
│                 │    Action: RESPOND_IMPOSSIBLE - explain why, suggest alt   │
├─────────────────┼────────────────────────────────────────────────────────────┤
│ SYSTEM_ERROR    │ 🚫 Infrastructure failure.                                 │
│                 │    Look for: database error, timeout, connection issue     │
│                 │    Action: RETRY with different tool, then RESPOND_IMPOSSIBLE│
└─────────────────┴────────────────────────────────────────────────────────────┘

CRITICAL: When result_type is USER_INPUT:
- This is NOT a failure!
- The tool worked correctly
- But user must make a choice before proceeding
- You MUST stop and present options
- DO NOT try other tools hoping for different result
"""

    def _get_decision_guide(self) -> str:
        """Guide for making decisions."""
        return """
═══════════════════════════════════════════════════════════════════════════════
DECISION GUIDE
═══════════════════════════════════════════════════════════════════════════════

Choose your decision based on the situation:

CALL_TOOL:
- You need information or want to perform an action
- You have all required parameters
- No criteria are blocked waiting for user input

RESPOND (with is_task_complete=true):
- ALL success criteria are COMPLETE
- You have confirmation/evidence for each criterion
- Time to give user the good news!

RESPOND_WITH_OPTIONS:
- A tool returned result_type=USER_INPUT
- You have alternatives to present
- User must choose before you can continue

RESPOND_IMPOSSIBLE:
- A tool returned result_type=FATAL
- The request cannot be fulfilled
- Explain why and suggest alternatives if any

CLARIFY:
- You don't have enough information
- Required parameters are missing
- Ask a specific question
- MUST specify clarification_question
- MUST specify awaiting_info (what you need)
- MUST specify entities (what you already have)

COLLECT_INFORMATION:
- You need to exit the agentic loop to collect information from the user
- Use this when you need to gather data before continuing execution
- Response will pass through focused response generation for natural output
- MUST specify awaiting_info (what you need)
- MUST specify entities (what you already have)

REQUEST_CONFIRMATION:
- Use BEFORE executing critical actions: book_appointment, cancel_appointment, reschedule_appointment
- MUST specify confirmation_summary with full details (action, details, tool_name, tool_input)
- Wait for user to confirm before calling the tool

RETRY:
- A tool returned result_type=SYSTEM_ERROR
- Haven't exceeded retry limit
- Tool returned recovery_action

═══════════════════════════════════════════════════════════════════════════════
"""

    def _build_thinking_prompt(
        self,
        message: str,
        context: Dict[str, Any],
        exec_context: ExecutionContext
    ) -> str:
        """
        Build the DYNAMIC thinking prompt (Layer 3 - NOT cached).

        This contains only information that changes per request:
        - User message
        - Intent from reasoning engine
        - Resolved entities (filtered for relevance)
        - Observations from execution
        - Plan status
        """
        logger.info(f"📝 [{self.agent_name}] Building dynamic thinking prompt...")

        # Extract context fields
        user_intent = context.get('user_intent') or context.get('what_user_means', 'Not specified')
        entities = context.get('entities', {})  # Internal use only (NOT shown in prompt)
        llm_entities = context.get('llm_entities', {})  # NEW: ISOLATED LLM-only entities
        constraints = context.get('constraints', [])
        routing_action = context.get('routing_action', context.get('action', 'Not specified'))

        # NOTE: llm_entities ONLY contains:
        #   1. Entities YOU explicitly updated via entities_to_update
        #   2. patient_id (auto-injected by system)
        # To get other data (slots, doctor info, etc.), use tools!


        # Build continuation section if applicable (unified reasoning v3)
        continuation_section = ""
        if context.get('is_continuation', False):
            continuation_type = context.get('continuation_type') or context.get('situation_type')
            selected_option = context.get('selected_option')

            # Extract information_collection if present
            continuation_ctx = context.get('continuation_context', {})
            information_collection = continuation_ctx.get('information_collection', {})

            # Build information collection section if applicable
            info_section = ""
            if information_collection and information_collection.get('information_needed'):
                collected = information_collection.get('collected_information', [])
                info_section = f"""
Collected Information:
{collected}
"""


        # Get plan status (compact)
        plan_status = "No plan"
        if exec_context.plan:
            plan = exec_context.plan
            completed = len(plan.get_completed_task_ids())
            total = len(plan.tasks)
            plan_status = f"{plan.objective} [{completed}/{total} complete]"

        # Build compact observations summary
        observations = exec_context.get_observations_summary() or "(none yet)"

        # Format LLM entities (max 9 shown)
        if llm_entities:
            entities_json = json.dumps(dict(list(llm_entities.items())[:9]), indent=2, default=str)
            if len(llm_entities) > 9:
                entities_json += f"\n  ... and {len(llm_entities) - 9} more"
            formatted_entities = entities_json
        else:
            formatted_entities = "  (empty - start fresh or patient_id will be injected)"

        # Assemble complete dynamic prompt with JSON schema
        prompt = f"""
═══════════════════════════════════════════════════════════════════════════════
CURRENT SITUATION
═══════════════════════════════════════════════════════════════════════════════

User's Message: {message}

Identidfied Intent: {routing_action} | {user_intent}


COLLECTED ENTITIES:
{formatted_entities}


═══════════════════════════════════════════════════════════════════════════════
EXECUTION STATUS
═══════════════════════════════════════════════════════════════════════════════

PREVIOUS TOOL RESULTS (Iteration {exec_context.iteration}):
{observations}

═══════════════════════════════════════════════════════════════════════════════
YOUR TASK
═══════════════════════════════════════════════════════════════════════════════

Analyze the situation and decide your next action.

Follow the OUTPUT FORMAT and DECISION GUIDE from the system prompt.
"""

        logger.info(f"📝 [{self.agent_name}] Dynamic prompt: {len(prompt)} chars")
        return prompt


    def _filter_relevant_entities(
        self,
        entities: Dict[str, Any],
        context: Dict[str, Any],
        max_entities: int = 10
    ) -> Dict[str, Any]:
        """
        Filter resolved entities to only include relevant ones.
        
        Keeps:
        - Core identifiers (patient_id, session_id)
        - Entities mentioned in current intent
        - Recently used entities
        """
        if not entities or len(entities) <= max_entities:
            return entities
        
        # Always keep core identifiers
        core_keys = {'patient_id', 'session_id', 'user_id'}
        
        # Get intent to check relevance
        intent = context.get('user_intent', '') + ' ' + context.get('what_user_means', '')
        intent_lower = intent.lower()
        
        # Score entities
        scored = {}
        for key, value in entities.items():
            if key in core_keys:
                scored[key] = 100  # Always include
            elif key.lower() in intent_lower or str(value).lower() in intent_lower:
                scored[key] = 50  # Mentioned in intent
            else:
                scored[key] = 1  # Low priority
        
        # Keep top N
        sorted_keys = sorted(scored.keys(), key=lambda k: scored[k], reverse=True)
        return {k: entities[k] for k in sorted_keys[:max_entities]}

    def _get_result_type_guide(self) -> str:
        """Guide for understanding tool result types."""
        return """
═══════════════════════════════════════════════════════════════════════════════
UNDERSTANDING TOOL RESULTS
═══════════════════════════════════════════════════════════════════════════════

After each tool call, the result has a result_type that tells you what to do:

┌─────────────────┬────────────────────────────────────────────────────────────┐
│ result_type     │ What it means & What to do                                 │
├─────────────────┼────────────────────────────────────────────────────────────┤
│ SUCCESS         │ ✅ Goal achieved! Mark criterion complete.                  │
│                 │    Look for: success=true, appointment_id, confirmation    │
│                 │    Action: Mark relevant criterion COMPLETE, continue      │
├─────────────────┼────────────────────────────────────────────────────────────┤
│ PARTIAL         │ ⏳ Progress made, more steps needed.                        │
│                 │    Look for: data returned but more actions needed         │
│                 │    Action: Continue to next logical step                   │
├─────────────────┼────────────────────────────────────────────────────────────┤
│ USER_INPUT      │ 🔄 STOP! Cannot proceed without user decision.             │
│                 │    Look for: alternatives array, available=false           │
│                 │    Action: RESPOND_WITH_OPTIONS - present choices to user  │
│                 │    DO NOT keep trying tools - user must choose!            │
├─────────────────┼────────────────────────────────────────────────────────────┤
│ RECOVERABLE     │ 🔧 Try a different approach.                               │
│                 │    Look for: recovery_action field                         │
│                 │    Action: Try suggested recovery action or alternative    │
├─────────────────┼────────────────────────────────────────────────────────────┤
│ FATAL           │ ❌ Cannot complete this request.                           │
│                 │    Look for: error with no recovery path                   │
│                 │    Action: RESPOND_IMPOSSIBLE - explain why, suggest alt   │
├─────────────────┼────────────────────────────────────────────────────────────┤
│ SYSTEM_ERROR    │ 🚫 Infrastructure failure.                                 │
│                 │    Look for: database error, timeout, connection issue     │
│                 │    Action: RETRY with different tool, then RESPOND_IMPOSSIBLE│
└─────────────────┴────────────────────────────────────────────────────────────┘

CRITICAL: When result_type is USER_INPUT:
- This is NOT a failure!
- The tool worked correctly
- But user must make a choice before proceeding
- You MUST stop and present options
- DO NOT try other tools hoping for different result
"""

    def _get_decision_guide(self) -> str:
        """Guide for making decisions."""
        return """
═══════════════════════════════════════════════════════════════════════════════
DECISION GUIDE
═══════════════════════════════════════════════════════════════════════════════

Choose your decision based on the situation:

CALL_TOOL:
- You need information or want to perform an action
- You have all required parameters
- No criteria are blocked waiting for user input

RESPOND_COMPLETE (with is_task_complete=true):
- ALL success criteria are COMPLETE
- You have confirmation/evidence for each criterion
- Time to give user the good news!

RESPOND_WITH_OPTIONS:
- A tool returned result_type=USER_INPUT
- You have alternatives to present
- User must choose before you can continue

RESPOND_IMPOSSIBLE:
- A tool returned result_type=FATAL
- The request cannot be fulfilled
- Explain why and suggest alternatives if any

CLARIFY:
- You don't have enough information
- Required parameters are missing
- Ask a specific question
- MUST specify clarification_question
- MUST specify awaiting_info (what you need)
- MUST specify entities (what you already have)

COLLECT_INFORMATION:
- You need to exit the agentic loop to collect information from the user
- Use this when you need to gather data before continuing execution
- Response will pass through focused response generation for natural output
- MUST specify awaiting_info (what you need)
- MUST specify entities (what you already have)

REQUEST_CONFIRMATION:
- Use BEFORE executing critical actions: book_appointment, cancel_appointment, reschedule_appointment
- MUST specify confirmation_summary with full details (action, details, tool_name, tool_input)
- Wait for user to confirm before calling the tool

RETRY:
- A tool returned result_type=SYSTEM_ERROR
- Haven't exceeded retry limit
- Tool returned recovery_action
"""
        
        # Get tools description
        logger.info(f"📝 [{self.agent_name}] Step 1: Getting tools description...")
        tools_desc = self._get_tools_description()
        logger.info(f"📝 [{self.agent_name}]   ✅ Tools description length: {len(tools_desc)} chars")
        logger.debug(f"📝 [{self.agent_name}]   → Tools description:\n{tools_desc}")
        
        # Extract key context fields with safe defaults
        logger.info(f"📝 [{self.agent_name}] Step 2: Extracting context fields...")
        user_intent = context.get('user_intent') or context.get('what_user_means', 'Not specified')
        entities = context.get('entities', {})
        constraints = context.get('constraints', [])
        prior_context = context.get('prior_context', 'None')
        routing_action = context.get('routing_action', context.get('action', 'Not specified'))

        # Get entities from the plan (accumulated during execution)
        # NOTE: These come from AgentPlan.entities, NOT GlobalState.
        # GlobalState.entities was removed to eliminate confusion.
        plan_entities = {}
        if exec_context.plan and exec_context.plan.entities:
            plan_entities = exec_context.plan.entities

        logger.info(f"📝 [{self.agent_name}]   → User intent: {user_intent}")
        logger.info(f"📝 [{self.agent_name}]   → Routing action: {routing_action}")
        logger.info(f"📝 [{self.agent_name}]   → Prior context: {prior_context}")
        logger.info(f"📝 [{self.agent_name}]   → Entities: {list(entities.keys()) if entities else 'None'}")
        logger.info(f"✍️ [{self.agent_name}]   → Plan entities: {list(plan_entities.keys()) if plan_entities else 'None'}")
        logger.info(f"📝 [{self.agent_name}]   → Constraints: {len(constraints)} constraint(s)")
        
        # Check for continuation
        logger.info(f"📝 [{self.agent_name}] Step 3: Checking for continuation context...")
        is_continuation = context.get('is_continuation', False)
        continuation_type = context.get('continuation_type')
        selected_option = context.get('selected_option')
        continuation_context = context.get('continuation_context', {})
        
        logger.info(f"📝 [{self.agent_name}]   → Is continuation: {is_continuation}")
        if is_continuation:
            logger.info(f"📝 [{self.agent_name}]   → Continuation type: {continuation_type}")
            logger.info(f"📝 [{self.agent_name}]   → Selected option: {selected_option}")
            logger.info(f"📝 [{self.agent_name}]   → Continuation context keys: {list(continuation_context.keys())}")
            logger.debug(f"📝 [{self.agent_name}]   → Continuation context:\n{json.dumps(continuation_context, indent=4, default=str)}")
        
        # Build continuation section if applicable
        continuation_section = ""
        if is_continuation:
            logger.info(f"📝 [{self.agent_name}] Step 4: Building continuation section...")
            continuation_section = f"""
═══════════════════════════════════════════════════════════════════════════════
🔄 CONTINUATION CONTEXT (Resuming Previous Flow)
═══════════════════════════════════════════════════════════════════════════════

Type: {continuation_type}
Selected Option: {selected_option if selected_option else 'None'}

Previous State:
{json.dumps(continuation_context, indent=2, default=str)}

This is a CONTINUATION - the user is responding to previous options/questions.
DO NOT start from scratch. BUILD ON what was already resolved.
"""
            logger.info(f"📝 [{self.agent_name}]   ✅ Continuation section built ({len(continuation_section)} chars)")
        else:
            logger.info(f"📝 [{self.agent_name}] Step 4: No continuation - skipping continuation section")
        
        # Get plan display (Phase 6 - replaced criteria with plan)
        logger.info(f"📝 [{self.agent_name}] Step 5: Getting plan display...")
        plan_display = ""
        if exec_context.plan:
            plan = exec_context.plan
            completed = len(plan.get_completed_task_ids())
            total = len(plan.tasks)
            plan_display = f"Plan: {plan.objective}\nStatus: {plan.status.value}\nProgress: {completed}/{total} tasks complete"
        else:
            plan_display = "No execution plan yet."
        logger.info(f"📝 [{self.agent_name}]   ✅ Plan display ({len(plan_display)} chars)")
        logger.debug(f"📝 [{self.agent_name}]   → Plan:\n{plan_display}")
        
        logger.info(f"📝 [{self.agent_name}] Step 6: Getting observations summary...")
        observations_summary = exec_context.get_observations_summary()
        logger.info(f"📝 [{self.agent_name}]   ✅ Observations summary ({len(observations_summary)} chars)")
        logger.debug(f"📝 [{self.agent_name}]   → Observations:\n{observations_summary}")

        # Get agent-specific instructions (only add section if instructions exist)
        logger.info(f"📝 [{self.agent_name}] Step 6.5: Getting agent-specific instructions...")
        agent_instructions = self._get_agent_instructions()
        instructions_section = f"""
═══════════════════════════════════════════════════════════════════════════════
⚠️  AGENT-SPECIFIC INSTRUCTIONS (FOLLOW STRICTLY)
═══════════════════════════════════════════════════════════════════════════════

{agent_instructions}

""" if agent_instructions else ""
        if agent_instructions:
            logger.info(f"📝 [{self.agent_name}]   ✅ Agent instructions ({len(agent_instructions)} chars)")
        else:
            logger.info(f"📝 [{self.agent_name}]   → No agent-specific instructions")

        # Build the final prompt
        logger.info(f"📝 [{self.agent_name}] Step 7: Assembling final prompt...")

        prompt = f"""
═══════════════════════════════════════════════════════════════════════════════
🎯 REASONING ENGINE ANALYSIS (Your Instructions)
═══════════════════════════════════════════════════════════════════════════════

The reasoning engine has already analyzed the user's request. USE THIS GUIDANCE:

What User Really Wants: {user_intent}
Recommended Action: {routing_action}
Prior Context: {prior_context}

{instructions_section}

Entities Identified (from reasoning):
{json.dumps(entities, indent=2, default=str) if entities else '(none)'}

Resolved Entities (from previous conversation turns):
{json.dumps(plan_entities, indent=2, default=str) if plan_entities else '(none)'}

Constraints:
{chr(10).join(f'  - {c}' for c in constraints) if constraints else '  (none)'}

⚠️  CRITICAL: The reasoning engine has done the intent analysis. Your job is to EXECUTE.
    Only ask for clarification if you discover during execution that critical data is MISSING
    (e.g., doctor not found in system, time slot unavailable).
    
    DO NOT re-analyze the user's intent. TRUST the reasoning engine's interpretation.
{continuation_section}
═══════════════════════════════════════════════════════════════════════════════
📋 CURRENT SITUATION
═══════════════════════════════════════════════════════════════════════════════

User's Literal Message: {message}

═══════════════════════════════════════════════════════════════════════════════
📋 EXECUTION PLAN (Phase 6 - Plan-based execution)
═══════════════════════════════════════════════════════════════════════════════

{plan_display}

═══════════════════════════════════════════════════════════════════════════════
📊 EXECUTION HISTORY (Iteration {exec_context.iteration})
═══════════════════════════════════════════════════════════════════════════════

{observations_summary}

═══════════════════════════════════════════════════════════════════════════════
🛠️  AVAILABLE TOOLS
═══════════════════════════════════════════════════════════════════════════════

{tools_desc}

═══════════════════════════════════════════════════════════════════════════════
🤔 YOUR TASK
═══════════════════════════════════════════════════════════════════════════════

Analyze the situation and decide your next action.

Respond with JSON:
{{
    "analysis": "What I observe and understand about the current state",
    
    "plan_status": {{
        "completed_tasks": ["list of completed task IDs"],
        "pending_tasks": ["list of pending task IDs"],
        "blocked_tasks": ["list of blocked task IDs with reasons"]
    }},
    
    "last_result_analysis": {{
        "tool": "name of last tool called (or null)",
        "result_type": "success/partial/user_input/recoverable/fatal/system_error",
        "interpretation": "what this result means for our task"
    }},
    
    "decision": "CALL_TOOL | RESPOND_COMPLETE | RESPOND_WITH_OPTIONS | RESPOND_IMPOSSIBLE | CLARIFY | COLLECT_INFORMATION | REQUEST_CONFIRMATION | RETRY",
    "reasoning": "Why I chose this decision",
    
    "is_task_complete": true/false,
    
    // ═══════════════════════════════════════════════════════════════
    // If CALL_TOOL - specify tool to call
    // ═══════════════════════════════════════════════════════════════
    "tool_name": "name of tool",
    "tool_input": {{}},
    
    // ═══════════════════════════════════════════════════════════════
    // RESPONSE DATA - Fill based on decision type
    // ═══════════════════════════════════════════════════════════════
    "response": {{
        // Always fill entities with EVERYTHING you know so far:
        //
        // - Include all relevant facts collected in this conversation
        // - Merge in new information from the current turn
        // - Overwrite values that changed (e.g., new doctor preference)
        // - Remove entries that are no longer valid for the current plan
        "entities": {{
            "patient_id": "...",
            "doctor_name": "...",
            "date": "...",
            "time": "..."
        }},
        
        // For COLLECT_INFORMATION:
        "information_needed": "visit_reason | preferred_time | doctor_preference | ...",
        "information_question": "Natural question to ask user",
        
        // For CLARIFY:
        "clarification_needed": "what's unclear",
        "clarification_question": "Question to resolve ambiguity",
        
        // For RESPOND_WITH_OPTIONS:
        "options": ["option1", "option2", "option3"],
        "options_context": "available_times | doctors | dates",
        "options_reason": "Why original request couldn't be fulfilled",
        
        // For RESPOND_IMPOSSIBLE:
        "failure_reason": "Why task cannot be completed",
        "failure_suggestion": "What user can do instead",
        
        // For RESPOND_COMPLETE:
        "completion_summary": "What was accomplished",
        "completion_details": {{
            "appointment_id": "...",
            "doctor": "...",
            "date": "...",
            "time": "..."
        }},
        
        // For REQUEST_CONFIRMATION:
        "confirmation_action": "book_appointment | cancel_appointment",
        "confirmation_details": {{}},
        "confirmation_question": "Should I proceed with...?"
    }}
}}

═══════════════════════════════════════════════════════════════════════════════
DECISION → RESPONSE MAPPING
═══════════════════════════════════════════════════════════════════════════════

COLLECT_INFORMATION:
  → Fill: information_needed, information_question, entities
  → Example: Need visit_reason before booking
  → Response will ASK for missing info, NOT promise action

CLARIFY:
  → Fill: clarification_needed, clarification_question, entities
  → Example: "3pm" could mean today or tomorrow
  → Response will ask clarifying question

RESPOND_WITH_OPTIONS:
  → Fill: options, options_context, options_reason, entities
  → Example: Requested time unavailable, here are alternatives
  → Response will present options

RESPOND_IMPOSSIBLE:
  → Fill: failure_reason, failure_suggestion, entities
  → Example: Doctor doesn't exist in system
  → Response will explain failure and suggest alternatives

RESPOND_COMPLETE:
  → Fill: completion_summary, completion_details, entities
  → Example: Appointment successfully booked
  → Response will confirm what was done with details

REQUEST_CONFIRMATION:
  → Fill: confirmation_action, confirmation_details, confirmation_question
  → Example: About to book - confirm details first
  → Response will ask for confirmation

CRITICAL RULES:
- NEVER say "I'll check..." or "I'll do..." unless you're about to CALL_TOOL
- COLLECT_INFORMATION = ASK for info, don't promise action
- Only RESPOND_COMPLETE after tools have actually succeeded
- Fill entities with EVERYTHING you know so far

DECISION RULES:
1. If reasoning engine provided clear guidance → EXECUTE IT (call appropriate tools)
2. If all criteria are ✅ → RESPOND_COMPLETE with is_task_complete=true
3. If last result has alternatives and available=false → RESPOND_WITH_OPTIONS
4. If task impossible → RESPOND_IMPOSSIBLE
5. Only CLARIFY if execution reveals missing CRITICAL data not in context
6. Use COLLECT_INFORMATION when you need to exit agentic loop to gather information from user
"""
            
        logger.info(f"📝 [{self.agent_name}]   ✅ Final prompt assembled")
        logger.info(f"📝 [{self.agent_name}]   → Total prompt length: {len(prompt)} chars")
        logger.info(f"📝 [{self.agent_name}] ─────────────────────────────────────────")
        logger.info(f"📝 [{self.agent_name}] PROMPT BUILDING COMPLETE")
        logger.info(f"📝 [{self.agent_name}] ─────────────────────────────────────────")
        logger.info(80*"📜")
        logger.info(f"📜 [{self.agent_name}]   → Prompt length: {len(prompt)} chars")
        logger.info(f"📜 [{self.agent_name}] Full thinking prompt content:\n{prompt}")
        logger.info(80*"📜")

        return prompt

    def _parse_thinking_response(
        self,
        response: str,
        exec_context: ExecutionContext
    ) -> ThinkingResult:
        """Parse the LLM's thinking response."""
        logger.info(f"🔍 [{self.agent_name}] ─────────────────────────────────────────")
        logger.info(f"🔍 [{self.agent_name}] PARSING THINKING RESPONSE")
        logger.info(f"🔍 [{self.agent_name}] ─────────────────────────────────────────")
        
        try:
            # Extract JSON
            import re
            logger.info(f"🔍 [{self.agent_name}] Step 1: Extracting JSON from response...")
            logger.info(f"🔍 [{self.agent_name}]   → Response preview: {response[:300]}...")
            
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                logger.error(f"🔍 [{self.agent_name}]   ❌ No JSON found in response!")
                logger.error(f"🔍 [{self.agent_name}]   → Full response: {response}")
                raise ValueError("No JSON found in response")
            
            json_str = json_match.group()
            logger.info(f"🔍 [{self.agent_name}]   ✅ JSON extracted successfully")
            logger.info(f"🔍 [{self.agent_name}]   → JSON length: {len(json_str)} chars")
            logger.debug(f"🔍 [{self.agent_name}]   → Extracted JSON:\n{json_str}")
            
            # Parse JSON
            logger.info(f"🔍 [{self.agent_name}] Step 2: Parsing JSON...")
            data = json.loads(json_str)
            logger.info(f"🔍 [{self.agent_name}]   ✅ JSON parsed successfully")
            logger.info(f"🔍 [{self.agent_name}]   → Top-level keys: {list(data.keys())}")
            
            # Parse decision
            logger.info(f"🔍 [{self.agent_name}] Step 3: Extracting decision...")
            decision_str = data.get("decision", "RESPOND").upper()
            logger.info(f"🔍 [{self.agent_name}]   → Raw decision string: '{decision_str}'")
            
            decision_map = {
                "CALL_TOOL": AgentDecision.CALL_TOOL,
                "RESPOND": AgentDecision.RESPOND,
                "RESPOND_WITH_OPTIONS": AgentDecision.RESPOND_WITH_OPTIONS,
                "RESPOND_OPTIONS": AgentDecision.RESPOND_WITH_OPTIONS,
                "RESPOND_COMPLETE": AgentDecision.RESPOND_COMPLETE,
                "RESPOND_IMPOSSIBLE": AgentDecision.RESPOND_IMPOSSIBLE,
                "CLARIFY": AgentDecision.CLARIFY,
                "RETRY": AgentDecision.RETRY,
                "EXECUTE_RECOVERY": AgentDecision.EXECUTE_RECOVERY,
                "COLLECT_INFORMATION": AgentDecision.COLLECT_INFORMATION,
                "REQUEST_CONFIRMATION": AgentDecision.REQUEST_CONFIRMATION,  # NEW
                "CONFIRM": AgentDecision.REQUEST_CONFIRMATION  # Alias
            }
            decision = decision_map.get(decision_str, AgentDecision.RESPOND)
            logger.info(f"🔍 [{self.agent_name}]   → Mapped to: {decision.value}")
            
            if decision_str not in decision_map:
                logger.warning(f"🔍 [{self.agent_name}]   ⚠️  Unknown decision '{decision_str}', defaulting to RESPOND")
            
            # Extract other fields
            logger.info(f"🔍 [{self.agent_name}] Step 4: Extracting additional fields...")
            
            analysis = data.get("analysis", "")
            logger.info(f"🔍 [{self.agent_name}]   → Analysis: {analysis[:150]}{'...' if len(analysis) > 150 else ''}")
            
            reasoning = data.get("reasoning", "")
            logger.info(f"🔍 [{self.agent_name}]   → Reasoning: {reasoning[:150]}{'...' if len(reasoning) > 150 else ''}")
            
            is_task_complete = data.get("is_task_complete", False)
            logger.info(f"🔍 [{self.agent_name}]   → Is task complete: {is_task_complete}")
            
            # Update criteria based on assessment
            logger.info(f"🔍 [{self.agent_name}] Step 5: Processing criteria assessment...")
            assessment = data.get("criteria_assessment", {})
            logger.info(f"🔍 [{self.agent_name}]   → Assessment keys: {list(assessment.keys()) if assessment else 'None'}")
            
            completed_criteria = assessment.get("complete", [])
            if completed_criteria:
                logger.info(f"🔍 [{self.agent_name}]   → Criteria marked complete: {completed_criteria}")
                for completed in completed_criteria:
                    logger.info(f"🔍 [{self.agent_name}]       • Marking '{completed}' as complete...")
                    exec_context.mark_criterion_complete(completed)
                    logger.info(f"🔍 [{self.agent_name}]         ✅ Marked")
            else:
                logger.info(f"🔍 [{self.agent_name}]   → No criteria marked complete")
            
            pending_criteria = assessment.get("pending", [])
            if pending_criteria:
                logger.info(f"🔍 [{self.agent_name}]   → Criteria still pending: {pending_criteria}")
            
            blocked_criteria = assessment.get("blocked", [])
            if blocked_criteria:
                logger.info(f"🔍 [{self.agent_name}]   → Criteria blocked: {blocked_criteria}")
            
            # Extract last result analysis
            logger.info(f"🔍 [{self.agent_name}] Step 6: Processing last result analysis...")
            last_result = data.get("last_result_analysis", {})
            if last_result:
                logger.info(f"🔍 [{self.agent_name}]   → Last tool: {last_result.get('tool', 'N/A')}")
                logger.info(f"🔍 [{self.agent_name}]   → Result type: {last_result.get('result_type', 'N/A')}")
                logger.info(f"🔍 [{self.agent_name}]   → Interpretation: {last_result.get('interpretation', 'N/A')[:100]}...")
            
            # Extract decision-specific data
            logger.info(f"🔍 [{self.agent_name}] Step 7: Extracting decision-specific data...")
            tool_name = None
            tool_input = None
            clarification_question = None
            awaiting_info = None
            entities = None
            
            if decision == AgentDecision.CALL_TOOL:
                tool_name = data.get("tool_name")
                tool_input = data.get("tool_input")
                logger.info(f"🔍 [{self.agent_name}]   → Tool to call: {tool_name}")
                logger.info(f"🔍 [{self.agent_name}]   → Tool input: {json.dumps(tool_input, indent=6, default=str) if tool_input else 'None'}")
                
                if not tool_name:
                    logger.warning(f"🔍 [{self.agent_name}]   ⚠️  CALL_TOOL decision but no tool_name provided!")
                
            elif decision == AgentDecision.CLARIFY:
                clarification_question = data.get("clarification_question")
                logger.info(f"🔍 [{self.agent_name}]   → Clarification question: {clarification_question}")
                
                if not clarification_question:
                    logger.warning(f"🔍 [{self.agent_name}]   ⚠️  CLARIFY decision but no clarification_question provided!")
            
            elif decision == AgentDecision.COLLECT_INFORMATION:
                logger.info(f"🔍 [{self.agent_name}]   → Collecting information - will generate focused response")
            
            # Extract awaiting_info and entities for COLLECT_INFORMATION and CLARIFY
            if decision in [AgentDecision.COLLECT_INFORMATION, AgentDecision.CLARIFY]:
                awaiting_info = data.get("awaiting_info")
                entities = data.get("entities", {})
                logger.info(f"🔍 [{self.agent_name}]   → Awaiting info: {awaiting_info}")
                logger.info(f"✍️ [{self.agent_name}]   → Resolved entities (from thinking JSON): {list(entities.keys()) if entities else 'None'}")
            
            # Extract confirmation_summary for REQUEST_CONFIRMATION (legacy)
            confirmation_summary = None
            if decision == AgentDecision.REQUEST_CONFIRMATION:
                confirmation_data = data.get("confirmation_summary")
                if confirmation_data:
                    try:
                        confirmation_summary = ConfirmationSummary(**confirmation_data)
                        logger.info(f"🔍 [{self.agent_name}]   → Confirmation action: {confirmation_summary.action}")
                        logger.info(f"🔍 [{self.agent_name}]   → Confirmation tool: {confirmation_summary.tool_name}")
                    except Exception as e:
                        logger.warning(f"🔍 [{self.agent_name}]   ⚠️  Failed to parse confirmation_summary: {e}")
                else:
                    logger.warning(f"🔍 [{self.agent_name}]   ⚠️  REQUEST_CONFIRMATION but no confirmation_summary provided!")
            
            # Extract tool expansion request (optional fields)
            request_all_tools = data.get("request_all_tools", False)
            tool_expansion_reason = data.get("tool_expansion_reason")
            
            if request_all_tools:
                logger.info(f"🔍 [{self.agent_name}]   → Expansion requested: {tool_expansion_reason}")
            
            # Extract response object (NEW structured approach)
            logger.info(f"🔍 [{self.agent_name}] Step 8: Extracting response data...")
            response_data = None
            response_obj = data.get("response", {})
            
            if response_obj:
                try:
                    # Create AgentResponseData from response object
                    response_data = AgentResponseData(**response_obj)
                    logger.info(f"🔍 [{self.agent_name}]   ✅ Response data extracted successfully")
                    logger.info(f"🔍 [{self.agent_name}]   → Response fields: {list(response_obj.keys())}")
                except Exception as e:
                    logger.warning(f"🔍 [{self.agent_name}]   ⚠️  Failed to parse response object: {e}")
                    logger.warning(f"🔍 [{self.agent_name}]   → Response object: {response_obj}")
                    # Create empty response data as fallback
                    response_data = AgentResponseData()
            else:
                # Fallback to legacy fields for backward compatibility
                logger.info(f"🔍 [{self.agent_name}]   → No response object found, using legacy fields")
                response_data = AgentResponseData(
                    information_needed=awaiting_info,
                    information_question=None,  # Will be generated from information_needed
                    clarification_needed=None,
                    clarification_question=clarification_question,
                    entities=entities or {},
                    confirmation_action=confirmation_summary.action if confirmation_summary else None,
                    confirmation_details=confirmation_summary.details if confirmation_summary else {},
                    confirmation_question=None  # Will be generated
                )
            
            # Extract suggested_response from top-level JSON
            suggested_response = data.get("suggested_response")
            if suggested_response and response_data:
                response_data.suggested_response = suggested_response
                logger.info(f"🔍 [{self.agent_name}]   → Suggested response extracted: {suggested_response[:100]}...")
            
            # Build result
            logger.info(f"🔍 [{self.agent_name}] Step 9: Building ThinkingResult object...")

            # Determine updated_entities for persistent conversation memory.
            # Priority order:
            # 1) entities_to_update (delta format) - merge into current entities
            # 2) response.entities if present (from LLM structured response - complete state)
            # 3) legacy entities field if present (from LLM JSON)
            # 4) fall back to plan's entities (accumulated during execution)
            # NOTE: GlobalState.entities was removed - use plan or context instead.
            previous_resolved = {}
            if exec_context.plan and exec_context.plan.entities:
                previous_resolved = exec_context.plan.entities.copy()
            updated_resolved = previous_resolved.copy()

            # NEW: Initialize LLM delta tracking (ISOLATED from multi-source entities)
            llm_delta = {}

            # ✅ NEW: Check for delta format first
            if response_obj and "entities_to_update" in response_obj:
                # Delta update format
                delta = response_obj.get("entities_to_update", {})
                llm_delta = delta.copy()  # NEW: Preserve for LLM-only tracking
                
                # Track metrics
                try:
                    from patient_ai_service.core.observability import (
                        delta_update_count,
                        delta_size
                    )
                    delta_update_count.labels(format='delta').inc()
                    delta_size.observe(len(delta))
                except Exception:
                    pass  # Don't fail on metrics errors
                
                # Apply delta (update existing, add new)
                for key, value in delta.items():
                    updated_resolved[key] = value

                logger.info(
                    f"✅ Applied delta update: {len(delta)} entities changed/added "
                    f"(LLM track: {len(llm_delta)}, Total internal: {len(updated_resolved)})"
                )
                
                # Also update response_data.entities for backward compatibility
                if response_data:
                    response_data.entities = updated_resolved.copy()
            
            # Fallback to complete state format
            elif response_data and response_data.entities:
                # Track metrics
                try:
                    from patient_ai_service.core.observability import delta_update_count
                    delta_update_count.labels(format='complete').inc()
                except Exception:
                    pass
                
                updated_resolved.update(response_data.entities)
                logger.info(
                    f"Using complete state format: {len(response_data.entities)} entities"
                )
            elif entities:
                updated_resolved.update(entities)

            result = ThinkingResult(
                analysis=analysis,
                task_status=assessment,
                decision=decision,
                reasoning=reasoning,
                tool_name=tool_name,
                tool_input=tool_input,
                response_text=None,  # No longer extracted from thinking - generated after decision
                clarification_question=clarification_question,  # Legacy
                is_task_complete=is_task_complete,
                awaiting_info=awaiting_info,  # Legacy
                entities=entities,  # Legacy
                confirmation_summary=confirmation_summary,  # Legacy
                response=response_data,  # NEW structured response
                updated_entities=updated_resolved,
                llm_delta=llm_delta,  # NEW: LLM-only entity updates (ISOLATED)
                request_all_tools=request_all_tools,
                tool_expansion_reason=tool_expansion_reason
            )
            
            logger.info(f"🔍 [{self.agent_name}]   ✅ ThinkingResult created successfully")
            logger.info(f"🔍 [{self.agent_name}] ─────────────────────────────────────────")
            logger.info(f"🔍 [{self.agent_name}] PARSING COMPLETE - Decision: {decision.value}")
            logger.info(f"🔍 [{self.agent_name}] ─────────────────────────────────────────")
            
            return result
        
        except json.JSONDecodeError as e:
            logger.error(f"🔍 [{self.agent_name}]   ❌ JSON parsing failed!")
            logger.error(f"🔍 [{self.agent_name}]   → Error: {str(e)}")
            logger.error(f"🔍 [{self.agent_name}]   → Error location: line {e.lineno}, col {e.colno}")
            logger.error(f"🔍 [{self.agent_name}]   → Response preview: {response[:500]}")
            logger.debug(f"🔍 [{self.agent_name}]   → Full response: {response}")
            
            # Default to asking for clarification on parse error
            logger.info(f"🔍 [{self.agent_name}]   → Returning fallback CLARIFY decision")
            return ThinkingResult(
                decision=AgentDecision.CLARIFY,
                reasoning=f"JSON parse error: {e}",
                clarification_question="I'm having trouble understanding. Could you please rephrase your request?"
            )
        
        except Exception as e:
            logger.error(f"🔍 [{self.agent_name}]   ❌ Unexpected error during parsing!")
            logger.error(f"🔍 [{self.agent_name}]   → Error type: {type(e).__name__}")
            logger.error(f"🔍 [{self.agent_name}]   → Error message: {str(e)}")
            logger.error(f"🔍 [{self.agent_name}]   → Stack trace:", exc_info=True)
            logger.debug(f"🔍 [{self.agent_name}]   → Response was: {response[:500]}")
            
            # Default to asking for clarification on parse error
            logger.info(f"🔍 [{self.agent_name}]   → Returning fallback CLARIFY decision")
            return ThinkingResult(
                decision=AgentDecision.CLARIFY,
                reasoning=f"Parse error: {e}",
                clarification_question="I'm having trouble understanding. Could you please rephrase your request?"
            )

    def _extract_entities(self, exec_context: ExecutionContext) -> Dict[str, Any]:
        """Extract what we've already resolved from execution context and observations."""
        resolved = {}
        
        # From context
        context = self._context.get(exec_context.session_id, {})
        entities = context.get('entities', {})
        for key in ['patient_id', 'doctor_id', 'doctor_name', 'date', 'time', 'procedure']:
            if key in entities and entities[key]:
                resolved[key] = entities[key]
        
        # From successful tool results
        for obs in exec_context.observations:
            if obs.is_success() and obs.type == "tool":
                result = obs.result
                # Extract key booking entities from tool results
                if 'doctor_id' in result:
                    resolved['doctor_id'] = result['doctor_id']
                if 'doctor_name' in result:
                    resolved['doctor_name'] = result['doctor_name']
                if 'date' in result:
                    resolved['date'] = result['date']
                if 'available_slots' in result and result['available_slots']:
                    resolved['available_slots'] = result['available_slots']
        
        return resolved

    def _mark_plan_blocked_and_store(
        self,
        session_id: str,
        exec_context: ExecutionContext,
        awaiting: str,
        options: Optional[List[str]] = None
    ):
        """
        Mark the current plan as BLOCKED and persist it for next turn.

        Called when the agent needs user input (CLARIFY, COLLECT_INFORMATION, REQUEST_CONFIRMATION).
        This ensures the plan can be resumed when the user responds.

        Args:
            session_id: Session identifier
            exec_context: Current execution context (contains the plan)
            awaiting: What we're waiting for from the user
            options: Optional list of valid options (for selections/confirmations)
        """
        if exec_context.plan:
            reason = f"Waiting for user to provide: {awaiting}"
            exec_context.plan.mark_blocked(
                reason=reason,
                awaiting=awaiting,
                options=options or []
            )

            # 🔍 ADD DETAILED LOGGING HERE
            logger.info(f"🔍🔍🔍 [PLAN SAVE DEBUG] ═══════════════════════════════")
            logger.info(f"🔍 Session ID: {session_id}")
            logger.info(f"🔍 Agent name (self): {self.agent_name}")
            logger.info(f"🔍 Agent name (plan): {exec_context.plan.agent_name}")
            logger.info(f"🔍 Plan status: {exec_context.plan.status}")
            logger.info(f"🔍 Plan objective: {exec_context.plan.objective}")
            logger.info(f"🔍 Plan entities: {list(exec_context.plan.entities.keys())}")
            logger.info(f"🔍 Plan awaiting: {exec_context.plan.awaiting_info}")
            logger.info(f"🔍🔍🔍 ════════════════════════════════════════════════")

            self.state_manager.store_agent_plan(session_id, exec_context.plan)
            logger.info(f"🛑 [{self.agent_name}] Plan marked BLOCKED and stored: {reason}")
        else:
            logger.warning(f"📋 [{self.agent_name}] No plan to mark blocked (awaiting={awaiting})")
    
    def _build_confirmation_summary(
        self, 
        action: str, 
        details: Dict[str, Any],
        tool: Optional[str]
    ) -> str:
        """Build human-readable confirmation summary for UnifiedReasoning."""
        if action == "book_appointment" or tool == "book_appointment":
            doctor = details.get("doctor_name", "the doctor")
            date = details.get("date", "the selected date")
            time = details.get("time", "the selected time")
            return f"Confirm booking appointment with {doctor} on {date} at {time}"
        
        elif action == "cancel_appointment" or tool == "cancel_appointment":
            appt_id = details.get("appointment_id", "the appointment")
            return f"Confirm cancellation of appointment {appt_id}"
        
        elif action == "reschedule_appointment" or tool == "reschedule_appointment":
            new_date = details.get("new_date", details.get("date", ""))
            new_time = details.get("new_time", details.get("time", ""))
            return f"Confirm rescheduling appointment to {new_date} at {new_time}"
        
        elif action == "register_patient" or tool == "register_patient":
            name = details.get("name", details.get("first_name", "patient"))
            return f"Confirm registration for {name}"
        
        else:
            return f"Confirm action: {action}"
    
    def _get_action_type(self, action: str) -> str:
        """Map action to action_type category."""
        action_lower = action.lower()
        if "book" in action_lower:
            return "booking"
        elif "cancel" in action_lower:
            return "cancellation"
        elif "reschedule" in action_lower:
            return "reschedule"
        elif "register" in action_lower:
            return "registration"
        else:
            return "action"

    def _get_tools_description(self, filter_tools: Optional[List[str]] = None) -> str:
        """
        Get description of available tools with FULL parameter details for the prompt.

        Args:
            filter_tools: Optional list of tool names to include. If None, includes all tools.
                         Each included tool receives its COMPLETE description with all parameters.

        Returns:
            Formatted string description of tools
        """
        if not hasattr(self, '_tool_schemas') or not self._tool_schemas:
            return "No tools available."

        # Apply tool filter if provided
        if filter_tools is not None:
            if not filter_tools:
                return "No tools available."

            # Filter to only requested tools (but keep FULL descriptions)
            filtered_schemas = [s for s in self._tool_schemas if s.get("name") in filter_tools]

            if not filtered_schemas:
                logger.warning(f"[{self.agent_name}] Tool filter requested {filter_tools} but none found in schemas")
                return "No tools available."

            schemas_to_describe = filtered_schemas
        else:
            # No filter - use all tools
            schemas_to_describe = self._tool_schemas

        lines = []

        for schema in schemas_to_describe:
            name = schema.get("name")
            description = schema.get("description", "No description")
            params = schema['input_schema'].get('properties', {})
            required = set(schema['input_schema'].get('required', []))

            # Tool header
            lines.append(f"\n═══ {name} ═══")
            lines.append(f"{description}")

            # Parameters
            if params:
                lines.append("\nParameters:")
                for param_name, param_info in params.items():
                    param_type = param_info.get('type', 'any')
                    param_desc = param_info.get('description', '')
                    is_required = param_name in required

                    req_marker = "REQUIRED" if is_required else "optional"
                    lines.append(f"  • {param_name} ({param_type}) [{req_marker}]: {param_desc}")
            else:
                lines.append("\nNo parameters required")

        return "\n".join(lines)

    def _infer_result_type(self, result: Dict[str, Any]) -> str:
        """Infer result type from result content if not explicitly set."""
        # Check for explicit result_type
        if "result_type" in result:
            try:
                return ToolResultType(result["result_type"]).value
            except ValueError:
                pass
        
        # Infer from content
        if result.get("success") is False:
            if result.get("recovery_action"):
                return ToolResultType.RECOVERABLE.value
            elif result.get("should_retry"):
                return ToolResultType.SYSTEM_ERROR.value
            else:
                return ToolResultType.FATAL.value
        
        if result.get("alternatives") and not result.get("available", True):
            return ToolResultType.USER_INPUT_NEEDED.value
        
        if result.get("next_action") or result.get("can_proceed") is True:
            return ToolResultType.PARTIAL.value
        
        if result.get("success") is True:
            if result.get("satisfies_criteria") or result.get("appointment_id"):
                return ToolResultType.SUCCESS.value
            return ToolResultType.PARTIAL.value
        
        return ToolResultType.PARTIAL.value  # Default

    def _create_error_signature(self, tool_name: str, error: str) -> str:
        """Create normalized error signature for tracking recovery attempts."""
        import re
        # Normalize error: take first 100 chars, lowercase, remove variable parts
        normalized = str(error)[:100].lower().strip()
        # Remove UUIDs
        normalized = re.sub(r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b', '<uuid>', normalized)
        # Remove dates
        normalized = re.sub(r'\d{4}-\d{2}-\d{2}', '<date>', normalized)
        # Remove timestamps
        normalized = re.sub(r'\d{2}:\d{2}:\d{2}', '<time>', normalized)
        return f"{tool_name}:{normalized}"

    # ==================== NEW AGENTIC EXECUTION METHODS ====================

    async def _think(
        self,
        session_id: str,
        message: str,
        exec_context: ExecutionContext
    ) -> ThinkingResult:
        """
        Think about the current situation and decide next action.
        
        Uses layered prompt caching:
        - Layer 1 (Cached): Universal guides, decision rules
        - Layer 2 (Not Cached): Agent identity, tools
        - Layer 3 (Not Cached): Dynamic context
        """
        logger.info(f"🧠 [{self.agent_name}] ════════════════════════════════════════════════════════")
        logger.info(f"🧠 [{self.agent_name}] ENTERING _think() - Iteration {exec_context.iteration}")
        logger.info(f"🧠 [{self.agent_name}] ════════════════════════════════════════════════════════")
        plan = exec_context.plan
        logger.info(f"🧠 [{self.agent_name}] ════════════════════════════════════════════════════════")
        if plan:
            logger.info(f"🧠 [{self.agent_name}] Thinking - Plan status: {plan.status.value}")
            logger.info(f"🧠 [{self.agent_name}]   Completed: {len(plan.get_completed_task_ids())}/{len(plan.tasks)}")
        else:
            logger.info(f"🧠 [{self.agent_name}] Thinking - No plan (no_plan action)")
        
        # ═════════════════════════════════════════════════════════════════════
        # STEP 1: GATHER INPUT CONTEXT
        # ═════════════════════════════════════════════════════════════════════
        logger.info(f"🧠 [{self.agent_name}] [STEP 1/5] Gathering input context...")
        
        context = self._context.get(session_id, {})
        logger.info(f"🧠 [{self.agent_name}]   → Session context keys: {list(context.keys())}")
        logger.info(f"🧠 [{self.agent_name}]   → User message: '{message[:100]}{'...' if len(message) > 100 else ''}'")
        logger.info(f"🧠 [{self.agent_name}]   → User intent: {context.get('user_intent') or context.get('what_user_means', 'N/A')}")
        logger.info(f"🧠 [{self.agent_name}]   → Routing action: {context.get('routing_action', context.get('action', 'N/A'))}")
        logger.info(f"🧠 [{self.agent_name}]   → Is continuation: {context.get('is_continuation', False)}")
        if context.get('is_continuation'):
            logger.info(f"🧠 [{self.agent_name}]   → Continuation type: {context.get('continuation_type')}")
            logger.info(f"🧠 [{self.agent_name}]   → Selected option: {context.get('selected_option')}")
        
        # Log entities and constraints
        entities = context.get('entities', {})
        if entities:
            logger.info(f"🧠 [{self.agent_name}]   → Entities: {json.dumps(entities, indent=6, default=str)}")
        constraints = context.get('constraints', [])
        if constraints:
            logger.info(f"🧠 [{self.agent_name}]   → Constraints: {constraints}")
        
        # Log current criteria state
        logger.info(f"🧠 [{self.agent_name}]   → Success criteria ({len(exec_context.criteria)}):")
        for crit_id, crit in exec_context.criteria.items():
            logger.info(f"🧠 [{self.agent_name}]       • [{crit.state.value.upper()}] {crit.description}")
            if crit.state == CriterionState.COMPLETE:
                logger.info(f"🧠 [{self.agent_name}]         Evidence: {crit.completion_evidence}")
            elif crit.state == CriterionState.BLOCKED:
                logger.info(f"🧠 [{self.agent_name}]         Reason: {crit.blocked_reason}")
                if crit.blocked_options:
                    logger.info(f"🧠 [{self.agent_name}]         Options: {crit.blocked_options[:3]}{'...' if len(crit.blocked_options) > 3 else ''}")
            elif crit.state == CriterionState.FAILED:
                logger.info(f"🧠 [{self.agent_name}]         Reason: {crit.failed_reason}")
        
        # Log observations (tool results so far)
        logger.info(f"🧠 [{self.agent_name}]   → Observations so far ({len(exec_context.observations)}):")
        for i, obs in enumerate(exec_context.observations, 1):
            status = "✅" if obs.is_success() else "❌"
            logger.info(f"🧠 [{self.agent_name}]       {i}. {status} [{obs.result_type.value if obs.result_type else 'unknown'}] {obs.type}:{obs.name}")
            # Log result summary (first 200 chars)
            result_str = json.dumps(obs.result, default=str)[:200]
            logger.info(f"🧠 [{self.agent_name}]          Result: {result_str}{'...' if len(json.dumps(obs.result, default=str)) > 200 else ''}")
        
        # ═════════════════════════════════════════════════════════════════════
        # STEP 2: BUILD THINKING PROMPT
        # ═════════════════════════════════════════════════════════════════════
        logger.info(f"🧠 [{self.agent_name}] [STEP 2/5] Building thinking prompt...")
        
        prompt = self._build_thinking_prompt(message, context, exec_context)
                
        # ═════════════════════════════════════════════════════════════════════
        # STEP 3: PREPARE LLM CALL
        # ═════════════════════════════════════════════════════════════════════
        logger.info(f"🧠 [{self.agent_name}] [STEP 3/5] Preparing LLM call...")
        
        # Get observability logger
        obs_logger = get_observability_logger(session_id) if settings.enable_observability else None
        
        # Get LLM config
        llm_config = self._get_llm_config_for_function("_think")
        llm_client = self._get_llm_client_for_function("_think")
        
        # Build prompt layers
        # Layer 1: Universal (cached)
        universal_content = get_universal_system_content()
        
        # Layer 2: Agent-specific (not cached)
        agent_content = self._get_thinking_system_prompt(exec_context)
        
        # Layer 3: Dynamic (user prompt)
        dynamic_prompt = self._build_thinking_prompt(message, context, exec_context)
        
        # Check if caching is enabled and client supports it
        use_caching = (
            llm_config.prompt_cache_enabled and
            hasattr(llm_client, 'create_message_with_cache_and_usage') and
            llm_config.provider == 'anthropic'
        )

        logger.info(f"🧠 [{self.agent_name}] ════════════════════════════════════════════════════════")
        logger.info(f"🧠 [{self.agent_name}] PROMPT STRUCTURE:")
        logger.info(f"🧠 [{self.agent_name}] Cache enabled: {use_caching} (config: {llm_config.prompt_cache_enabled}, provider: {llm_config.provider})")
        logger.info(f"🧠 [{self.agent_name}] ════════════════════════════════════════════════════════")

        if use_caching:
            logger.info(f"🧠 [{self.agent_name}] 📌 SYSTEM PROMPT (CACHED) - Layer 1: Universal Content (~{len(universal_content)//4} tokens)")
            logger.info(f"🧠 [{self.agent_name}] ─────────────────────────────────────────────────────────")
            logger.info(f"🧠 [{self.agent_name}] {universal_content}")
            logger.info(f"🧠 [{self.agent_name}] ─────────────────────────────────────────────────────────")
            logger.info(f"🧠 [{self.agent_name}] 📄 SYSTEM PROMPT (NOT CACHED) - Layer 2: Agent-Specific (~{len(agent_content)//4} tokens)")
            logger.info(f"🧠 [{self.agent_name}] ─────────────────────────────────────────────────────────")
            logger.info(f"🧠 [{self.agent_name}] {agent_content}")
            logger.info(f"🧠 [{self.agent_name}] ─────────────────────────────────────────────────────────")
        else:
            combined_system = universal_content + "\n\n" + agent_content
            logger.info(f"🧠 [{self.agent_name}] 📄 SYSTEM PROMPT (NO CACHING) - Combined (~{len(combined_system)//4} tokens)")
            logger.info(f"🧠 [{self.agent_name}] ─────────────────────────────────────────────────────────")
            logger.info(f"🧠 [{self.agent_name}] {combined_system}")
            logger.info(f"🧠 [{self.agent_name}] ─────────────────────────────────────────────────────────")

        logger.info(f"🧠 [{self.agent_name}] 💬 USER PROMPT - Layer 3: Dynamic Context (~{len(dynamic_prompt)//4} tokens)")
        logger.info(f"🧠 [{self.agent_name}] ─────────────────────────────────────────────────────────")
        logger.info(f"🧠 [{self.agent_name}] {dynamic_prompt}")
        logger.info(f"🧠 [{self.agent_name}] ════════════════════════════════════════════════════════")
        
        llm_start_time = time.time()
        
        try:
            if use_caching:
                # Use cached method
                response, tokens = llm_client.create_message_with_cache_and_usage(
                    system_cached=universal_content,
                    system_dynamic=agent_content,
                    messages=[{"role": "user", "content": dynamic_prompt}],
                    temperature=llm_config.temperature,
                    max_tokens=llm_config.max_tokens
                )
                
                # Log cache status
                if tokens.cache_read_input_tokens > 0:
                    logger.info(f"🎯 [{self.agent_name}] CACHE HIT: {tokens.cache_read_input_tokens} tokens from cache")
                elif tokens.cache_creation_input_tokens > 0:
                    logger.info(f"📝 [{self.agent_name}] CACHE WRITE: {tokens.cache_creation_input_tokens} tokens to cache")
            else:
                # Fall back to non-cached method
                system_prompt = universal_content + "\n\n" + agent_content
                
                if hasattr(llm_client, 'create_message_with_usage'):
                    response, tokens = llm_client.create_message_with_usage(
                        system=system_prompt,
                        messages=[{"role": "user", "content": dynamic_prompt}],
                        temperature=llm_config.temperature,
                        max_tokens=llm_config.max_tokens
                    )
                else:
                    response = llm_client.create_message(
                        system=system_prompt,
                        messages=[{"role": "user", "content": dynamic_prompt}],
                        temperature=llm_config.temperature,
                        max_tokens=llm_config.max_tokens
                    )
                    tokens = TokenUsage()
            
            llm_duration_seconds = time.time() - llm_start_time

            logger.info(f"🧠 [{self.agent_name}] ════════════════════════════════════════════════════════")
            logger.info(f"🧠 [{self.agent_name}] ✅ LLM call completed in {llm_duration_seconds:.3f}s")
            logger.info(f"🧠 [{self.agent_name}] Tokens: {tokens.input_tokens} in / {tokens.output_tokens} out")
            if tokens.cache_read_input_tokens > 0:
                logger.info(f"🧠 [{self.agent_name}] Cache hit rate: {tokens.cache_hit_rate:.1%}")

            # Log raw LLM response
            logger.info(f"🧠 [{self.agent_name}] ─────────────────────────────────────────────────────────")
            logger.info(f"🧠 [{self.agent_name}] 📤 RAW LLM RESPONSE:")
            logger.info(f"🧠 [{self.agent_name}] ─────────────────────────────────────────────────────────")
            logger.info(f"🧠 [{self.agent_name}] {response}")
            logger.info(f"🧠 [{self.agent_name}] ════════════════════════════════════════════════════════")
            
            # Record LLM call in observability
            if obs_logger:
                obs_logger.record_llm_call(
                    component=f"agent.{self.agent_name}.think",
                    provider=llm_config.provider,
                    model=llm_config.model,
                    tokens=tokens,
                    duration_seconds=llm_duration_seconds,
                    system_prompt_length=len(universal_content) + len(agent_content),
                    messages_count=1,
                    temperature=llm_config.temperature,
                    max_tokens=llm_config.max_tokens,
                    function_name="_think"
                )
            
            # Record tool scoping statistics
            if obs_logger and hasattr(obs_logger, 'agent_execution'):
                scoping_stats = {}
                
                if exec_context.plan:
                    scoping_stats['plan_tool_count'] = len(exec_context.plan.required_tools)
                    scoping_stats['scoping_enabled'] = exec_context.plan.should_use_scoped_tools()
                    scoping_stats['expansion_requested'] = exec_context.plan.all_tools_enabled
                    scoping_stats['expansion_iteration'] = exec_context.plan.tool_expansion_requested_at
                    
                    if scoping_stats['scoping_enabled']:
                        # Calculate actual token savings
                        full_desc = self._get_tools_description()
                        scoped_desc = self._get_tools_description(
                            filter_tools=exec_context.plan.required_tools
                        )
                        saved_chars = len(full_desc) - len(scoped_desc)
                        scoping_stats['estimated_tokens_saved'] = saved_chars // 4
                    else:
                        scoping_stats['estimated_tokens_saved'] = 0
                    
                    scoping_stats['available_tool_count'] = (
                        len(exec_context.plan.required_tools)
                        if scoping_stats['scoping_enabled']
                        else len(self._tool_schemas)
                    )
                else:
                    # No plan - always full catalog
                    scoping_stats = {
                        'plan_tool_count': 0,
                        'scoping_enabled': False,
                        'expansion_requested': False,
                        'expansion_iteration': None,
                        'estimated_tokens_saved': 0,
                        'available_tool_count': len(self._tool_schemas)
                    }
                
                obs_logger.agent_execution.tool_scoping_stats = scoping_stats
                logger.info(f"📊 [{self.agent_name}] Tool scoping stats: {scoping_stats}")
            
            # ═════════════════════════════════════════════════════════════════
            # STEP 5: PARSE RESPONSE
            # ═════════════════════════════════════════════════════════════════
            logger.info(f"🧠 [{self.agent_name}] [STEP 5/5] Parsing LLM response...")
            
            thinking_result = self._parse_thinking_response(response, exec_context)
            
            logger.info(f"🧠 [{self.agent_name}]   ✅ Parsing completed successfully")
            logger.info(f"🧠 [{self.agent_name}]   → Decision: {thinking_result.decision.value if thinking_result.decision else 'N/A'}")
            logger.info(f"🧠 [{self.agent_name}]   → Reasoning: {thinking_result.reasoning[:200]}{'...' if len(thinking_result.reasoning) > 200 else ''}")
            logger.info(f"🧠 [{self.agent_name}]   → Is task complete: {thinking_result.is_task_complete}")
            
            if thinking_result.analysis:
                logger.info(f"🧠 [{self.agent_name}]   → Analysis: {thinking_result.analysis[:200]}{'...' if len(thinking_result.analysis) > 200 else ''}")
            
            if thinking_result.decision == AgentDecision.CALL_TOOL:
                logger.info(f"🧠 [{self.agent_name}]   → Tool to call: {thinking_result.tool_name}")
                logger.info(f"🧠 [{self.agent_name}]   → Tool input: {json.dumps(thinking_result.tool_input, indent=6, default=str) if thinking_result.tool_input else 'None'}")
            
            if thinking_result.decision == AgentDecision.CLARIFY:
                logger.info(f"🧠 [{self.agent_name}]   → Clarification question: {thinking_result.clarification_question}")
            
            if thinking_result.decision == AgentDecision.COLLECT_INFORMATION:
                logger.info(f"🧠 [{self.agent_name}]   → Collecting information - will exit agentic loop and generate focused response")
            
            if thinking_result.task_status:
                logger.info(f"🧠 [{self.agent_name}]   → Task status assessment:")
                logger.info(f"🧠 [{self.agent_name}]       • Complete: {thinking_result.task_status.get('complete', [])}")
                logger.info(f"🧠 [{self.agent_name}]       • Pending: {thinking_result.task_status.get('pending', [])}")
                logger.info(f"🧠 [{self.agent_name}]       • Blocked: {thinking_result.task_status.get('blocked', [])}")
            
            logger.info(f"🧠 [{self.agent_name}] ════════════════════════════════════════════════════════")
            logger.info(f"🧠 [{self.agent_name}] EXITING _think() - Decision: {thinking_result.decision.value if thinking_result.decision else 'N/A'}")
            logger.info(f"🧠 [{self.agent_name}] ════════════════════════════════════════════════════════")
            
            return thinking_result
        
        except Exception as e:
            llm_duration_seconds = time.time() - llm_start_time
            logger.error(f"🧠 [{self.agent_name}] ❌ LLM call FAILED after {llm_duration_seconds:.3f}s: {e}")
            
            if obs_logger:
                obs_logger.record_llm_call(
                    component=f"agent.{self.agent_name}.think",
                    provider=llm_config.provider,
                    model=llm_config.model,
                    tokens=TokenUsage(),
                    duration_seconds=llm_duration_seconds,
                    system_prompt_length=0,
                    messages_count=1,
                    temperature=llm_config.temperature,
                    max_tokens=llm_config.max_tokens,
                    function_name="_think",
                    error=str(e)
                )
            
            logger.info(f"🧠 [{self.agent_name}] ════════════════════════════════════════════════════════")
            logger.info(f"🧠 [{self.agent_name}] EXITING _think() - ERROR")
            logger.info(f"🧠 [{self.agent_name}] ════════════════════════════════════════════════════════")
            
            # Re-raise the exception
            raise

    async def _execute_tool(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        execution_log: ExecutionLog
    ) -> Dict[str, Any]:
        """Execute a tool and return the result."""
        logger.info(f"🛠️  [{self.agent_name}] ════════════════════════════════════════════════════════")
        logger.info(f"🛠️  [{self.agent_name}] EXECUTING TOOL: {tool_name}")
        logger.info(f"🛠️  [{self.agent_name}] ════════════════════════════════════════════════════════")
        
        # Check if tool exists
        logger.info(f"🛠️  [{self.agent_name}] Step 1: Validating tool...")
        if not hasattr(self, '_tools') or tool_name not in self._tools:
            logger.error(f"🛠️  [{self.agent_name}]   ❌ Unknown tool: {tool_name}")
            available = list(self._tools.keys()) if hasattr(self, '_tools') and self._tools else []
            logger.error(f"🛠️  [{self.agent_name}]   → Available tools: {available}")
            
            error_result = {
                "success": False,
                "result_type": ToolResultType.FATAL.value,
                "error": f"Unknown tool: {tool_name}",
                "available_tools": available
            }
            logger.info(f"🛠️  [{self.agent_name}] ════════════════════════════════════════════════════════")
            return error_result
        
        logger.info(f"🛠️  [{self.agent_name}]   ✅ Tool found in registry")
        
        # =========================================================================
        # NEW: Check if we can skip this call using derived entities
        # =========================================================================
        session_id = tool_input.get('session_id')
        if session_id:
            skip_result = self._check_derived_entity_cache(session_id, tool_name, tool_input)
            if skip_result is not None:
                logger.info(f"[{self.agent_name}] Using cached derived entity for {tool_name}")
                # Log to execution log
                execution_log.tools_used.append(ToolExecution(
                    tool_name=tool_name,
                    inputs=tool_input,
                    outputs=skip_result,
                    success=True,
                    duration_seconds=0.0
                ))
                return skip_result
        
        # Log inputs
        logger.info(f"🛠️  [{self.agent_name}] Step 2: Tool inputs...")
        logger.info(f"🛠️  [{self.agent_name}]   → Input keys: {list(tool_input.keys())}")
        logger.info(f"🛠️  [{self.agent_name}]   → Full inputs:")
        for key, value in tool_input.items():
            value_str = str(value)
            if len(value_str) > 100:
                value_str = value_str[:100] + "..."
            logger.info(f"🛠️  [{self.agent_name}]       • {key}: {value_str}")
        logger.debug(f"🛠️  [{self.agent_name}]   → Full inputs (debug):\n{json.dumps(tool_input, indent=4, default=str)}")

        # Get tool method to inspect its signature
        tool_method = self._tools[tool_name]
        
        # Get actual function parameters from the tool's signature
        sig = inspect.signature(tool_method)
        valid_params = set(sig.parameters.keys())
        
        # Handle nested tool_input dict if present (merge with top-level, top-level takes precedence)
        merged_input = dict(tool_input)
        if 'tool_input' in merged_input and isinstance(merged_input['tool_input'], dict):
            nested = merged_input.pop('tool_input')
            # Merge nested dict into top-level (top-level values take precedence)
            merged_input = {**nested, **merged_input}
            logger.info(f"🛠️  [{self.agent_name}]   → Merged nested tool_input dict with top-level keys")
        
        # Filter input to only include keys that match the tool's actual parameters
        clean_tool_input = {k: v for k, v in merged_input.items() if k in valid_params}
        
        # Log what was filtered out
        filtered_keys = [k for k in merged_input.keys() if k not in valid_params]
        if filtered_keys:
            logger.info(f"🛠️  [{self.agent_name}]   → Filtered out unrelated inputs: {filtered_keys}")
            logger.info(f"🛠️  [{self.agent_name}]   → Valid tool parameters: {sorted(valid_params)}")
            logger.info(f"🛠️  [{self.agent_name}]   → Cleaned input keys: {sorted(clean_tool_input.keys())}")
        start_time = time.time()
        logger.info(f"🛠️  [{self.agent_name}] Step 3: Calling tool method at {datetime.utcnow().isoformat()}...")

        try:
            # Call tool (may be sync or async)
            import asyncio
            if asyncio.iscoroutinefunction(tool_method):
                logger.info(f"🛠️  [{self.agent_name}]   → Tool is async, awaiting...")
                result = await tool_method(**clean_tool_input)
            else:
                logger.info(f"🛠️  [{self.agent_name}]   → Tool is sync, calling directly...")
                result = tool_method(**clean_tool_input)
            
            duration_seconds = time.time() - start_time
            logger.info(f"🛠️  [{self.agent_name}]   ✅ Tool executed in {duration_seconds:.3f}s")
            
            # Ensure result is a dict
            logger.info(f"🛠️  [{self.agent_name}] Step 4: Processing result...")
            logger.info(f"🛠️  [{self.agent_name}]   → Result type: {type(result).__name__}")
            
            if not isinstance(result, dict):
                logger.info(f"🛠️  [{self.agent_name}]   → Converting non-dict result to dict...")
                result = {"success": True, "data": result}
            
            logger.info(f"🛠️  [{self.agent_name}]   → Result keys: {list(result.keys())}")
            logger.info(f"🛠️  [{self.agent_name}]   → Success: {result.get('success', 'N/A')}")
            
            # Ensure result_type is set
            if "result_type" not in result:
                logger.info(f"🛠️  [{self.agent_name}]   → No result_type set, inferring...")
                inferred = self._infer_result_type(result)
                result["result_type"] = inferred
                logger.info(f"🛠️  [{self.agent_name}]       • Inferred: {inferred}")
            else:
                logger.info(f"🛠️  [{self.agent_name}]   → Result type: {result['result_type']}")
            
            # Log result details
            logger.info(f"🛠️  [{self.agent_name}]   → Result summary:")
            result_preview = json.dumps(result, default=str)[:500]
            logger.info(f"🛠️  [{self.agent_name}]       {result_preview}{'...' if len(json.dumps(result, default=str)) > 500 else ''}")
            logger.debug(f"🛠️  [{self.agent_name}]   → Full result:\n{json.dumps(result, indent=4, default=str)}")
            
            # Log to execution log
            logger.info(f"🛠️  [{self.agent_name}] Step 5: Recording to execution log...")
            execution_log.tools_used.append(ToolExecution(
                tool_name=tool_name,
                inputs=tool_input,
                outputs=result,
                success=result.get("success", True),
                duration_seconds=duration_seconds
            ))
            logger.info(f"🛠️  [{self.agent_name}]   ✅ Recorded (total tools: {len(execution_log.tools_used)})")
            
            # =====================================================================
            # NEW: Store derived entities from tool result
            # =====================================================================
            session_id = tool_input.get('session_id')
            if session_id:
                self._store_derived_entities(session_id, tool_name, tool_input, result)
            
            logger.info(f"🛠️  [{self.agent_name}] ════════════════════════════════════════════════════════")
            logger.info(f"🛠️  [{self.agent_name}] TOOL EXECUTION COMPLETE: {tool_name} ✅")
            logger.info(f"🛠️  [{self.agent_name}] ════════════════════════════════════════════════════════")
            return result
        
        except Exception as e:
            duration_seconds = time.time() - start_time
            logger.error(f"🛠️  [{self.agent_name}]   ❌ Tool execution FAILED after {duration_seconds:.3f}s")
            logger.error(f"🛠️  [{self.agent_name}]   → Error type: {type(e).__name__}")
            logger.error(f"🛠️  [{self.agent_name}]   → Error message: {str(e)}")
            logger.error(f"🛠️  [{self.agent_name}]   → Stack trace:", exc_info=True)
            
            error_result = {
                "success": False,
                "result_type": ToolResultType.SYSTEM_ERROR.value,
                "error": str(e),
                "should_retry": True
            }
            
            logger.info(f"🛠️  [{self.agent_name}] Recording error to execution log...")
            execution_log.tools_used.append(ToolExecution(
                tool_name=tool_name,
                inputs=tool_input,
                outputs=error_result,
                success=False,
                error=str(e)
            ))
            
            logger.info(f"🛠️  [{self.agent_name}] ════════════════════════════════════════════════════════")
            logger.info(f"🛠️  [{self.agent_name}] TOOL EXECUTION FAILED: {tool_name} ❌")
            logger.info(f"🛠️  [{self.agent_name}] ════════════════════════════════════════════════════════")
            return error_result

    async def _process_tool_result(
        self,
        session_id: str,
        tool_name: str,
        tool_result: Dict[str, Any],
        exec_context: ExecutionContext
    ) -> Optional[AgentDecision]:
        """
        Process a tool result and determine if it changes our course.
        
        Returns:
            AgentDecision if we should override normal flow, None to continue
        """
        logger.info(f"🔧 [{self.agent_name}] ════════════════════════════════════════════════════════")
        logger.info(f"🔧 [{self.agent_name}] PROCESSING TOOL RESULT: {tool_name}")
        logger.info(f"🔧 [{self.agent_name}] ════════════════════════════════════════════════════════")
        
        # Step 1: Determine result type
        logger.info(f"🔧 [{self.agent_name}] Step 1: Determining result type...")
        result_type_str = tool_result.get("result_type", "partial")
        logger.info(f"🔧 [{self.agent_name}]   → Raw result_type: '{result_type_str}'")
        
        try:
            result_type = ToolResultType(result_type_str)
            logger.info(f"🔧 [{self.agent_name}]   ✅ Mapped to: {result_type.value}")
        except ValueError:
            logger.warning(f"🔧 [{self.agent_name}]   ⚠️  Unknown result_type '{result_type_str}', defaulting to PARTIAL")
            result_type = ToolResultType.PARTIAL
        
        # Log tool result details
        logger.info(f"🔧 [{self.agent_name}]   → Success: {tool_result.get('success', 'N/A')}")
        logger.info(f"🔧 [{self.agent_name}]   → Error: {tool_result.get('error', 'None')}")
        logger.info(f"🔧 [{self.agent_name}]   → Result keys: {list(tool_result.keys())}")
        logger.debug(f"🔧 [{self.agent_name}]   → Full result:\n{json.dumps(tool_result, indent=4, default=str)}")
        
        # Step 2: Add observation
        logger.info(f"🔧 [{self.agent_name}] Step 2: Adding observation to execution context...")
        exec_context.add_observation(
            obs_type="tool",
            name=tool_name,
            result=tool_result,
            result_type=result_type
        )
        logger.info(f"🔧 [{self.agent_name}]   ✅ Observation added (total: {len(exec_context.observations)})")
        
        # Step 2.5: Store derived entities
        self._store_derived_entities(session_id, tool_name, {}, tool_result)
        
        # Step 2.6: Update task status if task-based execution
        task_id = exec_context.task_plan.current_task_id if exec_context.task_plan else None
        if task_id and exec_context.task_plan:
            task = exec_context.task_plan.get_task(task_id)
            
            if task:
                if result_type == ToolResultType.SUCCESS:
                    exec_context.mark_task_complete(task_id, tool_result)
                    
                elif result_type == ToolResultType.USER_INPUT_NEEDED:
                    exec_context.mark_task_blocked(
                        task_id,
                        reason=tool_result.get("error", "User input needed"),
                        options=tool_result.get("alternatives", []),
                        suggested_response=tool_result.get("suggested_response")
                    )
                    
                elif result_type in [ToolResultType.FATAL, ToolResultType.SYSTEM_ERROR]:
                    if task.can_retry() and result_type == ToolResultType.SYSTEM_ERROR:
                        task.retry()
                        logger.info(f"Retrying task {task_id} (attempt {task.retry_count})")
                    else:
                        exec_context.mark_task_failed(task_id, tool_result.get("error", "Task failed"))
                        
                elif result_type == ToolResultType.PARTIAL:
                    # Partial success - task continues or completes based on next step
                    exec_context.mark_task_complete(task_id, tool_result)
                
                exec_context.task_plan.update_metrics()

        # Check if tool result override is enabled
        from patient_ai_service.core.feature_flags import is_tool_result_override_enabled
        if not is_tool_result_override_enabled(session_id):
            logger.info(f"🔧 [{self.agent_name}] Tool result override DISABLED - returning to LLM")
            logger.info(f"🔧 [{self.agent_name}] ════════════════════════════════════════════════════════")
            return None  # Let LLM see the result and decide

        # Step 3: Handle based on result type (OVERRIDE LOGIC)
        logger.info(f"🔧 [{self.agent_name}] Step 3: Handling result based on type: {result_type.value}")
        
        if result_type == ToolResultType.SUCCESS:
            logger.info(f"🔧 [{self.agent_name}]   → Result type: SUCCESS ✅")
            
            # Check if this satisfies any criteria
            satisfied = tool_result.get("satisfies_criteria", [])
            if satisfied:
                logger.info(f"🔧 [{self.agent_name}]   → Tool explicitly satisfies criteria: {satisfied}")
                for criterion in satisfied:
                    logger.info(f"🔧 [{self.agent_name}]       • Marking '{criterion}' as complete...")
                    exec_context.mark_criterion_complete(criterion, evidence=str(tool_result))
            else:
                logger.info(f"🔧 [{self.agent_name}]   → No explicit satisfies_criteria field")
            
            # Auto-detect criteria satisfaction
            if tool_result.get("appointment_id"):
                logger.info(f"🔧 [{self.agent_name}]   → Detected appointment_id: {tool_result['appointment_id']}")
                logger.info(f"🔧 [{self.agent_name}]   → Auto-detecting booking-related criteria...")
                # Look for booking-related criteria
                found_booking_criteria = False
                for crit in exec_context.criteria.values():
                    if "booked" in crit.description.lower() and crit.state == CriterionState.PENDING:
                        logger.info(f"🔧 [{self.agent_name}]       • Found booking criterion: '{crit.description}'")
                        logger.info(f"🔧 [{self.agent_name}]       • Marking as complete...")
                        exec_context.mark_criterion_complete(
                            crit.id,
                            evidence=f"appointment_id: {tool_result['appointment_id']}"
                        )
                        found_booking_criteria = True
                if not found_booking_criteria:
                    logger.info(f"🔧 [{self.agent_name}]       • No pending booking criteria found")
            
            logger.info(f"🔧 [{self.agent_name}]   → Decision: Continue normally (no override)")
            logger.info(f"🔧 [{self.agent_name}] ════════════════════════════════════════════════════════")
            return None  # Continue normally
        
        elif result_type == ToolResultType.PARTIAL:
            logger.info(f"🔧 [{self.agent_name}]   → Result type: PARTIAL ⏳")
            logger.info(f"🔧 [{self.agent_name}]   → Tool made progress but more steps needed")
            logger.info(f"🔧 [{self.agent_name}]   → Decision: Continue normally (no override)")
            logger.info(f"🔧 [{self.agent_name}] ════════════════════════════════════════════════════════")
            return None  # Continue normally
        
        elif result_type == ToolResultType.USER_INPUT_NEEDED:
            logger.info(f"🔧 [{self.agent_name}]   → Result type: USER_INPUT_NEEDED 🔄")
            logger.info(f"🔧 [{self.agent_name}]   → Tool requires user decision before proceeding")
            
            # Mark relevant criteria as blocked
            blocked_criteria = tool_result.get("blocks_criteria")
            alternatives = tool_result.get("alternatives", [])
            reason = tool_result.get("reason", "awaiting_user_input")
            
            logger.info(f"🔧 [{self.agent_name}]   → Blocked criteria: {blocked_criteria}")
            logger.info(f"🔧 [{self.agent_name}]   → Alternatives: {alternatives[:3]}{'...' if len(alternatives) > 3 else ''} ({len(alternatives)} total)")
            logger.info(f"🔧 [{self.agent_name}]   → Reason: {reason}")
            
            if blocked_criteria:
                logger.info(f"🔧 [{self.agent_name}]   → Marking specific criterion as blocked: '{blocked_criteria}'")
                exec_context.mark_criterion_blocked(blocked_criteria, reason, alternatives)
            else:
                logger.info(f"🔧 [{self.agent_name}]   → No specific blocked_criteria, auto-detecting...")
                # Block any pending booking criteria
                found_criteria = False
                for crit in exec_context.criteria.values():
                    if crit.state == CriterionState.PENDING and "booked" in crit.description.lower():
                        logger.info(f"🔧 [{self.agent_name}]       • Blocking criterion: '{crit.description}'")
                        exec_context.mark_criterion_blocked(crit.id, reason, alternatives)
                        found_criteria = True
                if not found_criteria:
                    logger.info(f"🔧 [{self.agent_name}]       • No pending booking criteria found to block")
            
            # Store for response generation
            logger.info(f"🔧 [{self.agent_name}]   → Storing user options and suggested response...")
            exec_context.pending_user_options = alternatives
            exec_context.suggested_response = tool_result.get("suggested_response")
            logger.info(f"🔧 [{self.agent_name}]       • Pending options: {len(alternatives)}")
            logger.info(f"🔧 [{self.agent_name}]       • Suggested response: {exec_context.suggested_response[:100] if exec_context.suggested_response else 'None'}...")
            
            # Store continuation context
            logger.info(f"🔧 [{self.agent_name}]   → Building continuation context...")
            continuation_data = {
                "awaiting": "user_selection",
                "options": alternatives,
                "original_request": tool_result.get("requested_time"),
                **{k: v for k, v in tool_result.items() if k in ["doctor_id", "date", "procedure"]}
            }
            exec_context.set_continuation_context(**continuation_data)
            logger.info(f"🔧 [{self.agent_name}]       • Context keys: {list(exec_context.continuation_context.keys())}")
            logger.debug(f"🔧 [{self.agent_name}]       • Full context:\n{json.dumps(exec_context.continuation_context, indent=4, default=str)}")
            
            # Persist for next turn
            logger.info(f"🔧 [{self.agent_name}]   → Persisting to state manager...")
            self.state_manager.update_agentic_state(
                session_id,
                status="blocked",
                continuation_context=exec_context.continuation_context
            )
            logger.info(f"🔧 [{self.agent_name}]       ✅ State persisted")
            
            logger.info(f"🔧 [{self.agent_name}]   → Decision: RESPOND_WITH_OPTIONS (override)")
            logger.info(f"🔧 [{self.agent_name}] ════════════════════════════════════════════════════════")
            return AgentDecision.RESPOND_WITH_OPTIONS
        
        elif result_type == ToolResultType.RECOVERABLE:
            logger.info(f"🔧 [{self.agent_name}]   → Result type: RECOVERABLE 🔧")
            recovery_action = tool_result.get("recovery_action")
            logger.info(f"🔧 [{self.agent_name}]   → Error can be recovered from")
            logger.info(f"🔧 [{self.agent_name}]   → Suggested recovery action: {recovery_action}")
            logger.info(f"🔧 [{self.agent_name}]   → Original error: {tool_result.get('error', 'N/A')}")
            
            logger.info(f"🔧 [{self.agent_name}]   → Adding recovery hint observation...")
            exec_context.add_observation(
                "recovery_hint", tool_name,
                {"suggested_action": recovery_action, "original_error": tool_result.get("error")}
            )
            
            logger.info(f"🔧 [{self.agent_name}]   → Decision: Continue (let agent decide recovery in next think)")
            logger.info(f"🔧 [{self.agent_name}] ════════════════════════════════════════════════════════")
            return None  # Let agent figure out recovery in next think
        
        elif result_type == ToolResultType.FATAL:
            logger.warning(f"🔧 [{self.agent_name}]   → Result type: FATAL ❌")
            logger.warning(f"🔧 [{self.agent_name}]   → Cannot recover from this error")
            error_msg = tool_result.get('error', 'Unknown fatal error')
            logger.warning(f"🔧 [{self.agent_name}]   → Error: {error_msg}")
            
            exec_context.fatal_error = tool_result
            logger.info(f"🔧 [{self.agent_name}]   → Fatal error stored in execution context")
            
            # Mark relevant criteria as failed
            logger.info(f"🔧 [{self.agent_name}]   → Marking all pending/in-progress criteria as failed...")
            failed_count = 0
            for crit in exec_context.criteria.values():
                if crit.state in [CriterionState.PENDING, CriterionState.IN_PROGRESS]:
                    logger.info(f"🔧 [{self.agent_name}]       • Failing criterion: '{crit.description}'")
                    exec_context.mark_criterion_failed(crit.id, error_msg)
                    failed_count += 1
            logger.info(f"🔧 [{self.agent_name}]       → Total criteria failed: {failed_count}")
            
            logger.info(f"🔧 [{self.agent_name}]   → Decision: RESPOND_IMPOSSIBLE (override)")
            logger.info(f"🔧 [{self.agent_name}] ════════════════════════════════════════════════════════")
            return AgentDecision.RESPOND_IMPOSSIBLE
        
        elif result_type == ToolResultType.SYSTEM_ERROR:
            logger.warning(f"🔧 [{self.agent_name}]   → Result type: SYSTEM_ERROR 🚫")
            
            # Check if recovery_action is provided
            recovery_action = tool_result.get("recovery_action")
            recovery_message = tool_result.get("recovery_message")
            error_msg = tool_result.get("error", "")
            
            logger.warning(f"🔧 [{self.agent_name}]   → System/infrastructure error occurred")
            logger.warning(f"🔧 [{self.agent_name}]   → Error: {error_msg}")
            logger.info(f"🔧 [{self.agent_name}]   → Recovery action provided: {recovery_action or 'None'}")
            logger.info(f"🔧 [{self.agent_name}]   → Recovery message: {recovery_message or 'None'}")
            
            # Create error signature for tracking
            logger.info(f"🔧 [{self.agent_name}]   → Creating error signature for tracking...")
            error_signature = self._create_error_signature(tool_name, error_msg)
            logger.info(f"🔧 [{self.agent_name}]       • Signature: {error_signature[:80]}...")
            
            # Check if we've already tried recovery for this error
            logger.info(f"🔧 [{self.agent_name}]   → Checking recovery attempts history...")
            recovery_info = exec_context.recovery_attempts.get(error_signature)
            has_tried_recovery = recovery_info and recovery_info.get("attempted", False)
            logger.info(f"🔧 [{self.agent_name}]       • Previous attempts: {len(exec_context.recovery_attempts)}")
            logger.info(f"🔧 [{self.agent_name}]       • This error attempted: {has_tried_recovery}")
            
            if recovery_action and not has_tried_recovery:
                # First time seeing this error with recovery_action - try recovery
                logger.info(f"🔧 [{self.agent_name}]   → FIRST ATTEMPT - will try recovery")
                logger.info(f"🔧 [{self.agent_name}]   → Recording recovery attempt...")
                
                exec_context.recovery_attempts[error_signature] = {
                    "attempted": True,
                    "recovery_action": recovery_action,
                    "recovery_message": recovery_message,
                    "original_tool": tool_name,
                    "original_error": error_msg,
                    "iteration": exec_context.iteration
                }
                exec_context.recovery_executed = True
                
                logger.info(f"🔧 [{self.agent_name}]       ✅ Recovery attempt recorded")
                logger.info(f"🔧 [{self.agent_name}]       • Recovery action: {recovery_action}")
                logger.info(f"🔧 [{self.agent_name}]       • At iteration: {exec_context.iteration}")
                logger.info(f"🔧 [{self.agent_name}]   → Decision: EXECUTE_RECOVERY (override)")
                logger.info(f"🔧 [{self.agent_name}] ════════════════════════════════════════════════════════")
                return AgentDecision.EXECUTE_RECOVERY
            
            elif has_tried_recovery:
                # Same error after recovery attempt - give up immediately
                logger.error(f"🔧 [{self.agent_name}]   → ERROR PERSISTED after recovery!")
                logger.error(f"🔧 [{self.agent_name}]   → Previous recovery: {recovery_info.get('recovery_action')}")
                logger.error(f"🔧 [{self.agent_name}]   → Previous attempt at iteration: {recovery_info.get('iteration')}")
                logger.error(f"🔧 [{self.agent_name}]   → Giving up - marking as fatal")
                
                exec_context.fatal_error = tool_result
                logger.info(f"🔧 [{self.agent_name}]   → Decision: RESPOND_IMPOSSIBLE (override)")
                logger.info(f"🔧 [{self.agent_name}] ════════════════════════════════════════════════════════")
                return AgentDecision.RESPOND_IMPOSSIBLE
            
            elif exec_context.retry_count < exec_context.max_retries:
                # No recovery_action, use standard retry logic
                logger.info(f"🔧 [{self.agent_name}]   → NO recovery_action provided")
                logger.info(f"🔧 [{self.agent_name}]   → Using standard retry logic")
                logger.info(f"🔧 [{self.agent_name}]   → Retry count: {exec_context.retry_count + 1}/{exec_context.max_retries}")
                logger.info(f"🔧 [{self.agent_name}]   → Decision: RETRY (override)")
                logger.info(f"🔧 [{self.agent_name}] ════════════════════════════════════════════════════════")
                return AgentDecision.RETRY
            else:
                # Max retries exceeded
                logger.error(f"🔧 [{self.agent_name}]   → MAX RETRIES EXCEEDED")
                logger.error(f"🔧 [{self.agent_name}]   → Retry count: {exec_context.retry_count}/{exec_context.max_retries}")
                logger.error(f"🔧 [{self.agent_name}]   → Giving up - marking as fatal")
                
                exec_context.fatal_error = tool_result
                logger.info(f"🔧 [{self.agent_name}]   → Decision: RESPOND_IMPOSSIBLE (override)")
                logger.info(f"🔧 [{self.agent_name}] ════════════════════════════════════════════════════════")
                return AgentDecision.RESPOND_IMPOSSIBLE
        
        logger.info(f"🔧 [{self.agent_name}]   → No override decision needed")
        logger.info(f"🔧 [{self.agent_name}] ════════════════════════════════════════════════════════")
        return None

    async def _handle_override(
        self,
        session_id: str,
        override: AgentDecision,
        exec_context: ExecutionContext
    ) -> Optional[str]:
        """
        Handle an override decision from result processing.

        Routes through _generate_response for proper LLM-based response generation
        with full context (including suggestions, alternatives, etc.).
        """

        if override == AgentDecision.RESPOND_WITH_OPTIONS:
            # Build AgentResponseData from exec_context
            response_data = self._build_options_response_data(exec_context)
            return await self._generate_response(
                session_id,
                AgentDecision.RESPOND_WITH_OPTIONS,
                response_data,
                exec_context
            )

        elif override == AgentDecision.RESPOND_IMPOSSIBLE:
            # Build AgentResponseData from fatal_error
            response_data = self._build_failure_response_data(exec_context)
            return await self._generate_response(
                session_id,
                AgentDecision.RESPOND_IMPOSSIBLE,
                response_data,
                exec_context
            )

        elif override == AgentDecision.RETRY:
            return None  # Continue loop

        elif override == AgentDecision.EXECUTE_RECOVERY:
            return None  # Continue loop (handled in main loop)

        return None

    def _build_options_response_data(self, exec_context: ExecutionContext) -> AgentResponseData:
        """Build AgentResponseData for RESPOND_WITH_OPTIONS from exec_context."""
        options = []
        options_reason = None
        options_context = None

        # Check for pending user options (from USER_INPUT result)
        if exec_context.pending_user_options:
            options = exec_context.pending_user_options

        # Check blocked criteria for additional context
        blocked = exec_context.get_blocked_criteria()
        if blocked and blocked[0].blocked_options:
            if not options:
                options = blocked[0].blocked_options
            options_reason = blocked[0].blocked_reason

        # Get options_context from continuation context
        if exec_context.continuation_context:
            options_context = exec_context.continuation_context.get("awaiting", "user_selection")

        return AgentResponseData(
            options=options,
            options_reason=options_reason,
            options_context=options_context
        )

    def _build_failure_response_data(self, exec_context: ExecutionContext) -> AgentResponseData:
        """Build AgentResponseData for RESPOND_IMPOSSIBLE from exec_context."""
        failure_reason = None
        failure_suggestion = None

        if exec_context.fatal_error:
            error = exec_context.fatal_error
            # Extract error message
            failure_reason = (
                error.get("message") or
                error.get("error_message") or
                error.get("error", "Unable to complete request")
            )
            # Extract suggestion - check multiple fields
            failure_suggestion = (
                error.get("suggestion") or  # "Did you mean one of these?..."
                error.get("failure_suggestion") or
                (f"Available alternatives: {', '.join(error['alternatives'][:3])}"
                 if error.get("alternatives") else None)
            )
        else:
            # Check failed criteria
            failed = [c for c in exec_context.criteria.values() if c.state == CriterionState.FAILED]
            if failed:
                reasons = [c.failed_reason for c in failed if c.failed_reason]
                if reasons:
                    failure_reason = reasons[0]

        if not failure_reason:
            failure_reason = "Unable to complete your request"

        return AgentResponseData(
            failure_reason=failure_reason,
            failure_suggestion=failure_suggestion
        )

    def _generate_options_response(self, exec_context: ExecutionContext) -> str:
        """Generate response presenting options to user."""
        
        # Use suggested response if available
        if exec_context.suggested_response:
            return exec_context.suggested_response
        
        # Build from blocked criteria
        blocked = exec_context.get_blocked_criteria()
        if blocked and blocked[0].blocked_options:
            options = blocked[0].blocked_options
            reason = blocked[0].blocked_reason or "that option isn't available"
            
            # Format options nicely
            if len(options) <= 3:
                options_str = ", ".join(str(o) for o in options[:-1]) + f" or {options[-1]}"
            else:
                options_str = ", ".join(str(o) for o in options[:3]) + f" (and {len(options)-3} more)"
            
            return f"I'm sorry, {reason}. Would {options_str} work instead?"
        
        # Generic fallback
        if exec_context.pending_user_options:
            options = exec_context.pending_user_options
            return f"I have a few options available: {', '.join(str(o) for o in options[:5])}. Which would you prefer?"
        
        return "I need some additional information to proceed. Could you please clarify your preference?"

    def _generate_failure_response(self, exec_context: ExecutionContext) -> str:
        """Generate response explaining failure."""
        
        if exec_context.fatal_error:
            error = exec_context.fatal_error
            message = error.get("message") or error.get("error_message") or error.get("error", "Unable to complete request")
            
            # Check for alternatives
            if error.get("alternatives"):
                return f"{message} However, I can help with: {', '.join(error['alternatives'][:3])}"
            
            return f"I'm sorry, {message}. Is there something else I can help you with?"
        
        # Check failed criteria
        failed = [c for c in exec_context.criteria.values() if c.state == CriterionState.FAILED]
        if failed:
            reasons = [c.failed_reason for c in failed if c.failed_reason]
            if reasons:
                return f"I wasn't able to complete your request: {reasons[0]}. Would you like to try something else?"
        
        return "I encountered an issue and couldn't complete your request. Would you like to try again or try something different?"

    def _generate_max_iterations_response(self, exec_context: ExecutionContext) -> str:
        """Generate response when max iterations reached."""
        
        completion = exec_context.check_completion()
        
        parts = []
        
        if completion.completed_criteria:
            parts.append(f"I was able to complete: {', '.join(completion.completed_criteria)}")
        
        if completion.pending_criteria:
            parts.append(f"Still pending: {', '.join(completion.pending_criteria)}")
        
        if completion.blocked_criteria:
            parts.append("I need your input to continue with some items.")
        
        if parts:
            return " ".join(parts) + " Would you like me to continue?"
        
        return "I'm still working on your request. Could you please provide more details or try a simpler request?"

    async def _generate_response(
        self,
        session_id: str,
        decision: AgentDecision,
        response_data: AgentResponseData,
        exec_context: ExecutionContext
    ) -> str:
        """
        Unified response generator for all decision types.
        
        Takes structured response data and generates appropriate user-facing message.
        """
        logger.info(f"🎯 [{self.agent_name}] Generating response for decision: {decision.value}")
        logger.info(f"🎯 [{self.agent_name}]   Response data: {response_data.dict(exclude_none=True)}")
        
        context = self._context.get(session_id, {})
        language = context.get("current_language", "en")
        dialect = context.get("current_dialect")
        tone = context.get("tone", "helpful")
        
        # Get patient context properly
        patient_context = PatientContext.from_session(
            session_id, 
            self.state_manager,
            entities=response_data.entities
        )
        
        logger.info(f"🎯 [{self.agent_name}] Patient context: {patient_context.get_prompt_context()}")
        
        # Format resolved entities
        known_info = self._format_entities(response_data.entities)
        
        # Get recent conversation history from NativeLanguageMemory
        # This provides original language context for better tone matching in responses
        from patient_ai_service.core.native_language_memory import get_native_language_memory_manager
        native_memory_manager = get_native_language_memory_manager()
        recent_turns_data = native_memory_manager.get_recent_turns(session_id, limit=4)
        logger.info(f"🔍 [_generate_response] Retrieved {len(recent_turns_data)} turns from NativeLanguageMemory")
        if recent_turns_data:
            logger.info(f"🔍 [_generate_response] Language: {recent_turns_data[0].get('language', 'unknown')}")

        # Exclude current user message if it's the last one
        if recent_turns_data and recent_turns_data[-1]["role"] == "user":
            recent_turns_data = recent_turns_data[:-1]

        # Get last 3 messages
        recent_turns_data = recent_turns_data[-3:] if len(recent_turns_data) > 3 else recent_turns_data

        # Format recent messages for inclusion in system prompt
        if recent_turns_data:
            recent_context = "\n".join([
                f"{'User' if t['role'] == 'user' else 'Assistant'}: {t['content'][:200]}"
                for t in recent_turns_data
            ])
        else:
            recent_context = "(No previous messages)"
        
        # Build decision-specific prompt
        if decision == AgentDecision.COLLECT_INFORMATION:
            prompt = self._build_collect_info_prompt(response_data, known_info, patient_context, tone, language, dialect, recent_context)
            
        elif decision == AgentDecision.CLARIFY:
            prompt = self._build_clarify_prompt(response_data, known_info, patient_context, tone, language, dialect, recent_context)
            
        elif decision == AgentDecision.RESPOND_WITH_OPTIONS:
            prompt = self._build_options_prompt(response_data, known_info, patient_context, tone, language, dialect, recent_context)
            
        elif decision == AgentDecision.RESPOND_IMPOSSIBLE:
            prompt = self._build_impossible_prompt(response_data, known_info, patient_context, tone, language, dialect, recent_context)
            
        elif decision in [AgentDecision.RESPOND, AgentDecision.RESPOND_COMPLETE]:
            prompt = self._build_completion_prompt(response_data, exec_context, patient_context, tone, language, dialect, recent_context)
            
        elif decision == AgentDecision.REQUEST_CONFIRMATION:
            prompt = self._build_confirmation_prompt(response_data, known_info, patient_context, tone, language, dialect, recent_context)
            
        else:
            # Fallback
            prompt = self._build_generic_prompt(response_data, exec_context, patient_context, tone, language, dialect, recent_context)
        
        # Log all inputs and built prompt
        logger.info("=" * 80)
        logger.info(f"💬 [{self.agent_name}] UNIFIED RESPONSE GENERATOR - INPUTS:")
        logger.info("=" * 80)
        inputs = {
            "decision": decision.value,
            "response_data": response_data.dict(exclude_none=True),
            "session_id": session_id,
            "language": language,
            "dialect": dialect,
            "tone": tone,
            "patient_context": patient_context.get_prompt_context(),
            "known_info": known_info,
            "exec_context_summary": {
                "iteration": exec_context.iteration,
                "tool_calls": exec_context.tool_calls,
                "llm_calls": exec_context.llm_calls,
                "observations_count": len(exec_context.observations),
                "criteria_count": len(exec_context.criteria),
            }
        }
        logger.info(f"💬 [{self.agent_name}] Inputs JSON:\n{json.dumps(inputs, indent=2, default=str)}")
        logger.info("=" * 80)
        logger.info(f"💬 [{self.agent_name}] BUILT PROMPT:")
        logger.info("=" * 80)
        logger.info(f"💬 [{self.agent_name}] System Prompt:\n{prompt}")
        logger.info("=" * 80)
        
        # Generate response
        try:
            obs_logger = get_observability_logger(session_id) if settings.enable_observability else None
            llm_start_time = time.time()
            
            # Get hierarchical LLM config for _generate_focused_response function
            # (This method serves as the focused response generator)
            llm_config = self._get_llm_config_for_function("_generate_focused_response")
            response_temperature = llm_config.temperature
            llm_client = self._get_llm_client_for_function("_generate_focused_response")
            
            # Messages array is now just the instruction - conversation history is in system prompt
            messages = [{"role": "user", "content": "Generate the response."}]
            
            logger.info(f"💬 [{self.agent_name}] Recent conversation context included in system prompt: {len(recent_turns_data)} messages")

            if hasattr(llm_client, 'create_message_with_usage'):
                response, tokens = llm_client.create_message_with_usage(
                    system=prompt,
                    messages=messages,
                    temperature=response_temperature,
                    max_tokens=llm_config.max_tokens
                )
            else:
                response = llm_client.create_message(
                    system=prompt,
                    messages=messages,
                    temperature=response_temperature,
                    max_tokens=llm_config.max_tokens
                )
                tokens = TokenUsage()
            
            llm_duration_seconds = time.time() - llm_start_time
            
            # Log raw LLM response
            logger.info("=" * 80)
            logger.info(f"💬 [{self.agent_name}] RAW LLM RESPONSE:")
            logger.info("=" * 80)
            logger.info(f"💬 [{self.agent_name}] Response Content:\n{response}")
            logger.info("=" * 80)
            
            # Record LLM call
            if obs_logger:
                try:
                    obs_logger.record_llm_call(
                        component=f"agent.{self.agent_name}.generate_response",
                        provider=llm_config.provider,
                        model=llm_config.model,
                        tokens=tokens,
                        duration_seconds=llm_duration_seconds,
                        system_prompt_length=len(prompt),
                        messages_count=1,
                        temperature=response_temperature,
                        max_tokens=llm_config.max_tokens,
                        function_name="_generate_focused_response",
                        error=None
                    )
                except Exception as obs_error:
                    logger.warning(f"🎯 [{self.agent_name}] Error in observability logging (non-critical): {obs_error}")
            
            # Clean response to remove any formatting, analysis, or extra content
            cleaned_response = self._clean_response_text(response)
            
            logger.info(f"🎯 [{self.agent_name}] Generated: {cleaned_response}")
            return cleaned_response
            
        except Exception as e:
            logger.error(f"🎯 [{self.agent_name}] Error generating response: {e}")
            return self._get_fallback_response(decision, response_data)
    
    def _clean_response_text(self, text: str) -> str:
        """Clean response text to remove formatting, analysis, and extra content."""
        import re
        
        # Remove markdown headers
        text = re.sub(r'^#+\s+.*$', '', text, flags=re.MULTILINE)
        
        # Remove "Generated Response" or similar headers
        text = re.sub(r'^.*Generated Response.*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
        
        # Remove sections like "**Arabic (UAE Dialect):**" and keep only the content after it
        if "**Arabic" in text or "Arabic (UAE Dialect):" in text:
            # Extract text after Arabic section marker
            match = re.search(r'(?:Arabic.*?:|Arabic \(.*?\):)\s*\n\s*(.+?)(?:\n\s*---|\n\s*\*\*|$)', text, re.DOTALL | re.IGNORECASE)
            if match:
                text = match.group(1).strip()
        
        # Remove "English Translation:" section and everything after it
        if "English Translation:" in text or "**English Translation:**" in text:
            text = re.split(r'(?:English Translation|English translation):', text, flags=re.IGNORECASE)[0].strip()
        
        # Remove "Analysis:" section and everything after it
        if "Analysis:" in text or "**Analysis:**" in text:
            text = re.split(r'Analysis:', text, flags=re.IGNORECASE)[0].strip()
        
        # Remove markdown formatting (**, ---, etc.)
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Remove bold
        text = re.sub(r'^---+\s*$', '', text, flags=re.MULTILINE)  # Remove horizontal rules
        text = re.sub(r'^✅\s*', '', text, flags=re.MULTILINE)  # Remove checkmarks
        
        # Remove quoted text markers
        text = text.strip('"').strip("'")
        
        # Clean up extra whitespace
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple newlines to double
        text = text.strip()
        
        return text

    def _get_user_intent(self, session_id: str, exec_context: Optional[ExecutionContext] = None) -> Optional[str]:
        """
        Get user intent from execution context or agent context.
        
        Args:
            session_id: Session identifier
            exec_context: Optional execution context (preferred source)
        
        Returns:
            User intent string or None
        """
        # Try exec_context first (most reliable)
        if exec_context and exec_context.task_context:
            user_intent = exec_context.task_context.get("user_intent")
            if user_intent:
                return user_intent
        
        # Fallback to agent context
        context = self._context.get(session_id, {})
        return context.get("what_user_means") or context.get("user_intent")

    async def _humanize_response(
        self,
        session_id: str,
        suggested_response: str,
        user_intent: Optional[str] = None
    ) -> str:
        """
        Humanize response using conversation context. ~200 tokens input.

        Takes suggested_response from _think and makes it conversational,
        matching the patient's language and style from recent messages.
        
        Args:
            session_id: Session identifier
            suggested_response: The suggested response to humanize
            user_intent: Optional user intent from unified reasoning (what_user_means)
        """
        logger.info(f"🎨 [{self.agent_name}] Humanizing response...")
        logger.info(f"🎨 [{self.agent_name}]   Input: {suggested_response[:100]}...")
        if user_intent:
            logger.info(f"🎨 [{self.agent_name}]   User intent: {user_intent[:100]}...")

        # Get recent messages from NativeLanguageMemory
        # This provides original language context for better tone/style matching
        from patient_ai_service.core.native_language_memory import get_native_language_memory_manager
        native_memory_manager = get_native_language_memory_manager()
        recent_turns_data = native_memory_manager.get_recent_turns(session_id, limit=4)
        logger.info(f"🔍 [_humanize_response] Retrieved {len(recent_turns_data)} turns from NativeLanguageMemory")
        if recent_turns_data:
            logger.info(f"🔍 [_humanize_response] Language: {recent_turns_data[0].get('language', 'unknown')}")

        # Build conversation context (last 4 messages)
        lines = []
        for turn_data in recent_turns_data:
            role = "patient" if turn_data["role"] == "user" else "you"
            timestamp = turn_data.get("timestamp")
            time_ago = format_time_ago(timestamp) if timestamp else None
            time_str = f" • {time_ago}" if time_ago else ""
            content = turn_data["content"][:150] if turn_data.get("content") else ""
            lines.append(f"[{role}{time_str}] {content}")

        conversation_context = "\n".join(lines) if lines else "[No previous messages]"
        patient_context = PatientContext.from_session(session_id, self.state_manager)
        patient_name = patient_context.name or "we don't know yet"

        # Build user prompt with optional user intent
        user_intent_section = ""
        if user_intent:
            user_intent_section = f"\nUSER INTENT: {user_intent}\n"

        user_prompt = f"""
Patient name: {patient_name}

{user_intent_section}

CONVERSATION CONTEXT:
{conversation_context}

SUGGESTED RESPONSE: {suggested_response}

OUTPUT FORMAT:
Provide ONLY the response text.

Response:"""

        logger.info(f"🎨 [{self.agent_name}] Humanizer prompt:\n{user_prompt}")

        # Call lightweight LLM
        llm_config = self._get_llm_config_for_function("_humanize_response")
        llm_client = self._get_llm_client_for_function("_humanize_response")

        try:
            llm_start_time = time.time()

            if hasattr(llm_client, 'create_message_with_usage'):
                response, tokens = llm_client.create_message_with_usage(
                    system=HUMANIZE_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_prompt}],
                    temperature=llm_config.temperature,
                    max_tokens=100
                )
            else:
                response = llm_client.create_message(
                    system=HUMANIZE_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_prompt}],
                    temperature=llm_config.temperature,
                    max_tokens=100
                )
                tokens = TokenUsage()

            llm_duration = time.time() - llm_start_time
            result = response.strip()

            logger.info(f"🎨 [{self.agent_name}] Humanized ({llm_duration:.2f}s): {result}")
            return result

        except Exception as e:
            logger.warning(f"🎨 [{self.agent_name}] Humanizer failed, using suggested_response: {e}")
            return suggested_response  # Fallback to original

    def _build_collect_info_prompt(
        self,
        response_data: AgentResponseData,
        known_info: str,
        patient_context: PatientContext,
        tone: str,
        language: str,
        dialect: Optional[str],
        recent_context: str = ""
    ) -> str:
        """Build prompt for COLLECT_INFORMATION decision."""
        lang_instruction = ""
        if language == "ar":
            # IMPORTANT: Always use Emirati Arabic for Arabic responses
            dialect = dialect or "ae"  # Force Emirati if dialect is None
            dialect_map = {"ae": "Emirati Arabic (UAE)", "sa": "Saudi Arabic", "eg": "Egyptian Arabic", "lv": "Levantine Arabic"}
            dialect_name = dialect_map.get(dialect, "Emirati Arabic (UAE)")  # Default to Emirati
            lang_instruction = f"Respond in {dialect_name}. Use natural, conversational Arabic."
        elif language == "en":
            lang_instruction = "Respond in English."
        else:
            lang_instruction = f"Respond in {language}."
        
        return f"""Generate a {tone} question to collect missing information.

PATIENT CONTEXT:
{patient_context.get_prompt_context()}

⚠️  CRITICAL: If patient name is not available, DO NOT invent one!

WHAT WE ALREADY KNOW:
{known_info}

WHAT WE NEED:
{response_data.information_needed}

RECENT CONVERSATION:
{recent_context}

RULES:
- Ask ONLY for the missing information listed in "WHAT WE NEED"
- Be conversational and natural
- DO NOT mention appointments, booking, scheduling, or any actions
- DO NOT say "I'll check", "I'll do X", "I'll help you", "I'd be happy to help you schedule/book/do X" - nothing has been done yet!
- DO NOT promise any action or future action
- DO NOT reference what the user might want to do later (appointments, booking, etc.)
- DO NOT invent patient names - use only what's in PATIENT CONTEXT
- Keep it to 1-2 sentences
- Simply ask for the information needed, nothing more
- {lang_instruction}

OUTPUT FORMAT:
Provide ONLY the response text.

Response:"""

    def _build_clarify_prompt(
        self,
        response_data: AgentResponseData,
        known_info: str,
        patient_context: PatientContext,
        tone: str,
        language: str,
        dialect: Optional[str],
        recent_context: str = ""
    ) -> str:
        """Build prompt for CLARIFY decision."""
        lang_instruction = ""
        if language == "ar":
            # IMPORTANT: Always use Emirati Arabic for Arabic responses
            dialect = dialect or "ae"  # Force Emirati if dialect is None
            dialect_map = {"ae": "Emirati Arabic (UAE)", "sa": "Saudi Arabic", "eg": "Egyptian Arabic", "lv": "Levantine Arabic"}
            dialect_name = dialect_map.get(dialect, "Emirati Arabic (UAE)")  # Default to Emirati
            lang_instruction = f"Respond in {dialect_name}. Use natural, conversational Arabic."
        elif language == "en":
            lang_instruction = "Respond in English."
        else:
            lang_instruction = f"Respond in {language}."
        
        return f"""Generate a {tone} question to clarify ambiguous input.

PATIENT CONTEXT:
{patient_context.get_prompt_context()}

⚠️  CRITICAL: If patient name is not available, DO NOT invent one!

WHAT WE ALREADY KNOW:
{known_info}

WHAT'S UNCLEAR:
{response_data.clarification_needed or "Something needs clarification"}

SUGGESTED QUESTION:
{response_data.clarification_question or "Could you please clarify?"}

RECENT CONVERSATION:
{recent_context}

RULES:
- Ask a clarifying question
- Be conversational and natural, but avoid redundancy - don't repeat what was already said
- DO NOT invent patient names
- Keep it to 1-2 sentences
- {lang_instruction}

OUTPUT FORMAT:
Provide ONLY the response text.

Response:"""

    def _build_options_prompt(
        self,
        response_data: AgentResponseData,
        known_info: str,
        patient_context: PatientContext,
        tone: str,
        language: str,
        dialect: Optional[str],
        recent_context: str = ""
    ) -> str:
        """Build prompt for RESPOND_WITH_OPTIONS decision."""
        options_formatted = "\n".join([f"  • {opt}" for opt in response_data.options])
        
        lang_instruction = ""
        if language == "ar":
            # IMPORTANT: Always use Emirati Arabic for Arabic responses
            dialect = dialect or "ae"  # Force Emirati if dialect is None
            dialect_map = {"ae": "Emirati Arabic (UAE)", "sa": "Saudi Arabic", "eg": "Egyptian Arabic", "lv": "Levantine Arabic"}
            dialect_name = dialect_map.get(dialect, "Emirati Arabic (UAE)")  # Default to Emirati
            lang_instruction = f"Respond in {dialect_name}. Use natural, conversational Arabic."
        elif language == "en":
            lang_instruction = "Respond in English."
        else:
            lang_instruction = f"Respond in {language}."
        
        return f"""Generate a {tone} response presenting options to the user.

═══════════════════════════════════════════════════════════════
PATIENT CONTEXT
═══════════════════════════════════════════════════════════════
{patient_context.get_prompt_context()}

CONTEXT:
{known_info}

WHY OPTIONS ARE NEEDED:
{response_data.options_reason or "Original request couldn't be fulfilled exactly"}

OPTIONS TO PRESENT ({response_data.options_context}):
{options_formatted}

RECENT CONVERSATION:
{recent_context}

RULES:
- Briefly explain why we're offering alternatives
- Present the options clearly
- Ask which they prefer
- Keep it concise (2-4 sentences)
- Don't apologize excessively
- Be conversational and natural, but avoid redundancy - don't repeat what was already said
- DO NOT invent patient names
- {lang_instruction}

OUTPUT FORMAT:
Provide ONLY the response text.

Response:"""

    def _build_impossible_prompt(
        self,
        response_data: AgentResponseData,
        known_info: str,
        patient_context: PatientContext,
        tone: str,
        language: str,
        dialect: Optional[str],
        recent_context: str = ""
    ) -> str:
        """Build prompt for RESPOND_IMPOSSIBLE decision."""
        lang_instruction = ""
        if language == "ar":
            # IMPORTANT: Always use Emirati Arabic for Arabic responses
            dialect = dialect or "ae"  # Force Emirati if dialect is None
            dialect_map = {"ae": "Emirati Arabic (UAE)", "sa": "Saudi Arabic", "eg": "Egyptian Arabic", "lv": "Levantine Arabic"}
            dialect_name = dialect_map.get(dialect, "Emirati Arabic (UAE)")  # Default to Emirati
            lang_instruction = f"Respond in {dialect_name}. Use natural, conversational Arabic."
        elif language == "en":
            lang_instruction = "Respond in English."
        else:
            lang_instruction = f"Respond in {language}."
        
        return f"""Generate a {tone} response explaining why the request cannot be fulfilled.

PATIENT CONTEXT:
{patient_context.get_prompt_context()}

WHAT WAS ATTEMPTED:
{known_info}

WHY IT'S NOT POSSIBLE:
{response_data.failure_reason}

SUGGESTION FOR USER:
{response_data.failure_suggestion or "No specific suggestion"}

RECENT CONVERSATION:
{recent_context}

RULES:
- Be empathetic but direct
- Explain the issue clearly
- Offer the suggestion if available
- Don't over-apologize
- Keep it to 2-3 sentences
- Be conversational and natural, but avoid redundancy - don't repeat what was already said
- DO NOT invent patient names
- {lang_instruction}

OUTPUT FORMAT:
Provide ONLY the response text.

Response:"""

    def _build_completion_prompt(
        self,
        response_data: AgentResponseData,
        exec_context: ExecutionContext,
        patient_context: PatientContext,
        tone: str,
        language: str,
        dialect: Optional[str],
        recent_context: str = ""
    ) -> str:
        """Build prompt for RESPOND_COMPLETE decision."""
        # Get execution summary from actual tool results
        execution_summary = self._build_execution_summary(exec_context)
        
        details_formatted = "\n".join([
            f"  • {k}: {v}" for k, v in response_data.completion_details.items() if v
        ]) if response_data.completion_details else "No details"
        
        # Determine how to address patient
        if patient_context.name:
            address_instruction = f"You may address the patient as '{patient_context.name}'"
        else:
            address_instruction = "DO NOT use any name - patient name is not available"
        
        lang_instruction = ""
        if language == "ar":
            # IMPORTANT: Always use Emirati Arabic for Arabic responses
            dialect = dialect or "ae"  # Force Emirati if dialect is None
            dialect_map = {"ae": "Emirati Arabic (UAE)", "sa": "Saudi Arabic", "eg": "Egyptian Arabic", "lv": "Levantine Arabic"}
            dialect_name = dialect_map.get(dialect, "Emirati Arabic (UAE)")  # Default to Emirati
            lang_instruction = f"Respond in {dialect_name}. Use natural, conversational Arabic."
        elif language == "en":
            lang_instruction = "Respond in English."
        else:
            lang_instruction = f"Respond in {language}."
        
        return f"""Generate a {tone} response confirming what was accomplished.

PATIENT CONTEXT:
{patient_context.get_prompt_context()}
{address_instruction}

WHAT WAS DONE:
{response_data.completion_summary or "Task completed"}

DETAILS:
{details_formatted}

EXECUTION LOG:
{execution_summary}

RECENT CONVERSATION:
{recent_context}

RULES:
- Confirm the action was completed
- Include key details (confirmation number, date, time, etc.)
- Be warm but concise
- No technical details or UUIDs
- Be conversational and natural, but avoid redundancy - don't repeat what was already said
- DO NOT invent patient names
- 2-3 sentences
- {lang_instruction}

OUTPUT FORMAT:
Provide ONLY the response text.

Response:"""

    def _build_confirmation_prompt(
        self,
        response_data: AgentResponseData,
        known_info: str,
        patient_context: PatientContext,
        tone: str,
        language: str,
        dialect: Optional[str],
        recent_context: str = ""
    ) -> str:
        """Build prompt for REQUEST_CONFIRMATION decision."""
        details_formatted = "\n".join([
            f"  • {k}: {v}" for k, v in response_data.confirmation_details.items() if v
        ]) if response_data.confirmation_details else "No details"
        
        lang_instruction = ""
        if language == "ar":
            # IMPORTANT: Always use Emirati Arabic for Arabic responses
            dialect = dialect or "ae"  # Force Emirati if dialect is None
            dialect_map = {"ae": "Emirati Arabic (UAE)", "sa": "Saudi Arabic", "eg": "Egyptian Arabic", "lv": "Levantine Arabic"}
            dialect_name = dialect_map.get(dialect, "Emirati Arabic (UAE)")  # Default to Emirati
            lang_instruction = f"Respond in {dialect_name}. Use natural, conversational Arabic."
        elif language == "en":
            lang_instruction = "Respond in English."
        else:
            lang_instruction = f"Respond in {language}."
        
        return f"""Generate a {tone} response asking for confirmation before proceeding.

PATIENT CONTEXT:
{patient_context.get_prompt_context()}

ACTION TO CONFIRM:
{response_data.confirmation_action}

DETAILS:
{details_formatted}

SUGGESTED QUESTION:
{response_data.confirmation_question or "Should I proceed?"}

RECENT CONVERSATION:
{recent_context}

RULES:
- Summarize what you're about to do
- Include all relevant details
- Ask for explicit confirmation
- Keep it clear and concise
- Be conversational and natural, but avoid redundancy - don't repeat what was already said
- DO NOT invent patient names
- {lang_instruction}

OUTPUT FORMAT:
Provide ONLY the response text.

Response:"""

    def _build_generic_prompt(
        self,
        response_data: AgentResponseData,
        exec_context: ExecutionContext,
        patient_context: PatientContext,
        tone: str,
        language: str,
        dialect: Optional[str],
        recent_context: str = ""
    ) -> str:
        """Fallback generic prompt builder."""
        execution_summary = self._build_execution_summary(exec_context)
        known_info = self._format_entities(response_data.entities)
        lang_instruction = ""
        if language == "ar":
            # IMPORTANT: Always use Emirati Arabic for Arabic responses
            dialect = dialect or "ae"  # Force Emirati if dialect is None
            dialect_map = {"ae": "Emirati Arabic (UAE)", "sa": "Saudi Arabic", "eg": "Egyptian Arabic", "lv": "Levantine Arabic"}
            dialect_name = dialect_map.get(dialect, "Emirati Arabic (UAE)")  # Default to Emirati
            lang_instruction = f"Respond in {dialect_name}. Use natural, conversational Arabic."
        elif language == "en":
            lang_instruction = "Respond in English."
        else:
            lang_instruction = f"Respond in {language}."
        
        return f"""Generate a {tone} response.

PATIENT CONTEXT:
{patient_context.get_prompt_context()}

CONTEXT:
{known_info}

EXECUTION LOG:
{execution_summary}

RECENT CONVERSATION:
{recent_context}

RULES:
- Be conversational and natural, but avoid redundancy - don't repeat what was already said
- {lang_instruction}

OUTPUT FORMAT:
Provide ONLY the response text.

Response:"""

    def _format_entities(self, entities: Dict[str, Any]) -> str:
        """Format resolved entities for prompts."""
        if not entities:
            return "Nothing collected yet."
        
        lines = []
        for key, value in entities.items():
            if value:
                # Make keys human-readable
                readable_key = key.replace("_", " ").title()
                lines.append(f"  • {readable_key}: {value}")
        
        return "\n".join(lines) if lines else "Nothing collected yet."

    def _get_fallback_response(self, decision: AgentDecision, response_data: AgentResponseData) -> str:
        """Get fallback response if LLM generation fails."""
        fallbacks = {
            AgentDecision.COLLECT_INFORMATION: f"Could you please provide your {response_data.information_needed.replace('_', ' ') if response_data.information_needed else 'information'}?",
            AgentDecision.CLARIFY: response_data.clarification_question or "Could you please clarify?",
            AgentDecision.RESPOND_WITH_OPTIONS: "Here are some options for you to choose from.",
            AgentDecision.RESPOND_IMPOSSIBLE: "I'm sorry, but I wasn't able to complete that request.",
            AgentDecision.RESPOND_COMPLETE: "Your request has been completed.",
            AgentDecision.REQUEST_CONFIRMATION: "Would you like me to proceed?",
        }
        return fallbacks.get(decision, "How can I help you?")

    def _build_execution_summary(self, exec_context: ExecutionContext) -> str:
        """Build compact summary of execution for response generation."""
        lines = []
        for obs in exec_context.observations:
            if obs.type == "tool":
                result = obs.result
                if result.get("success"):
                    # Extract key user-facing data
                    key_data = self._extract_key_data(obs.name, result)
                    lines.append(f"SUCCESS - {obs.name}: {key_data}")
                else:
                    error = result.get("error") or result.get("error_message") or "Failed"
                    lines.append(f"FAILED - {obs.name}: {error}")
        return "\n".join(lines) if lines else "No actions were taken."

    def _extract_key_data(self, tool_name: str, result: Dict[str, Any]) -> str:
        """Extract key user-facing data from tool result."""
        if tool_name == "book_appointment":
            return f"Booked {result.get('doctor_name', 'doctor')} on {result.get('date', '')} at {result.get('time', '')}. Confirmation: {result.get('confirmation_number', result.get('appointment_id', 'N/A'))}"
        elif tool_name == "cancel_appointment":
            return f"Cancelled appointment {result.get('appointment_id', '')}"
        elif tool_name == "check_availability":
            if result.get("available"):
                return f"Available at {result.get('available_slots', result.get('availability_ranges', []))}"
            return f"Not available. Alternatives: {result.get('alternatives', [])}"
        elif tool_name == "list_doctors":
            doctors = result.get("doctors", [])
            if doctors:
                return f"Found {len(doctors)} doctor(s)"
            return "No doctors found"
        elif tool_name == "find_doctor_by_name":
            return f"Found doctor: {result.get('doctor_name', 'Unknown')}"
        # Generic fallback
        return str(result.get("message", result.get("data", "Completed")))[:100]

    def _extract_key_restrictions(self, system_prompt: str) -> str:
        """
        Extract key restrictions from agent's system prompt for focused response generation.
        Looks for sections about what to collect/not collect.
        """
        # Look for common restriction patterns
        restrictions = []
        
        # Check for "ONLY" or "DO NOT" patterns
        lines = system_prompt.split('\n')
        in_restriction_section = False
        restriction_lines = []
        
        for i, line in enumerate(lines):
            # Look for key restriction markers
            if any(marker in line.upper() for marker in ['ONLY', 'DO NOT', 'REQUIRED FIELDS', 'COLLECT ONLY', 'YOUR ROLE']):
                in_restriction_section = True
                restriction_lines.append(line)
            elif in_restriction_section:
                # Continue collecting until we hit a section break or empty line followed by new section
                if line.strip() and not line.strip().startswith('═'):
                    restriction_lines.append(line)
                elif line.strip().startswith('═') or (line.strip() == '' and i < len(lines) - 1 and lines[i+1].strip().startswith('═')):
                    # Hit a new section, stop collecting
                    break
        
        if restriction_lines:
            # Extract the most relevant parts (usually the first few lines with restrictions)
            key_lines = []
            for line in restriction_lines[:15]:  # Limit to first 15 lines
                if any(keyword in line.upper() for keyword in ['ONLY', 'DO NOT', 'REQUIRED', 'COLLECT', 'NOT ASK']):
                    key_lines.append(line.strip())
            
            if key_lines:
                return "\n".join(key_lines)
        
        return ""

# ==================== HELPER CLASSES ====================

class SimpleExecutionContext:
    """
    Tracks the execution state during an agentic loop.

    This maintains a record of all observations (tool results, errors, etc.)

    so the agent can reason about what has happened.

    """
    def __init__(self, user_request: str, session_id: str):
        self.user_request = user_request
        self.session_id = session_id
        self.observations: List[Dict[str, Any]] = []
        self.start_time = datetime.utcnow()

    def add_observation(
        self,
        obs_type: str,
        name: str,
        result: Dict[str, Any]
    ):
        """Add an observation to the execution history."""
        self.observations.append({
            "type": obs_type,
            "name": name,
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        })

    def get_successful_tools(self) -> List[str]:
        """Get list of successfully executed tools."""
        return [
            obs['name'] for obs in self.observations
            if obs['type'] == 'tool' and obs['result'].get('success')
        ]

    def get_failed_tools(self) -> List[str]:
        """Get list of failed tools."""
        return [
            obs['name'] for obs in self.observations
            if obs['type'] == 'tool' and 'error' in obs['result']
        ]

    def has_any_success(self) -> bool:
        """Check if any tool succeeded."""
        return len(self.get_successful_tools()) > 0


class CompletionCheckResult:
    """Result of task completion verification."""
    def __init__(
        self,
        is_complete: bool,
        completed_items: List[str],
        missing_items: List[str],
        response: str
    ):
        self.is_complete = is_complete
        self.completed_items = completed_items
        self.missing_items = missing_items
        self.response = response
