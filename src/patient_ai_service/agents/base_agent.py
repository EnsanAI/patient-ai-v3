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
    
    def get_observations_summary(self) -> str:
        """Get a formatted summary of all observations for the thinking prompt."""
        if not self.observations:
            return "No observations yet."
        
        lines = []
        for obs in self.observations:
            status = "✅" if obs.is_success() else "❌"
            result_type_str = f"[{obs.result_type.value}]" if obs.result_type else ""
            
            # Format result (truncate if too long)
            result_str = json.dumps(obs.result, indent=2, default=str)
            if len(result_str) > 500:
                result_str = result_str[:500] + "..."
            
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
        # Identify required parameters (those without defaults)
        required = [
            param_name for param_name, param_schema in parameters.items()
            if param_schema.get("required", True) and "default" not in param_schema
        ]
        schema = {
            "name": name,
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": parameters,
                "required": required if required else list(parameters.keys())
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
        
        # Make LLM call with token tracking
        llm_start_time = time.time()
        system_prompt = "You are a task planner. Output only valid JSON."
        plan_temperature = 0.2
        plan_max_tokens = 500
        
        if hasattr(self.llm_client, 'create_message_with_usage'):
            response, tokens = self.llm_client.create_message_with_usage(
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
                temperature=plan_temperature,
                max_tokens=plan_max_tokens
            )
        else:
            response = self.llm_client.create_message(
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
                temperature=plan_temperature,
                max_tokens=plan_max_tokens
            )
            tokens = TokenUsage()
        
        llm_duration_seconds = time.time() - llm_start_time
        
        # Record LLM call
        if obs_logger:
            llm_call = obs_logger.record_llm_call(
                component=f"agent.{self.agent_name}.plan",
                provider=settings.llm_provider.value,
                model=settings.get_llm_model(),
                tokens=tokens,
                duration_seconds=llm_duration_seconds,
                system_prompt_length=len(system_prompt),
                messages_count=1,
                temperature=plan_temperature,
                max_tokens=plan_max_tokens
            )
            
            # Full observability logging for token calculation
            logger.info(f"📋 [{self.agent_name}] 📊 Plan Generation Token Usage Details:")
            logger.info(f"📋 [{self.agent_name}]   Input tokens: {tokens.input_tokens}")
            logger.info(f"📋 [{self.agent_name}]   Output tokens: {tokens.output_tokens}")
            logger.info(f"📋 [{self.agent_name}]   Total tokens: {tokens.total_tokens}")
            logger.info(f"📋 [{self.agent_name}]   LLM Duration: {llm_duration_seconds * 1000:.2f}ms ({llm_duration_seconds:.3f}s)")
            logger.info(f"📋 [{self.agent_name}]   Model: {settings.get_llm_model()}")
            logger.info(f"📋 [{self.agent_name}]   Provider: {settings.llm_provider.value}")
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
        except Exception as e:
            logger.error(f"📋 [{self.agent_name}] Failed to parse plan: {e}")
            # Fallback: single generic task
            plan.add_task(
                description="Complete the objective",
                tool=None,
                params={}
            )
    
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
        
        # Handle confirmation-related actions directly (no agentic loop needed)
        if routing_action == 'execute_confirmed_action':
            return await self._execute_confirmed_action(session_id, continuation_context, execution_log)
        
        elif routing_action == 'handle_rejection':
            return await self._handle_rejection(session_id, continuation_context, execution_log)
        
        elif routing_action == 'handle_modification':
            # Modification needs to update pending action with new entities, then re-confirm
            return await self._handle_modification(session_id, message, context, execution_log)
        
        # ═══════════════════════════════════════════════════════════════
        # PLAN LIFECYCLE MANAGEMENT (Phase 4)
        # ═══════════════════════════════════════════════════════════════
        plan_action_str = context.get("plan_action", "create_new")
        existing_plan_data = context.get("existing_plan")
        
        plan = None
        if plan_action_str == "create_new" or plan_action_str == "abandon_create":
            # Generate new plan
            logger.info(f"📋 [{self.agent_name}] Creating new plan...")
            plan = await self._plan(
                session_id=session_id,
                objective=context.get("objective", ""),
                entities=context.get("entities", {}),
                constraints=context.get("constraints", [])
            )
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
                    self.state_manager.store_agent_plan(session_id, plan)
            else:
                # Fallback: create new if no existing plan data
                logger.warning(f"📋 [{self.agent_name}] No existing plan data for {plan_action_str}, creating new")
                plan = await self._plan(
                    session_id=session_id,
                    objective=context.get("objective", context.get("user_intent", "")),
                    entities=context.get("entities", {})
                )
                self.state_manager.store_agent_plan(session_id, plan)
        else:
            # Fallback: create new
            logger.warning(f"📋 [{self.agent_name}] Unknown plan_action: {plan_action_str}, creating new")
            plan = await self._plan(
                session_id=session_id,
                objective=context.get("objective", context.get("user_intent", "")),
                entities=context.get("entities", {})
            )
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
        }
        
        # Load continuation context if resuming
        continuation = context.get("continuation_context", {})
        if continuation:
            exec_context.continuation_context = continuation
            logger.info(f"Resuming with continuation context: {list(continuation.keys())}")
        
        logger.info(f"[{self.agent_name}] Starting agentic loop for session {session_id}")
        logger.info(f"[{self.agent_name}] Message: {message[:100]}...")
        if exec_context.plan:
            logger.info(f"[{self.agent_name}] Plan: {exec_context.plan.objective} ({len(exec_context.plan.tasks)} tasks)")
        else:
            logger.warning(f"[{self.agent_name}] No plan in execution context!")
        
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

            # Persist updated resolved entities to global state, if provided
            if getattr(thinking, "updated_resolved_entities", None):
                try:
                    self.state_manager.update_global_state(
                        session_id,
                        resolved_entities=thinking.updated_resolved_entities
                    )
                    logger.info(
                        f"✍️ [{self.agent_name}]   → Updated global resolved_entities: "
                        f"{list(thinking.updated_resolved_entities.keys())}"
                    )
                except Exception as e:
                    logger.warning(
                        f"[{self.agent_name}] ⚠️ Failed to persist resolved_entities: {e}"
                    )
            
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
                                response = self._handle_override(recovery_override, exec_context)
                                if response:
                                    return response, execution_log
                            
                            # Recovery executed, continue loop
                            exec_context.recovery_executed = False
                            continue
                    
                    # Handle other overrides normally
                    response = self._handle_override(override, exec_context)
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
                        response = self._handle_override(override, exec_context)
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
                    # Use RESPOND_COMPLETE decision for unified response
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
                response = await self._generate_response(
                    session_id,
                    AgentDecision.RESPOND_WITH_OPTIONS,
                    thinking.response,
                    exec_context
                )
                return response, execution_log
            
            elif thinking.decision == AgentDecision.RESPOND_COMPLETE:
                logger.info(f"[{self.agent_name}] Task complete - generating response")
                response = await self._generate_response(
                    session_id,
                    AgentDecision.RESPOND_COMPLETE,
                    thinking.response,
                    exec_context
                )
                return response, execution_log
            
            elif thinking.decision == AgentDecision.RESPOND_IMPOSSIBLE:
                logger.info(f"[{self.agent_name}] Task impossible")
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
                # Use updated_resolved_entities as the canonical view if available
                resolved = thinking.updated_resolved_entities or \
                    thinking.response.resolved_entities or \
                    thinking.resolved_entities or \
                    self._extract_resolved_entities(exec_context)
                
                self.state_manager.set_continuation_context(
                    session_id,
                    awaiting=awaiting,
                    options=[],
                    original_request=exec_context.user_request,
                    resolved_entities=resolved,
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

                # Generate response using unified method
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
                awaiting = thinking.response.information_needed or thinking.awaiting_info or "user_information"
                # Use updated_resolved_entities as the canonical view if available
                resolved = thinking.updated_resolved_entities or \
                    thinking.response.resolved_entities or \
                    thinking.resolved_entities or \
                    self._extract_resolved_entities(exec_context)
                
                self.state_manager.set_continuation_context(
                    session_id,
                    awaiting=awaiting,
                    options=[],  # No options for free-form input
                    original_request=exec_context.user_request,
                    resolved_entities=resolved,
                    blocked_criteria=[c.description for c in exec_context.get_blocked_criteria()]
                )
                
                # Update agentic state to blocked
                self.state_manager.update_agentic_state(
                    session_id,
                    status="blocked",
                    active_agent=self.agent_name
                )

                # Mark plan as BLOCKED and persist for next turn (Fix 9)
                self._mark_plan_blocked_and_store(session_id, exec_context, awaiting)

                logger.info(f"[{self.agent_name}] Saved continuation: awaiting={awaiting}, resolved={list(resolved.keys())}")

                # Generate response using unified method
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
                    confirmation_details = thinking.response.confirmation_details
                    confirmation_action = thinking.response.confirmation_action
                    pending_tool = None
                    pending_tool_input = {}
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
                
                # Save the pending action so we can execute it after confirmation
                self.state_manager.set_continuation_context(
                    session_id,
                    awaiting="confirmation",
                    options=["yes", "no", "confirm", "cancel"],
                    original_request=exec_context.user_request,
                    resolved_entities={
                        "pending_action": confirmation_action,
                        "pending_tool": pending_tool,
                        "pending_tool_input": pending_tool_input,
                        **confirmation_details
                    },
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

                # Generate confirmation message using unified method
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
        """Build thinking prompt using cached static content."""
        
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
        try:
            stored = self.state_manager.store_tool_result(
                session_id=session_id,
                tool_name=tool_name,
                tool_params=tool_input,
                tool_result=tool_result
            )
            
            if stored:
                logger.debug(f"[{self.agent_name}] Stored derived entities: {stored}")
        
        except Exception as e:
            # Don't fail the tool call if entity storage fails
            logger.warning(f"Failed to store derived entities: {e}")

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
        3. Plan's resolved_entities
        
        Templates like {{task_1.doctor_id}} or {{doctor_uuid}} are resolved to actual values.
        """
        import re
        
        resolved_input = {}
        
        # Build a resolution context from all available sources
        resolution_context = {}
        
        # 1. From plan's resolved_entities
        if exec_context.plan and exec_context.plan.resolved_entities:
            resolution_context.update(exec_context.plan.resolved_entities)
        
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
        verification_temperature = 0.1

        try:
            # Try to get token usage if available
            if hasattr(self.llm_client, 'create_message_with_usage'):
                response, tokens = self.llm_client.create_message_with_usage(
                    system=system_prompt,
                    messages=[{"role": "user", "content": "Verify task completion."}],
                    temperature=verification_temperature
                )
            else:
                response = self.llm_client.create_message(
                    system=system_prompt,
                    messages=[{"role": "user", "content": "Verify task completion."}],
                    temperature=verification_temperature
                )
                tokens = TokenUsage()

            llm_duration_seconds = time.time() - llm_start_time

            # Record LLM call
            if obs_logger:
                obs_logger.record_llm_call(
                    component=f"agent.{self.agent_name}.verify_completion",
                    provider=settings.llm_provider.value,
                    model=settings.get_llm_model(),
                    tokens=tokens,
                    duration_seconds=llm_duration_seconds,
                    system_prompt_length=len(system_prompt),
                    messages_count=1,
                    temperature=verification_temperature,
                    max_tokens=settings.llm_max_tokens,
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
                    provider=settings.llm_provider.value,
                    model=settings.get_llm_model(),
                    tokens=TokenUsage(),
                    duration_seconds=llm_duration_seconds,
                    system_prompt_length=len(system_prompt),
                    messages_count=1,
                    temperature=verification_temperature,
                    max_tokens=settings.llm_max_tokens,
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
        response_temperature = 0.5

        try:
            user_message = f"Generate a response for: {execution_context.user_request}"
            
            # Try to get token usage if available
            if hasattr(self.llm_client, 'create_message_with_usage'):
                response, tokens = self.llm_client.create_message_with_usage(
                    system=system_prompt,
                    messages=[{
                        "role": "user",
                        "content": user_message
                    }],
                    temperature=response_temperature
                )
            else:
                response = self.llm_client.create_message(
                    system=system_prompt,
                    messages=[{
                        "role": "user",
                        "content": user_message
                    }],
                    temperature=response_temperature
                )
                tokens = TokenUsage()

            llm_duration_seconds = time.time() - llm_start_time

            # Record LLM call
            if obs_logger:
                obs_logger.record_llm_call(
                    component=f"agent.{self.agent_name}.generate_response",
                    provider=settings.llm_provider.value,
                    model=settings.get_llm_model(),
                    tokens=tokens,
                    duration_seconds=llm_duration_seconds,
                    system_prompt_length=len(system_prompt),
                    messages_count=1,
                    temperature=response_temperature,
                    max_tokens=settings.llm_max_tokens,
                    error=None
                )

            return response

        except Exception as e:
            llm_duration_seconds = time.time() - llm_start_time
            
            # Record failed LLM call
            if obs_logger:
                obs_logger.record_llm_call(
                    component=f"agent.{self.agent_name}.generate_response",
                    provider=settings.llm_provider.value,
                    model=settings.get_llm_model(),
                    tokens=TokenUsage(),
                    duration_seconds=llm_duration_seconds,
                    system_prompt_length=len(system_prompt),
                    messages_count=1,
                    temperature=response_temperature,
                    max_tokens=settings.llm_max_tokens,
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

    def _get_thinking_system_prompt(self) -> str:
        """Get system prompt for thinking phase."""
        return f"""You are the thinking module for {self.agent_name}.

Your job is to:
1. Analyze the current situation
2. Decide the best next action
3. Know when to stop

{self._get_result_type_guide()}

{self._get_decision_guide()}

Always respond with valid JSON in the specified format."""

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
- MUST specify resolved_entities (what you already have)

COLLECT_INFORMATION:
- You need to exit the agentic loop to collect information from the user
- Use this when you need to gather data before continuing execution
- Response will pass through focused response generation for natural output
- MUST specify awaiting_info (what you need)
- MUST specify resolved_entities (what you already have)

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
        """Build the prompt for the thinking phase with FULL context from reasoning engine."""
        logger.info(f"📝 [{self.agent_name}] ─────────────────────────────────────────")
        logger.info(f"📝 [{self.agent_name}] BUILDING THINKING PROMPT")
        logger.info(f"📝 [{self.agent_name}] ─────────────────────────────────────────")
        
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
        
        # Get resolved_entities from GlobalState (conversation-scoped memory)
        global_state = self.state_manager.get_global_state(exec_context.session_id)
        resolved_entities = getattr(global_state, "resolved_entities", {}) or {}
        
        logger.info(f"📝 [{self.agent_name}]   → User intent: {user_intent}")
        logger.info(f"📝 [{self.agent_name}]   → Routing action: {routing_action}")
        logger.info(f"📝 [{self.agent_name}]   → Prior context: {prior_context}")
        logger.info(f"📝 [{self.agent_name}]   → Entities: {list(entities.keys()) if entities else 'None'}")
        logger.info(f"✍️ [{self.agent_name}]   → Resolved entities: {list(resolved_entities.keys()) if resolved_entities else 'None'}")
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
{json.dumps(resolved_entities, indent=2, default=str) if resolved_entities else '(none)'}

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
        // Always fill resolved_entities with EVERYTHING you know so far:
        //
        // - Include all relevant facts collected in this conversation
        // - Merge in new information from the current turn
        // - Overwrite values that changed (e.g., new doctor preference)
        // - Remove entries that are no longer valid for the current plan
        "resolved_entities": {{
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
  → Fill: information_needed, information_question, resolved_entities
  → Example: Need visit_reason before booking
  → Response will ASK for missing info, NOT promise action

CLARIFY:
  → Fill: clarification_needed, clarification_question, resolved_entities
  → Example: "3pm" could mean today or tomorrow
  → Response will ask clarifying question

RESPOND_WITH_OPTIONS:
  → Fill: options, options_context, options_reason, resolved_entities
  → Example: Requested time unavailable, here are alternatives
  → Response will present options

RESPOND_IMPOSSIBLE:
  → Fill: failure_reason, failure_suggestion, resolved_entities
  → Example: Doctor doesn't exist in system
  → Response will explain failure and suggest alternatives

RESPOND_COMPLETE:
  → Fill: completion_summary, completion_details, resolved_entities
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
- Fill resolved_entities with EVERYTHING you know so far

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
            resolved_entities = None
            
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
            
            # Extract awaiting_info and resolved_entities for COLLECT_INFORMATION and CLARIFY
            if decision in [AgentDecision.COLLECT_INFORMATION, AgentDecision.CLARIFY]:
                awaiting_info = data.get("awaiting_info")
                resolved_entities = data.get("resolved_entities", {})
                logger.info(f"🔍 [{self.agent_name}]   → Awaiting info: {awaiting_info}")
                logger.info(f"✍️ [{self.agent_name}]   → Resolved entities (from thinking JSON): {list(resolved_entities.keys()) if resolved_entities else 'None'}")
            
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
                    resolved_entities=resolved_entities or {},
                    confirmation_action=confirmation_summary.action if confirmation_summary else None,
                    confirmation_details=confirmation_summary.details if confirmation_summary else {},
                    confirmation_question=None  # Will be generated
                )
            
            # Build result
            logger.info(f"🔍 [{self.agent_name}] Step 9: Building ThinkingResult object...")

            # Determine updated_resolved_entities for persistent conversation memory.
            # Priority order:
            # 1) response.resolved_entities if present
            # 2) legacy resolved_entities field if present
            # 3) fall back to existing global_state.resolved_entities (no change)
            global_state = self.state_manager.get_global_state(exec_context.session_id)
            previous_resolved = getattr(global_state, "resolved_entities", {}) or {}
            updated_resolved = previous_resolved.copy()

            if response_data.resolved_entities:
                updated_resolved.update(response_data.resolved_entities)
            elif resolved_entities:
                updated_resolved.update(resolved_entities)

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
                resolved_entities=resolved_entities,  # Legacy
                confirmation_summary=confirmation_summary,  # Legacy
                response=response_data,  # NEW structured response
                updated_resolved_entities=updated_resolved
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

    def _extract_resolved_entities(self, exec_context: ExecutionContext) -> Dict[str, Any]:
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
            self.state_manager.store_agent_plan(session_id, exec_context.plan)
            logger.info(f"🛑 [{self.agent_name}] Plan marked BLOCKED and stored: {reason}")
        else:
            logger.warning(f"📋 [{self.agent_name}] No plan to mark blocked (awaiting={awaiting})")

    def _generate_confirmation_request(self, summary: ConfirmationSummary) -> str:
        """Generate a confirmation request message for the user."""
        action = summary.action
        details = summary.details
        
        if action == "book_appointment":
            doctor = details.get('doctor_name', 'the doctor')
            date = details.get('date', '')
            time = details.get('time', '')
            procedure = details.get('procedure', 'your appointment')
            
            return (
                f"I'm ready to book your {procedure} with {doctor} "
                f"on {date} at {time}. "
                f"Should I confirm this booking?"
            )
        
        elif action == "cancel_appointment":
            doctor = details.get('doctor_name', 'the doctor')
            date = details.get('date', '')
            time = details.get('time', '')
            
            return (
                f"I'll cancel your appointment with {doctor} "
                f"on {date} at {time}. "
                f"Are you sure you want to cancel?"
            )
        
        elif action == "reschedule_appointment":
            old_date = details.get('old_date', '')
            old_time = details.get('old_time', '')
            new_date = details.get('new_date', details.get('date', ''))
            new_time = details.get('new_time', details.get('time', ''))
            doctor = details.get('doctor_name', 'the doctor')
            
            return (
                f"I'll reschedule your appointment with {doctor} "
                f"from {old_date} at {old_time} "
                f"to {new_date} at {new_time}. "
                f"Should I proceed?"
            )
        
        else:
            # Generic confirmation
            return f"Please confirm: {action}. Proceed?"

    async def _execute_confirmed_action(
        self,
        session_id: str,
        continuation_context: Dict[str, Any],
        execution_log: ExecutionLog
    ) -> Tuple[str, ExecutionLog]:
        """Execute the pending action after user confirmed."""
        resolved = continuation_context.get('resolved_entities', {})
        pending_tool = resolved.get('pending_tool')
        pending_input = resolved.get('pending_tool_input', {})
        
        if not pending_tool:
            logger.warning(f"[{self.agent_name}] No pending tool to execute!")
            return "I'm sorry, I lost track of what we were doing. Could you repeat your request?", execution_log
        
        logger.info(f"[{self.agent_name}] Executing confirmed action: {pending_tool}")
        
        # Clear continuation context BEFORE executing
        self.state_manager.clear_continuation_context(session_id)
        
        # Execute the pending tool
        tool_result = await self._execute_tool(pending_tool, pending_input, execution_log)
        
        # Create execution context for response generation
        exec_context = ExecutionContext(session_id, self.max_iterations, user_request="Confirmed action")
        exec_context.add_observation("tool", pending_tool, tool_result)
        
        if tool_result.get('success'):
            response = await self._generate_focused_response(session_id, exec_context)
        else:
            error = tool_result.get('error', 'The action could not be completed')
            response = f"I'm sorry, there was an issue: {error}. Would you like to try again?"
        
        return response, execution_log

    async def _handle_rejection(
        self,
        session_id: str,
        continuation_context: Dict[str, Any],
        execution_log: ExecutionLog
    ) -> Tuple[str, ExecutionLog]:
        """Handle user rejecting the pending action."""
        logger.info(f"[{self.agent_name}] User rejected pending action")
        
        # Clear continuation context
        self.state_manager.clear_continuation_context(session_id)
        
        return "No problem, I've cancelled that. Is there anything else I can help you with?", execution_log

    async def _handle_modification(
        self,
        session_id: str,
        message: str,
        context: Dict[str, Any],
        execution_log: ExecutionLog
    ) -> Tuple[str, ExecutionLog]:
        """Handle user wanting to modify before confirming."""
        logger.info(f"[{self.agent_name}] User wants to modify pending action")
        
        continuation_context = context.get('continuation_context', {})
        resolved = continuation_context.get('resolved_entities', {}).copy()
        pending_tool = resolved.get('pending_tool')
        pending_input = resolved.get('pending_tool_input', {}).copy()
        
        # Get extracted entities from assessment (new values user provided)
        new_entities = context.get('entities', {})
        
        # Merge new entities into pending input
        # Map from preference keys to tool input keys
        entity_mapping = {
            'time_preference': 'time',
            'date_preference': 'date',
            'doctor_preference': 'doctor_name',
            'procedure_preference': 'procedure',
            'doctor_id': 'doctor_id',
            'time': 'time',
            'date': 'date',
            'doctor_name': 'doctor_name',
            'procedure': 'procedure'
        }
        
        updated = False
        for entity_key, tool_key in entity_mapping.items():
            if entity_key in new_entities and new_entities[entity_key]:
                pending_input[tool_key] = new_entities[entity_key]
                updated = True
                logger.info(f"[{self.agent_name}] Updated {tool_key} to {new_entities[entity_key]}")
        
        if not updated:
            # Couldn't figure out what to modify - ask for clarification
            return "I want to make sure I update the right thing. What would you like to change?", execution_log
        
        # Update the continuation context with new pending input
        resolved['pending_tool_input'] = pending_input
        
        # Also update details for confirmation message
        for key in ['time', 'date', 'doctor_name', 'procedure']:
            if key in pending_input:
                resolved[key] = pending_input[key]
        
        self.state_manager.set_continuation_context(
            session_id,
            awaiting="confirmation",
            options=["yes", "no"],
            original_request=continuation_context.get('original_request'),
            resolved_entities=resolved,
            blocked_criteria=[]
        )
        
        # Generate new confirmation request with updated details
        summary = ConfirmationSummary(
            action=resolved.get('pending_action', 'book_appointment'),
            details={k: v for k, v in resolved.items() if k in ['doctor_name', 'date', 'time', 'procedure']},
            tool_name=pending_tool,
            tool_input=pending_input
        )
        
        response = self._generate_confirmation_request(summary)
        return response, execution_log

    def _get_tools_description(self) -> str:
        """Get description of available tools with full parameter details for the prompt."""
        if not hasattr(self, '_tool_schemas') or not self._tool_schemas:
            return "No tools available."

        lines = []

        for schema in self._tool_schemas:
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
        
        This is where the LLM analyzes:
        - What has been done (observations)
        - What remains to do (plan)
        - What to do next (decision)
        """
        logger.info(f"🧠 [{self.agent_name}] ════════════════════════════════════════════════════════")
        logger.info(f"🧠 [{self.agent_name}] ENTERING _think() - Iteration {exec_context.iteration}")
        logger.info(f"🧠 [{self.agent_name}] ════════════════════════════════════════════════════════")
        plan = exec_context.plan
        logger.info(f"🧠 [{self.agent_name}] ════════════════════════════════════════════════════════")
        logger.info(f"🧠 [{self.agent_name}] Thinking - Plan status: {plan.status.value}")
        logger.info(f"🧠 [{self.agent_name}]   Completed: {len(plan.get_completed_task_ids())}/{len(plan.tasks)}")
        
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
        
        # Call LLM (use default temperature, can be overridden)
        thinking_temperature = getattr(self, 'thinking_temperature', 0.3)
        system_prompt = self._get_thinking_system_prompt()
        logger.info(f"🧠 [{self.agent_name}]   → Model: {settings.get_llm_model()}")
        logger.info(f"🧠 [{self.agent_name}]   → Provider: {settings.llm_provider.value}")
        logger.info(f"🧠 [{self.agent_name}]   → Temperature: {thinking_temperature}")
        logger.info(f"🧠 [{self.agent_name}]   → System prompt length: {len(system_prompt)} chars")
        logger.debug(f"🧠 [{self.agent_name}]   → System prompt:\n{system_prompt}")
        
        llm_start_time = time.time()
        logger.info(f"🧠 [{self.agent_name}]   → Calling LLM at {datetime.utcnow().isoformat()}...")
        
        try:
            # ═════════════════════════════════════════════════════════════════
            # STEP 4: EXECUTE LLM CALL
            # ═════════════════════════════════════════════════════════════════
            logger.info(f"🧠 [{self.agent_name}] [STEP 4/5] Executing LLM call...")
            
            # Try to get token usage if available
            if hasattr(self.llm_client, 'create_message_with_usage'):
                logger.info(f"🧠 [{self.agent_name}]   → Using create_message_with_usage (token tracking enabled)")
                response, tokens = self.llm_client.create_message_with_usage(
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=thinking_temperature
                )
            else:
                logger.info(f"🧠 [{self.agent_name}]   → Using create_message (no token tracking)")
                response = self.llm_client.create_message(
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=thinking_temperature
                )
                tokens = TokenUsage()
            
            llm_duration_seconds = time.time() - llm_start_time
            
            logger.info(f"🧠 [{self.agent_name}]   ✅ LLM call completed in {llm_duration_seconds:.3f}s")
            logger.info(f"🧠 [{self.agent_name}]   → Token usage:")
            logger.info(f"🧠 [{self.agent_name}]       • Input tokens: {tokens.input_tokens if hasattr(tokens, 'input_tokens') else 'N/A'}")
            logger.info(f"🧠 [{self.agent_name}]       • Output tokens: {tokens.output_tokens if hasattr(tokens, 'output_tokens') else 'N/A'}")
            logger.info(f"🧠 [{self.agent_name}]       • Total tokens: {tokens.total_tokens if hasattr(tokens, 'total_tokens') else 'N/A'}")
            logger.info(f"🧠 [{self.agent_name}]   → Response length: {len(response)} chars")
            
            # Log raw LLM response
            logger.info("💬" * 80)
            logger.info(f"🧠 [{self.agent_name}] RAW LLM RESPONSE:")
            logger.info("💬" * 80)
            logger.info(response)
            logger.info("💬" * 80)
            
            # Record LLM call with iteration context
            if obs_logger:
                obs_logger.record_llm_call(
                    component=f"agent.{self.agent_name}.think",
                    provider=settings.llm_provider.value,
                    model=settings.get_llm_model(),
                    tokens=tokens,
                    duration_seconds=llm_duration_seconds,
                    system_prompt_length=len(system_prompt),
                    messages_count=1,
                    temperature=thinking_temperature,
                    max_tokens=settings.llm_max_tokens,
                    error=None
                )
            
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
            
            logger.error(f"🧠 [{self.agent_name}]   ❌ LLM call FAILED after {llm_duration_seconds:.3f}s")
            logger.error(f"🧠 [{self.agent_name}]   → Error type: {type(e).__name__}")
            logger.error(f"🧠 [{self.agent_name}]   → Error message: {str(e)}")
            logger.error(f"🧠 [{self.agent_name}]   → Stack trace:", exc_info=True)
            
            # Record failed LLM call
            if obs_logger:
                obs_logger.record_llm_call(
                    component=f"agent.{self.agent_name}.think",
                    provider=settings.llm_provider.value,
                    model=settings.get_llm_model(),
                    tokens=TokenUsage(),
                    duration_seconds=llm_duration_seconds,
                    system_prompt_length=len(system_prompt),
                    messages_count=1,
                    temperature=thinking_temperature,
                    max_tokens=settings.llm_max_tokens,
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
        
        tool_method = self._tools[tool_name]
        start_time = time.time()
        logger.info(f"🛠️  [{self.agent_name}] Step 3: Calling tool method at {datetime.utcnow().isoformat()}...")
        
        try:
            # Call tool (may be sync or async)
            import asyncio
            if asyncio.iscoroutinefunction(tool_method):
                logger.info(f"🛠️  [{self.agent_name}]   → Tool is async, awaiting...")
                result = await tool_method(**tool_input)
            else:
                logger.info(f"🛠️  [{self.agent_name}]   → Tool is sync, calling directly...")
                result = tool_method(**tool_input)
            
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
        
        # Step 3: Handle based on result type
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

    def _handle_override(
        self,
        override: AgentDecision,
        exec_context: ExecutionContext
    ) -> Optional[str]:
        """Handle an override decision from result processing."""
        
        if override == AgentDecision.RESPOND_WITH_OPTIONS:
            return self._generate_options_response(exec_context)
        
        elif override == AgentDecision.RESPOND_IMPOSSIBLE:
            return self._generate_failure_response(exec_context)
        
        elif override == AgentDecision.RETRY:
            return None  # Continue loop
        
        elif override == AgentDecision.EXECUTE_RECOVERY:
            return None  # Continue loop (handled in main loop)
        
        return None

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

    async def _generate_focused_response(
        self,
        session_id: str,
        exec_context: ExecutionContext
    ) -> str:
        """
        Generate response AFTER agent decides to respond.
        Uses full execution context, what_user_means/user_intent, and tone.
        """
        logger.info(f"🅰 [{self.agent_name}] Initializing focused response generation (Option 2)")
        
        context = self._context.get(session_id, {})
        
        # Use user_intent as primary, with fallback to what_user_means
        user_intent = context.get("user_intent") or context.get("what_user_means", "")
        tone = context.get("tone", "helpful")
        language = context.get("current_language", "en")
        dialect = context.get("current_dialect")
        
        execution_summary = self._build_execution_summary(exec_context)
        
        # Get agent's system prompt to include critical restrictions
        agent_system_prompt = self._get_system_prompt(session_id)
        # Extract key restrictions section if present (for registration agent, this is critical)
        agent_restrictions = self._extract_key_restrictions(agent_system_prompt)
        
        logger.info(f"👩🏻‍💻 [{self.agent_name}] Focused response inputs:")
        logger.info(f"👩🏻‍💻   - user_intent: {user_intent}")
        logger.info(f"👩🏻‍💻   - tone: {tone}")
        logger.info(f"👩🏻‍💻   - language: {language}")
        logger.info(f"👩🏻‍💻   - dialect: {dialect}")
        logger.info(f"👩🏻‍💻   - execution_summary: {execution_summary}")
        logger.info(f"👩🏻‍💻   - observations_count: {len(exec_context.observations)}")
        logger.info(f"👩🏻‍💻   - agent_restrictions: {agent_restrictions[:200] if agent_restrictions else 'None'}...")
        
        system_prompt = f"""Generate a {tone} response for a clinic receptionist.

USER INTENT: {user_intent}
TONE: {tone}
LANGUAGE: {language}
DIALECT: {dialect if dialect else ""}

WHAT HAPPENED:
{execution_summary}

{agent_restrictions if agent_restrictions else ""}

RULES:
- Report what actually happened (based on execution results above)
- Be {tone} and super natural
- Don't be overly friendly, just warm
- Don't sound redundant
- No JSON, UUIDs, or technical details
- Use user's language preference
- Keep it concise (2-4 sentences)
- **CRITICAL**: If this is a registration agent asking for information, ONLY ask for the required fields specified in the agent restrictions above. DO NOT ask for optional fields like email, insurance, allergies, medications, etc."""

        logger.info(f"🅰 [{self.agent_name}] System prompt length: {len(system_prompt)} chars")
        logger.debug(f"🅰 [{self.agent_name}] Full system prompt:\n{system_prompt}")

        # Get observability logger
        obs_logger = get_observability_logger(session_id) if settings.enable_observability else None
        llm_start_time = time.time()
        
        logger.info(f"🅰 [{self.agent_name}] Calling LLM for focused response generation...")

        try:
            # Try to get token usage if available
            if hasattr(self.llm_client, 'create_message_with_usage'):
                response, tokens = self.llm_client.create_message_with_usage(
                    system=system_prompt,
                    messages=[{"role": "user", "content": "Generate the response."}],
                    temperature=0.5,
                    max_tokens=300
                )
            else:
                response = self.llm_client.create_message(
                    system=system_prompt,
                    messages=[{"role": "user", "content": "Generate the response."}],
                    temperature=0.5,
                    max_tokens=300
                )
                tokens = TokenUsage()

            llm_duration_seconds = time.time() - llm_start_time

            # Logging with error handling - don't let logging errors prevent response return
            try:
                logger.info(f"🅰 [{self.agent_name}] LLM call completed in {llm_duration_seconds:.2f}s")
                logger.info(f"🅰 [{self.agent_name}] Token usage - input: {tokens.input_tokens}, output: {tokens.output_tokens}, total: {tokens.total_tokens}")
                logger.info(f"🅰 [{self.agent_name}] Generated response ({len(response)} chars): {response}")
                logger.debug(f"🅰 [{self.agent_name}] Full response text:\n{response}")
            except Exception as log_error:
                # Logging error shouldn't prevent response return
                logger.warning(f"🅰 [{self.agent_name}] Error in logging (non-critical): {log_error}")

            # Record LLM call
            try:
                if obs_logger:
                    obs_logger.record_llm_call(
                        component=f"agent.{self.agent_name}.generate_focused_response",
                        provider=settings.llm_provider.value,
                        model=settings.get_llm_model(),
                        tokens=tokens,
                        duration_seconds=llm_duration_seconds,
                        system_prompt_length=len(system_prompt),
                        messages_count=1,
                        temperature=0.5,
                        max_tokens=300,
                        error=None
                    )
            except Exception as obs_error:
                # Observability logging error shouldn't prevent response return
                logger.warning(f"🅰 [{self.agent_name}] Error in observability logging (non-critical): {obs_error}")

            logger.info(f"🅰 [{self.agent_name}] Focused response generation completed successfully")
            return response

        except Exception as e:
            logger.error(f"🅰 [{self.agent_name}] Error generating focused response: {e}", exc_info=True)
            error_response = "I apologize, but I encountered an issue processing your request. Please try again."
            logger.info(f"🅰 [{self.agent_name}] Returning error fallback response: {error_response}")
            return error_response

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
            resolved_entities=response_data.resolved_entities
        )
        
        logger.info(f"🎯 [{self.agent_name}] Patient context: {patient_context.get_prompt_context()}")
        
        # Format resolved entities
        known_info = self._format_resolved_entities(response_data.resolved_entities)
        
        # Build decision-specific prompt
        if decision == AgentDecision.COLLECT_INFORMATION:
            prompt = self._build_collect_info_prompt(response_data, known_info, patient_context, tone, language, dialect)
            
        elif decision == AgentDecision.CLARIFY:
            prompt = self._build_clarify_prompt(response_data, known_info, patient_context, tone, language, dialect)
            
        elif decision == AgentDecision.RESPOND_WITH_OPTIONS:
            prompt = self._build_options_prompt(response_data, known_info, patient_context, tone, language, dialect)
            
        elif decision == AgentDecision.RESPOND_IMPOSSIBLE:
            prompt = self._build_impossible_prompt(response_data, known_info, patient_context, tone, language, dialect)
            
        elif decision in [AgentDecision.RESPOND, AgentDecision.RESPOND_COMPLETE]:
            prompt = self._build_completion_prompt(response_data, exec_context, patient_context, tone, language, dialect)
            
        elif decision == AgentDecision.REQUEST_CONFIRMATION:
            prompt = self._build_confirmation_prompt(response_data, known_info, patient_context, tone, language, dialect)
            
        else:
            # Fallback
            prompt = self._build_generic_prompt(response_data, exec_context, patient_context, tone, language, dialect)
        
        # Generate response
        try:
            obs_logger = get_observability_logger(session_id) if settings.enable_observability else None
            llm_start_time = time.time()
            
            if hasattr(self.llm_client, 'create_message_with_usage'):
                response, tokens = self.llm_client.create_message_with_usage(
                    system=prompt,
                    messages=[{"role": "user", "content": "Generate the response."}],
                    temperature=0.5,
                    max_tokens=300
                )
            else:
                response = self.llm_client.create_message(
                    system=prompt,
                    messages=[{"role": "user", "content": "Generate the response."}],
                    temperature=0.5,
                    max_tokens=300
                )
                tokens = TokenUsage()
            
            llm_duration_seconds = time.time() - llm_start_time
            
            # Record LLM call
            if obs_logger:
                try:
                    obs_logger.record_llm_call(
                        component=f"agent.{self.agent_name}.generate_response",
                        provider=settings.llm_provider.value,
                        model=settings.get_llm_model(),
                        tokens=tokens,
                        duration_seconds=llm_duration_seconds,
                        system_prompt_length=len(prompt),
                        messages_count=1,
                        temperature=0.5,
                        max_tokens=300,
                        error=None
                    )
                except Exception as obs_error:
                    logger.warning(f"🎯 [{self.agent_name}] Error in observability logging (non-critical): {obs_error}")
            
            logger.info(f"🎯 [{self.agent_name}] Generated: {response}")
            return response
            
        except Exception as e:
            logger.error(f"🎯 [{self.agent_name}] Error generating response: {e}")
            return self._get_fallback_response(decision, response_data)

    def _build_collect_info_prompt(
        self,
        response_data: AgentResponseData,
        known_info: str,
        patient_context: PatientContext,
        tone: str,
        language: str,
        dialect: Optional[str]
    ) -> str:
        """Build prompt for COLLECT_INFORMATION decision."""
        return f"""Generate a {tone} question to collect missing information.

═══════════════════════════════════════════════════════════════
PATIENT CONTEXT
═══════════════════════════════════════════════════════════════
{patient_context.get_prompt_context()}

⚠️  CRITICAL: If patient name is not available, DO NOT invent one!

═══════════════════════════════════════════════════════════════
WHAT WE ALREADY KNOW
═══════════════════════════════════════════════════════════════
{known_info}

═══════════════════════════════════════════════════════════════
WHAT WE NEED
═══════════════════════════════════════════════════════════════
{response_data.information_needed}

SUGGESTED QUESTION:
{response_data.information_question or f"Ask for {response_data.information_needed}"}

LANGUAGE: {language}
DIALECT: {dialect or "standard"}

RULES:
- Ask ONLY for the missing information
- Be conversational and natural
- DO NOT say "I'll check" or "I'll do X" - nothing has been done yet!
- DO NOT promise any action
- DO NOT invent patient names - use only what's in PATIENT CONTEXT
- Keep it to 1-2 sentences

FORBIDDEN:
- Inventing names (like "John", "Sarah", etc.) if not in patient context
- "I'll check availability"
- "I'll book that for you"  
- Any promise of future action"""

    def _build_clarify_prompt(
        self,
        response_data: AgentResponseData,
        known_info: str,
        patient_context: PatientContext,
        tone: str,
        language: str,
        dialect: Optional[str]
    ) -> str:
        """Build prompt for CLARIFY decision."""
        return f"""Generate a {tone} question to clarify ambiguous input.

═══════════════════════════════════════════════════════════════
PATIENT CONTEXT
═══════════════════════════════════════════════════════════════
{patient_context.get_prompt_context()}

⚠️  CRITICAL: If patient name is not available, DO NOT invent one!

═══════════════════════════════════════════════════════════════
WHAT WE ALREADY KNOW
═══════════════════════════════════════════════════════════════
{known_info}

═══════════════════════════════════════════════════════════════
WHAT'S UNCLEAR
═══════════════════════════════════════════════════════════════
{response_data.clarification_needed or "Something needs clarification"}

SUGGESTED QUESTION:
{response_data.clarification_question or "Could you please clarify?"}

LANGUAGE: {language}
DIALECT: {dialect or "standard"}

RULES:
- Ask a clarifying question
- Be conversational and natural
- DO NOT invent patient names
- Keep it to 1-2 sentences"""

    def _build_options_prompt(
        self,
        response_data: AgentResponseData,
        known_info: str,
        patient_context: PatientContext,
        tone: str,
        language: str,
        dialect: Optional[str]
    ) -> str:
        """Build prompt for RESPOND_WITH_OPTIONS decision."""
        options_formatted = "\n".join([f"  • {opt}" for opt in response_data.options])
        
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

LANGUAGE: {language}
DIALECT: {dialect or "standard"}

RULES:
- Briefly explain why we're offering alternatives
- Present the options clearly
- Ask which they prefer
- Keep it concise (2-4 sentences)
- Don't apologize excessively
- DO NOT invent patient names"""

    def _build_impossible_prompt(
        self,
        response_data: AgentResponseData,
        known_info: str,
        patient_context: PatientContext,
        tone: str,
        language: str,
        dialect: Optional[str]
    ) -> str:
        """Build prompt for RESPOND_IMPOSSIBLE decision."""
        return f"""Generate a {tone} response explaining why the request cannot be fulfilled.

═══════════════════════════════════════════════════════════════
PATIENT CONTEXT
═══════════════════════════════════════════════════════════════
{patient_context.get_prompt_context()}

WHAT WAS ATTEMPTED:
{known_info}

WHY IT'S NOT POSSIBLE:
{response_data.failure_reason}

SUGGESTION FOR USER:
{response_data.failure_suggestion or "No specific suggestion"}

LANGUAGE: {language}
DIALECT: {dialect or "standard"}

RULES:
- Be empathetic but direct
- Explain the issue clearly
- Offer the suggestion if available
- Don't over-apologize
- Keep it to 2-3 sentences
- DO NOT invent patient names"""

    def _build_completion_prompt(
        self,
        response_data: AgentResponseData,
        exec_context: ExecutionContext,
        patient_context: PatientContext,
        tone: str,
        language: str,
        dialect: Optional[str]
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
        
        return f"""Generate a {tone} response confirming what was accomplished.

═══════════════════════════════════════════════════════════════
PATIENT CONTEXT  
═══════════════════════════════════════════════════════════════
{patient_context.get_prompt_context()}
{address_instruction}

═══════════════════════════════════════════════════════════════
WHAT WAS DONE
═══════════════════════════════════════════════════════════════
{response_data.completion_summary or "Task completed"}

DETAILS:
{details_formatted}

EXECUTION LOG:
{execution_summary}

LANGUAGE: {language}
DIALECT: {dialect or "standard"}

RULES:
- Confirm the action was completed
- Include key details (confirmation number, date, time, etc.)
- Be warm but concise
- No technical details or UUIDs
- DO NOT invent patient names
- 2-3 sentences"""

    def _build_confirmation_prompt(
        self,
        response_data: AgentResponseData,
        known_info: str,
        patient_context: PatientContext,
        tone: str,
        language: str,
        dialect: Optional[str]
    ) -> str:
        """Build prompt for REQUEST_CONFIRMATION decision."""
        details_formatted = "\n".join([
            f"  • {k}: {v}" for k, v in response_data.confirmation_details.items() if v
        ]) if response_data.confirmation_details else "No details"
        
        return f"""Generate a {tone} response asking for confirmation before proceeding.

═══════════════════════════════════════════════════════════════
PATIENT CONTEXT
═══════════════════════════════════════════════════════════════
{patient_context.get_prompt_context()}

ACTION TO CONFIRM:
{response_data.confirmation_action}

DETAILS:
{details_formatted}

SUGGESTED QUESTION:
{response_data.confirmation_question or "Should I proceed?"}

LANGUAGE: {language}
DIALECT: {dialect or "standard"}

RULES:
- Summarize what you're about to do
- Include all relevant details
- Ask for explicit confirmation
- Keep it clear and concise
- DO NOT invent patient names"""

    def _build_generic_prompt(
        self,
        response_data: AgentResponseData,
        exec_context: ExecutionContext,
        patient_context: PatientContext,
        tone: str,
        language: str,
        dialect: Optional[str]
    ) -> str:
        """Fallback generic prompt builder."""
        execution_summary = self._build_execution_summary(exec_context)
        known_info = self._format_resolved_entities(response_data.resolved_entities)
        
        return f"""Generate a {tone} response.

═══════════════════════════════════════════════════════════════
PATIENT CONTEXT
═══════════════════════════════════════════════════════════════
{patient_context.get_prompt_context()}

CONTEXT:
{known_info}

EXECUTION LOG:
{execution_summary}

LANGUAGE: {language}
DIALECT: {dialect or "standard"}

RULES:
- Be helpful and natural
- DO NOT invent patient names
- Keep it concise"""

    def _format_resolved_entities(self, entities: Dict[str, Any]) -> str:
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
