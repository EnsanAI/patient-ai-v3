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
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

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
    AgentDecision
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
    
    def __init__(self, session_id: str, max_iterations: int = 15):
        self.session_id = session_id
        self.max_iterations = max_iterations
        self.iteration = 0
        
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
    
    def initialize_criteria(self, criteria_list: List[str]):
        """Initialize criteria from the reasoning engine."""
        for i, desc in enumerate(criteria_list):
            criterion_id = f"criterion_{i}"
            self.criteria[criterion_id] = Criterion(
                id=criterion_id,
                description=desc,
                state=CriterionState.PENDING
            )
        
        logger.info(f"Initialized {len(criteria_list)} success criteria")
    
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
        """Check if all criteria are met or if we're blocked."""
        result = CompletionCheck(is_complete=False)
        
        for criterion in self.criteria.values():
            if criterion.state == CriterionState.COMPLETE:
                result.completed_criteria.append(criterion.description)
            
            elif criterion.state == CriterionState.BLOCKED:
                result.blocked_criteria.append(criterion.description)
                result.has_blocked = True
                if criterion.blocked_options:
                    result.blocked_options[criterion.description] = criterion.blocked_options
                if criterion.blocked_reason:
                    result.blocked_reasons[criterion.description] = criterion.blocked_reason
            
            elif criterion.state == CriterionState.FAILED:
                result.failed_criteria.append(criterion.description)
                result.has_failed = True
            
            else:  # PENDING or IN_PROGRESS
                result.pending_criteria.append(criterion.description)
        
        # Complete if all criteria are complete (none pending, blocked, or failed)
        result.is_complete = (
            len(result.pending_criteria) == 0 and
            len(result.blocked_criteria) == 0 and
            len(result.failed_criteria) == 0 and
            len(result.completed_criteria) > 0
        )
        
        return result
    
    def has_blocked_criteria(self) -> bool:
        """Check if any criteria are blocked."""
        return any(c.state == CriterionState.BLOCKED for c in self.criteria.values())
    
    def get_blocked_criteria(self) -> List[Criterion]:
        """Get all blocked criteria."""
        return [c for c in self.criteria.values() if c.state == CriterionState.BLOCKED]
    
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

        # Register agent-specific tools
        self._register_tools()

        logger.info(f"Initialized {self.agent_name} agent with ReAct pattern (max_iterations={self.max_iterations})")

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
        logger.debug(f"Registered tool '{name}' for {self.agent_name}")

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
        
        # Initialize execution context
        context = self._context.get(session_id, {})
        exec_context = ExecutionContext(session_id, self.max_iterations)
        
        # Initialize criteria from context
        success_criteria = context.get("success_criteria", [])
        exec_context.initialize_criteria(success_criteria)
        
        # Load continuation context if resuming
        continuation = context.get("continuation_context", {})
        if continuation:
            exec_context.continuation_context = continuation
            logger.info(f"Resuming with continuation context: {list(continuation.keys())}")
        
        logger.info(f"[{self.agent_name}] Starting agentic loop for session {session_id}")
        logger.info(f"[{self.agent_name}] Message: {message[:100]}...")
        logger.info(f"[{self.agent_name}] Criteria: {success_criteria}")
        
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
            
            # -----------------------------------------------------------------
            # ACT: Execute based on decision
            # -----------------------------------------------------------------
            
            if thinking.decision == AgentDecision.CALL_TOOL:
                # Execute tool
                tool_name = thinking.tool_name
                tool_input = thinking.tool_input or {}
                
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
                    return thinking.response_text, execution_log
                
                elif completion.has_blocked:
                    logger.info(f"[{self.agent_name}] ⏸️ Criteria blocked - presenting options")
                    response = self._generate_options_response(exec_context)
                    return response, execution_log
                
                elif completion.has_failed:
                    logger.info(f"[{self.agent_name}] ❌ Criteria failed - explaining")
                    response = self._generate_failure_response(exec_context)
                    return response, execution_log
                
                else:
                    # Not complete but agent thinks it is
                    if thinking.is_task_complete:
                        # Agent explicitly marked complete - trust it
                        logger.info(f"[{self.agent_name}] Agent marked complete - trusting")
                        return thinking.response_text, execution_log
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
                response = thinking.response_text or self._generate_options_response(exec_context)
                return response, execution_log
            
            elif thinking.decision == AgentDecision.RESPOND_COMPLETE:
                logger.info(f"[{self.agent_name}] Task complete")
                return thinking.response_text, execution_log
            
            elif thinking.decision == AgentDecision.RESPOND_IMPOSSIBLE:
                logger.info(f"[{self.agent_name}] Task impossible")
                response = thinking.response_text or self._generate_failure_response(exec_context)
                return response, execution_log
            
            elif thinking.decision == AgentDecision.CLARIFY:
                logger.info(f"[{self.agent_name}] Asking for clarification")
                return thinking.clarification_question, execution_log
            
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

    async def _think(
        self,
        session_id: str,
        execution_context: 'ExecutionContext'
    ) -> ThinkingResult:
        """
        The THINK step of ReAct: Analyze and decide what to do next.

        This is where the LLM reasons about:

        1. What has been done so far (observations)

        2. What the user requested

        3. What still needs to be done

        4. Whether the task is complete

        """
        system_prompt = self._get_thinking_prompt(session_id, execution_context)

        # Build messages with execution context
        messages = self._build_thinking_messages(session_id, execution_context)

        # Get observability logger
        obs_logger = get_observability_logger(session_id) if settings.enable_observability else None
        llm_start_time = time.time()
        thinking_temperature = 0.2  # Lower temperature for more consistent reasoning

        try:
            # Try to get token usage if available
            if hasattr(self.llm_client, 'create_message_with_usage'):
                response, tokens = self.llm_client.create_message_with_usage(
                    system=system_prompt,
                    messages=messages,
                    temperature=thinking_temperature
                )
            else:
                response = self.llm_client.create_message(
                    system=system_prompt,
                    messages=messages,
                    temperature=thinking_temperature
                )
                tokens = TokenUsage()

            llm_duration_seconds = time.time() - llm_start_time

            # Record LLM call
            if obs_logger:
                obs_logger.record_llm_call(
                    component=f"agent.{self.agent_name}.think.legacy",
                    provider=settings.llm_provider.value,
                    model=settings.get_llm_model(),
                    tokens=tokens,
                    duration_seconds=llm_duration_seconds,
                    system_prompt_length=len(system_prompt),
                    messages_count=len(messages),
                    temperature=thinking_temperature,
                    max_tokens=settings.llm_max_tokens,
                    error=None
                )

            # Parse the thinking response
            return self._parse_thinking_response(response, execution_context)

        except Exception as e:
            llm_duration_seconds = time.time() - llm_start_time
            
            # Record failed LLM call
            if obs_logger:
                obs_logger.record_llm_call(
                    component=f"agent.{self.agent_name}.think.legacy",
                    provider=settings.llm_provider.value,
                    model=settings.get_llm_model(),
                    tokens=TokenUsage(),
                    duration_seconds=llm_duration_seconds,
                    system_prompt_length=len(system_prompt),
                    messages_count=len(messages),
                    temperature=thinking_temperature,
                    max_tokens=settings.llm_max_tokens,
                    error=str(e)
                )
            
            logger.error(f"Error in thinking step: {e}", exc_info=True)
            # On error, try to respond with what we have
            return ThinkingResult(
                decision=AgentDecision.RESPOND,
                reasoning=f"Error in reasoning: {str(e)}. Generating response from available data.",
                is_task_complete=False
            )

    def _get_thinking_prompt(
        self,
        session_id: str,
        execution_context: 'ExecutionContext'
    ) -> str:
        """Build the system prompt for the thinking step."""
        base_prompt = self._get_system_prompt(session_id)

        # Add ReAct reasoning instructions
        react_instructions = """
═══════════════════════════════════════════════════════════════
AGENTIC REASONING PROTOCOL (ReAct Pattern)
═══════════════════════════════════════════════════════════════
You are operating in a THINK → ACT → OBSERVE loop. Your job in this step is to:

1. ANALYZE what has been done (review observations below)

2. EVALUATE if the user's request is fulfilled

3. DECIDE your next action

CRITICAL RULES:

1. NEVER claim success before verifying tool results

   - If you called book_appointment, CHECK if the result shows success

   - If tool returned error, DO NOT say "appointment booked"

2. READ TOOL RESULTS CAREFULLY

   - Each observation shows what a tool returned

   - "success": true means the action worked

   - "error" in result means it failed

3. COMPLETE THE FULL TASK

   - If user asked for 2 appointments, book 2 appointments

   - Don't stop after 1 tool call if more are needed

4. VERIFY BEFORE RESPONDING

   - Before saying "done", check: Did all required actions succeed?

   - List what you accomplished vs what was requested

YOUR RESPONSE FORMAT:

```json
{
    "analysis": "What I observe from the execution history...",
    "task_status": {
        "user_requested": "Brief description of what user wants",
        "completed": ["List of completed actions with results"],
        "remaining": ["List of actions still needed"],
        "is_complete": true/false
    },
    "decision": "CALL_TOOL" | "RESPOND" | "RETRY" | "CLARIFY",
    "reasoning": "Why I'm making this decision...",
    "tool_call": {
        "name": "tool_name (if decision is CALL_TOOL)",
        "input": { "param": "value" }
    },
    "response": "Final response text (if decision is RESPOND or CLARIFY)"
}
```

DECISION GUIDE:

- CALL_TOOL: More actions needed to complete the task

- RESPOND: Task is complete AND verified (all tools succeeded)

- RETRY: A tool failed and I should try a different approach

- CLARIFY: I need more information from the user

"""

        # Add execution history summary
        observations_summary = self._format_observations(execution_context)

        full_prompt = f"""{base_prompt}

{react_instructions}

═══════════════════════════════════════════════════════════════
EXECUTION HISTORY (What has happened so far)
═══════════════════════════════════════════════════════════════

{observations_summary if observations_summary else "No actions taken yet."}

═══════════════════════════════════════════════════════════════
AVAILABLE TOOLS
═══════════════════════════════════════════════════════════════

{self._format_tools_for_prompt()}

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
        Generate the final response to the user based on execution results.

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

        if "user_wants" in context:
            parts.append(f"User wants: {context['user_wants']}")

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
        
        # Get tools description
        tools_desc = self._get_tools_description()
        
        # Extract key context fields with safe defaults
        user_intent = context.get('user_intent', context.get('user_wants', 'Not specified'))
        entities = context.get('entities', {})
        constraints = context.get('constraints', [])
        prior_context = context.get('prior_context', 'None')
        routing_action = context.get('routing_action', context.get('action', 'Not specified'))
        
        # Check for continuation
        is_continuation = context.get('is_continuation', False)
        continuation_type = context.get('continuation_type')
        selected_option = context.get('selected_option')
        continuation_context = context.get('continuation_context', {})
        
        # Build continuation section if applicable
        continuation_section = ""
        if is_continuation:
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
        
        return f"""
═══════════════════════════════════════════════════════════════════════════════
🎯 REASONING ENGINE ANALYSIS (Your Instructions)
═══════════════════════════════════════════════════════════════════════════════

The reasoning engine has already analyzed the user's request. USE THIS GUIDANCE:

What User Really Wants: {user_intent}
Recommended Action: {routing_action}
Prior Context: {prior_context}

Entities Identified:
{json.dumps(entities, indent=2, default=str) if entities else '(none)'}

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

Current Language: {context.get('current_language', 'en')}-{context.get('current_dialect', 'unknown')}

═══════════════════════════════════════════════════════════════════════════════
✅ SUCCESS CRITERIA (What You Must Accomplish, if you can)
═══════════════════════════════════════════════════════════════════════════════

{exec_context.get_criteria_display()}

═══════════════════════════════════════════════════════════════════════════════
📊 EXECUTION HISTORY (Iteration {exec_context.iteration})
═══════════════════════════════════════════════════════════════════════════════

{exec_context.get_observations_summary()}

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
    
    "criteria_assessment": {{
        "complete": ["list of completed criteria"],
        "pending": ["list of pending criteria"],
        "blocked": ["list of blocked criteria with reasons"]
    }},
    
    "last_result_analysis": {{
        "tool": "name of last tool called (or null)",
        "result_type": "success/partial/user_input/recoverable/fatal/system_error",
        "interpretation": "what this result means for our task"
    }},
    
    "decision": "CALL_TOOL | RESPOND | RESPOND_WITH_OPTIONS | RESPOND_IMPOSSIBLE | CLARIFY | RETRY",
    "reasoning": "Why I chose this decision",
    
    "is_task_complete": true/false,
    
    // If CALL_TOOL:
    "tool_name": "name of tool to call",
    "tool_input": {{}},
    
    // If RESPOND or RESPOND_WITH_OPTIONS or RESPOND_IMPOSSIBLE:
    "response": "The response to send to user",
    
    // If CLARIFY:
    "clarification_question": "The specific question to ask"
}}

DECISION RULES:
1. If reasoning engine provided clear guidance → EXECUTE IT (call appropriate tools)
2. If all criteria are ✅ → RESPOND with is_task_complete=true
3. If last result has alternatives and available=false → RESPOND_WITH_OPTIONS
4. If task impossible → RESPOND_IMPOSSIBLE
5. Only CLARIFY if execution reveals missing CRITICAL data not in context
"""

    def _parse_thinking_response(
        self,
        response: str,
        exec_context: ExecutionContext
    ) -> ThinkingResult:
        """Parse the LLM's thinking response."""
        try:
            # Extract JSON
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in response")
            
            data = json.loads(json_match.group())
            
            # Parse decision
            decision_str = data.get("decision", "RESPOND").upper()
            decision_map = {
                "CALL_TOOL": AgentDecision.CALL_TOOL,
                "RESPOND": AgentDecision.RESPOND,
                "RESPOND_WITH_OPTIONS": AgentDecision.RESPOND_WITH_OPTIONS,
                "RESPOND_OPTIONS": AgentDecision.RESPOND_WITH_OPTIONS,
                "RESPOND_COMPLETE": AgentDecision.RESPOND_COMPLETE,
                "RESPOND_IMPOSSIBLE": AgentDecision.RESPOND_IMPOSSIBLE,
                "CLARIFY": AgentDecision.CLARIFY,
                "RETRY": AgentDecision.RETRY,
                "EXECUTE_RECOVERY": AgentDecision.EXECUTE_RECOVERY
            }
            decision = decision_map.get(decision_str, AgentDecision.RESPOND)
            
            # Update criteria based on assessment
            assessment = data.get("criteria_assessment", {})
            for completed in assessment.get("complete", []):
                exec_context.mark_criterion_complete(completed)
            
            return ThinkingResult(
                analysis=data.get("analysis", ""),
                task_status=assessment,
                decision=decision,
                reasoning=data.get("reasoning", ""),
                tool_name=data.get("tool_name"),
                tool_input=data.get("tool_input"),
                response_text=data.get("response"),
                clarification_question=data.get("clarification_question"),
                is_task_complete=data.get("is_task_complete", False)
            )
        
        except Exception as e:
            logger.error(f"Error parsing thinking response: {e}")
            logger.debug(f"Response was: {response[:500]}")
            
            # Default to asking for clarification on parse error
            return ThinkingResult(
                decision=AgentDecision.CLARIFY,
                reasoning=f"Parse error: {e}",
                clarification_question="I'm having trouble understanding. Could you please rephrase your request?"
            )

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
        - What remains to do (criteria)
        - What to do next (decision)
        """
        context = self._context.get(session_id, {})
        prompt = self._build_thinking_prompt(message, context, exec_context)
        
        # Get observability logger
        obs_logger = get_observability_logger(session_id) if settings.enable_observability else None
        
        # Call LLM (use default temperature, can be overridden)
        thinking_temperature = getattr(self, 'thinking_temperature', 0.3)
        system_prompt = self._get_thinking_system_prompt()
        llm_start_time = time.time()
        
        try:
            # Try to get token usage if available
            if hasattr(self.llm_client, 'create_message_with_usage'):
                response, tokens = self.llm_client.create_message_with_usage(
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=thinking_temperature
                )
            else:
                response = self.llm_client.create_message(
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=thinking_temperature
                )
                tokens = TokenUsage()
            
            llm_duration_seconds = time.time() - llm_start_time
            
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
            
            # Parse response
            return self._parse_thinking_response(response, exec_context)
        
        except Exception as e:
            llm_duration_seconds = time.time() - llm_start_time
            
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
            
            # Re-raise the exception
            raise

    async def _execute_tool(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        execution_log: ExecutionLog
    ) -> Dict[str, Any]:
        """Execute a tool and return the result."""
        
        if not hasattr(self, '_tools') or tool_name not in self._tools:
            return {
                "success": False,
                "result_type": ToolResultType.FATAL.value,
                "error": f"Unknown tool: {tool_name}",
                "available_tools": list(self._tools.keys()) if hasattr(self, '_tools') and self._tools else []
            }
    
        
        tool_method = self._tools[tool_name]
        start_time = time.time()
        
        try:
            # Call tool (may be sync or async)
            import asyncio
            if asyncio.iscoroutinefunction(tool_method):
                result = await tool_method(**tool_input)
            else:
                result = tool_method(**tool_input)
            
            # Ensure result is a dict
            if not isinstance(result, dict):
                result = {"success": True, "data": result}
            
            # Ensure result_type is set
            if "result_type" not in result:
                result["result_type"] = self._infer_result_type(result)
            
            duration_seconds = time.time() - start_time
            
            # Log to execution log
            execution_log.tools_used.append(ToolExecution(
                tool_name=tool_name,
                inputs=tool_input,
                outputs=result,
                success=result.get("success", True),
                duration_seconds=duration_seconds
            ))
            
            return result
        
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}", exc_info=True)
            
            error_result = {
                "success": False,
                "result_type": ToolResultType.SYSTEM_ERROR.value,
                "error": str(e),
                "should_retry": True
            }
            
            execution_log.tools_used.append(ToolExecution(
                tool_name=tool_name,
                inputs=tool_input,
                outputs=error_result,
                success=False,
                error=str(e)
            ))
            
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
        result_type_str = tool_result.get("result_type", "partial")
        try:
            result_type = ToolResultType(result_type_str)
        except ValueError:
            result_type = ToolResultType.PARTIAL
        
        # Add observation
        exec_context.add_observation(
            obs_type="tool",
            name=tool_name,
            result=tool_result,
            result_type=result_type
        )
        
        # Handle based on result type
        if result_type == ToolResultType.SUCCESS:
            # Check if this satisfies any criteria
            satisfied = tool_result.get("satisfies_criteria", [])
            for criterion in satisfied:
                exec_context.mark_criterion_complete(criterion, evidence=str(tool_result))
            
            # Auto-detect criteria satisfaction
            if tool_result.get("appointment_id"):
                # Look for booking-related criteria
                for crit in exec_context.criteria.values():
                    if "booked" in crit.description.lower() and crit.state == CriterionState.PENDING:
                        exec_context.mark_criterion_complete(
                            crit.id,
                            evidence=f"appointment_id: {tool_result['appointment_id']}"
                        )
            
            return None  # Continue normally
        
        elif result_type == ToolResultType.PARTIAL:
            return None  # Continue normally
        
        elif result_type == ToolResultType.USER_INPUT_NEEDED:
            logger.info(f"Tool {tool_name} requires user input")
            
            # Mark relevant criteria as blocked
            blocked_criteria = tool_result.get("blocks_criteria")
            alternatives = tool_result.get("alternatives", [])
            reason = tool_result.get("reason", "awaiting_user_input")
            
            if blocked_criteria:
                exec_context.mark_criterion_blocked(blocked_criteria, reason, alternatives)
            else:
                # Block any pending booking criteria
                for crit in exec_context.criteria.values():
                    if crit.state == CriterionState.PENDING and "booked" in crit.description.lower():
                        exec_context.mark_criterion_blocked(crit.id, reason, alternatives)
            
            # Store for response generation
            exec_context.pending_user_options = alternatives
            exec_context.suggested_response = tool_result.get("suggested_response")
            
            # Store continuation context
            exec_context.set_continuation_context(
                awaiting="user_selection",
                options=alternatives,
                original_request=tool_result.get("requested_time"),
                **{k: v for k, v in tool_result.items() if k in ["doctor_id", "date", "procedure"]}
            )
            
            # Persist for next turn
            self.state_manager.update_agentic_state(
                session_id,
                status="blocked",
                continuation_context=exec_context.continuation_context
            )
            
            return AgentDecision.RESPOND_WITH_OPTIONS
        
        elif result_type == ToolResultType.RECOVERABLE:
            recovery_action = tool_result.get("recovery_action")
            logger.info(f"Recoverable error from {tool_name}. Suggested: {recovery_action}")
            
            exec_context.add_observation(
                "recovery_hint", tool_name,
                {"suggested_action": recovery_action, "original_error": tool_result.get("error")}
            )
            
            return None  # Let agent figure out recovery in next think
        
        elif result_type == ToolResultType.FATAL:
            logger.warning(f"Fatal error from {tool_name}: {tool_result.get('error')}")
            exec_context.fatal_error = tool_result
            
            # Mark relevant criteria as failed
            for crit in exec_context.criteria.values():
                if crit.state in [CriterionState.PENDING, CriterionState.IN_PROGRESS]:
                    exec_context.mark_criterion_failed(crit.id, tool_result.get("error", "Fatal error"))
            
            return AgentDecision.RESPOND_IMPOSSIBLE
        
        elif result_type == ToolResultType.SYSTEM_ERROR:
            # Check if recovery_action is provided
            recovery_action = tool_result.get("recovery_action")
            recovery_message = tool_result.get("recovery_message")
            error_msg = tool_result.get("error", "")
            
            # Create error signature for tracking
            error_signature = self._create_error_signature(tool_name, error_msg)
            
            # Check if we've already tried recovery for this error
            recovery_info = exec_context.recovery_attempts.get(error_signature)
            has_tried_recovery = recovery_info and recovery_info.get("attempted", False)
            
            if recovery_action and not has_tried_recovery:
                # First time seeing this error with recovery_action - try recovery
                exec_context.recovery_attempts[error_signature] = {
                    "attempted": True,
                    "recovery_action": recovery_action,
                    "recovery_message": recovery_message,
                    "original_tool": tool_name,
                    "original_error": error_msg,
                    "iteration": exec_context.iteration
                }
                exec_context.recovery_executed = True
                logger.info(
                    f"System error detected with recovery_action. Will execute: {recovery_action} "
                    f"(error_signature: {error_signature[:50]}...)"
                )
                return AgentDecision.EXECUTE_RECOVERY
            
            elif has_tried_recovery:
                # Same error after recovery attempt - give up immediately
                logger.error(
                    f"System error persisted after recovery attempt. "
                    f"Error: {error_signature[:50]}... Recovery: {recovery_info.get('recovery_action')}"
                )
                exec_context.fatal_error = tool_result
                return AgentDecision.RESPOND_IMPOSSIBLE
            
            elif exec_context.retry_count < exec_context.max_retries:
                # No recovery_action, use standard retry logic
                logger.info(f"System error, will retry ({exec_context.retry_count + 1}/{exec_context.max_retries})")
                return AgentDecision.RETRY
            else:
                # Max retries exceeded
                logger.error(f"System error after max retries")
                exec_context.fatal_error = tool_result
                return AgentDecision.RESPOND_IMPOSSIBLE
        
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
