"""
Enhanced Base Agent v2 - Complete Agentic Implementation

This version adds:
1. Tool Result Classification (SUCCESS, PARTIAL, USER_INPUT, RECOVERABLE, FATAL, SYSTEM_ERROR)
2. Criterion States (PENDING, IN_PROGRESS, COMPLETE, BLOCKED, FAILED, SKIPPED)
3. Smart Stop Conditions based on result types
4. Continuation support for blocked criteria
5. Response generators for different scenarios

Building on base_agent_improved.py with full result-aware execution.
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple, Set
from datetime import datetime
from pydantic import BaseModel, Field

from patient_ai_service.core.llm import LLMClient, get_llm_client
from patient_ai_service.core.config import settings
from patient_ai_service.core import get_state_manager
from patient_ai_service.core.observability import get_observability_logger
from patient_ai_service.models.validation import ExecutionLog, ToolExecution

logger = logging.getLogger(__name__)


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
            result_str = json.dumps(obs.result, indent=2)
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
    
    def mark_criterion_in_progress(self, description_or_id: str):
        """Mark a criterion as in progress."""
        criterion = self._find_criterion(description_or_id)
        if criterion and criterion.state == CriterionState.PENDING:
            criterion.state = CriterionState.IN_PROGRESS
            criterion.updated_at = datetime.utcnow()
    
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
    
    def get_criteria_summary(self) -> Dict[str, List[str]]:
        """Get summary of criteria by state."""
        summary = {
            "complete": [],
            "pending": [],
            "in_progress": [],
            "blocked": [],
            "failed": []
        }
        
        for criterion in self.criteria.values():
            state_key = criterion.state.value
            if state_key in summary:
                summary[state_key].append(criterion.description)
        
        return summary
    
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
    
    def all_criteria_terminal(self) -> bool:
        """Check if all criteria are in terminal states (complete, failed, skipped)."""
        terminal_states = {CriterionState.COMPLETE, CriterionState.FAILED, CriterionState.SKIPPED}
        return all(c.state in terminal_states for c in self.criteria.values())
    
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


# =============================================================================
# BASE AGENT CLASS - Core Implementation
# =============================================================================

class BaseAgent(ABC):
    """
    Enhanced Base Agent with result-aware agentic execution.
    
    Features:
    - Tool result classification (SUCCESS, PARTIAL, USER_INPUT, etc.)
    - Criterion state tracking (PENDING, COMPLETE, BLOCKED, etc.)
    - Smart stop conditions based on result types
    - Continuation support for multi-turn flows
    """
    
    def __init__(
        self,
        agent_name: str,
        llm_client: Optional[LLMClient] = None,
        db_client: Optional[Any] = None,
        max_iterations: int = 15,
        thinking_temperature: float = 0.2,
        response_temperature: float = 0.5
    ):
        self.agent_name = agent_name
        self.llm_client = llm_client or get_llm_client()
        self.db_client = db_client
        self.state_manager = get_state_manager()
        
        # Configuration
        self.max_iterations = max_iterations
        self.thinking_temperature = thinking_temperature
        self.response_temperature = response_temperature
        
        # Session-specific storage
        self._context: Dict[str, Dict[str, Any]] = {}
        self._tools: Dict[str, callable] = {}
        
        # Register tools
        self._register_tools()
        
        logger.info(f"Initialized {agent_name} with max_iterations={max_iterations}")
    
    # -------------------------------------------------------------------------
    # Abstract Methods (Implement in Subclass)
    # -------------------------------------------------------------------------
    
    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Return the agent's system prompt."""
        pass
    
    @abstractmethod
    def _register_tools(self):
        """Register available tools. Use self._tools[name] = method."""
        pass
    
    # -------------------------------------------------------------------------
    # Context Management
    # -------------------------------------------------------------------------
    
    def set_context(self, session_id: str, context: Dict[str, Any]):
        """
        Set task context from reasoning engine.
        
        Expected context:
        {
            "user_intent": str,
            "entities": Dict,
            "success_criteria": List[str],
            "constraints": List[str],
            "prior_context": str,
            "continuation_context": Dict (if resuming),
            "current_language": str,
            "current_dialect": str
        }
        """
        self._context[session_id] = context
        logger.debug(f"Set context for {self.agent_name}: {context.get('user_intent', '')[:50]}")
    
    def get_context(self, session_id: str) -> Dict[str, Any]:
        """Get context for a session."""
        return self._context.get(session_id, {})
    
    # -------------------------------------------------------------------------
    # Main Entry Point
    # -------------------------------------------------------------------------
    
    async def process_message(
        self,
        session_id: str,
        message: str
    ) -> str:
        """
        Process a message and return response.
        
        Simple interface - use process_message_with_log for detailed logging.
        """
        response, _ = await self.process_message_with_log(session_id, message)
        return response
    
    async def process_message_with_log(
        self,
        session_id: str,
        message: str,
        execution_log: Optional[ExecutionLog] = None
    ) -> Tuple[str, ExecutionLog]:
        """
        Process a message with full execution logging.
        
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
        context = self.get_context(session_id)
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
                
                logger.info(f"[{self.agent_name}] Calling tool: {tool_name}")
                logger.info(f"[{self.agent_name}] Input: {json.dumps(tool_input, indent=2)}")
                
                tool_result = await self._execute_tool(tool_name, tool_input, execution_log)
                
                logger.info(f"[{self.agent_name}] Result: {json.dumps(tool_result, indent=2)[:500]}")
                
                # Process result - may override next action
                override = await self._process_tool_result(
                    session_id, tool_name, tool_result, exec_context
                )
                
                if override:
                    response = self._handle_override(override, exec_context)
                    if response:
                        return response, execution_log
            
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
                exec_context.retry_count += 1
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
    
    # -------------------------------------------------------------------------
    # Thinking Phase
    # -------------------------------------------------------------------------
    
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
        context = self.get_context(session_id)
        prompt = self._build_thinking_prompt(message, context, exec_context)
        
        # Call LLM
        response = self.llm_client.create_message(
            system=self._get_thinking_system_prompt(),
            messages=[{"role": "user", "content": prompt}],
            temperature=self.thinking_temperature
        )
        
        # Parse response
        return self._parse_thinking_response(response, exec_context)
    
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
│                 │    Action: RETRY once, then RESPOND_IMPOSSIBLE             │
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
- Worth trying again

═══════════════════════════════════════════════════════════════════════════════
"""
    
    def _build_thinking_prompt(
        self,
        message: str,
        context: Dict[str, Any],
        exec_context: ExecutionContext
    ) -> str:
        """Build the prompt for the thinking phase."""
        
        # Get tools description
        tools_desc = self._get_tools_description()
        
        return f"""
═══════════════════════════════════════════════════════════════════════════════
CURRENT SITUATION
═══════════════════════════════════════════════════════════════════════════════

User Request: {message}
User Intent: {context.get('user_intent', 'Not specified')}
Entities: {json.dumps(context.get('entities', {}), indent=2)}
Constraints: {context.get('constraints', [])}

═══════════════════════════════════════════════════════════════════════════════
SUCCESS CRITERIA
═══════════════════════════════════════════════════════════════════════════════

{exec_context.get_criteria_display()}

═══════════════════════════════════════════════════════════════════════════════
EXECUTION HISTORY (Iteration {exec_context.iteration})
═══════════════════════════════════════════════════════════════════════════════

{exec_context.get_observations_summary()}

═══════════════════════════════════════════════════════════════════════════════
AVAILABLE TOOLS
═══════════════════════════════════════════════════════════════════════════════

{tools_desc}

═══════════════════════════════════════════════════════════════════════════════
YOUR TASK
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

Remember:
- If last result has alternatives and available=false → RESPOND_WITH_OPTIONS
- If all criteria are ✅ → RESPOND with is_task_complete=true
- If task impossible → RESPOND_IMPOSSIBLE
- Otherwise → CALL_TOOL for next step
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
                "RETRY": AgentDecision.RETRY
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
    
    # -------------------------------------------------------------------------
    # Tool Execution
    # -------------------------------------------------------------------------
    
    def _get_tools_description(self) -> str:
        """Get description of available tools for the prompt."""
        if not self._tools:
            return "No tools available."
        
        lines = []
        for name, method in self._tools.items():
            doc = method.__doc__ or "No description"
            # Get first line of docstring
            first_line = doc.strip().split('\n')[0]
            lines.append(f"- {name}: {first_line}")
        
        return "\n".join(lines)
    
    async def _execute_tool(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        execution_log: ExecutionLog
    ) -> Dict[str, Any]:
        """Execute a tool and return the result."""
        
        if tool_name not in self._tools:
            return {
                "success": False,
                "result_type": ToolResultType.FATAL.value,
                "error": f"Unknown tool: {tool_name}",
                "available_tools": list(self._tools.keys())
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
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Log to execution log
            execution_log.tools_used.append(ToolExecution(
                tool_name=tool_name,
                inputs=tool_input,
                outputs=result,
                success=result.get("success", True),
                duration_ms=duration_ms
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
    
    def _infer_result_type(self, result: Dict[str, Any]) -> str:
        """Infer result type from result content."""
        if result.get("success") is False:
            if result.get("recovery_action"):
                return ToolResultType.RECOVERABLE.value
            elif result.get("should_retry"):
                return ToolResultType.SYSTEM_ERROR.value
            else:
                return ToolResultType.FATAL.value
        
        if result.get("alternatives") and not result.get("available", True):
            return ToolResultType.USER_INPUT_NEEDED.value
        
        if result.get("appointment_id") or result.get("satisfies_criteria"):
            return ToolResultType.SUCCESS.value
        
        return ToolResultType.PARTIAL.value
    
    # -------------------------------------------------------------------------
    # Result Processing
    # -------------------------------------------------------------------------
    
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
            if exec_context.retry_count < exec_context.max_retries:
                logger.info(f"System error, will retry ({exec_context.retry_count + 1}/{exec_context.max_retries})")
                return AgentDecision.RETRY
            else:
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
        
        return None
    
    # -------------------------------------------------------------------------
    # Response Generation
    # -------------------------------------------------------------------------
    
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
    
    # -------------------------------------------------------------------------
    # Lifecycle Hooks
    # -------------------------------------------------------------------------
    
    async def on_activated(self, session_id: str, reasoning: Any):
        """Called when this agent is activated by orchestrator."""
        logger.debug(f"{self.agent_name} activated for session {session_id}")
    
    async def on_deactivated(self, session_id: str):
        """Called when this agent is deactivated."""
        logger.debug(f"{self.agent_name} deactivated for session {session_id}")
