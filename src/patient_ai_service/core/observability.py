"""
Core observability system for tracking system operations.
"""

import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from contextlib import contextmanager

from patient_ai_service.core.config import settings
from patient_ai_service.core.cost_calculator import CostCalculator
from patient_ai_service.core.observability_broadcaster import get_observability_broadcaster
from patient_ai_service.models.observability import (
    TokenUsage,
    CostInfo,
    LLMCall,
    ToolExecutionDetail,
    AgentContext,
    ReasoningStep,
    ReasoningDetails,
    PipelineStep,
    AgentExecutionDetails,
    ValidationDetails,
    FinalizationDetails,
    SessionObservability
)

logger = logging.getLogger(__name__)


class TokenTracker:
    """Tracks token usage across components."""
    
    def __init__(self):
        self._tokens_by_component: Dict[str, TokenUsage] = {}
        self._total_tokens = TokenUsage()
    
    def record_tokens(
        self,
        component: str,
        input_tokens: int,
        output_tokens: int
    ):
        """Record token usage for a component."""
        tokens = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens
        )
        
        if component not in self._tokens_by_component:
            self._tokens_by_component[component] = TokenUsage()
        
        self._tokens_by_component[component] += tokens
        self._total_tokens += tokens
    
    def get_component_tokens(self, component: str) -> TokenUsage:
        """Get token usage for a specific component."""
        return self._tokens_by_component.get(component, TokenUsage())
    
    def get_total_tokens(self) -> TokenUsage:
        """Get total token usage."""
        return self._total_tokens
    
    def reset(self):
        """Reset all token tracking."""
        self._tokens_by_component.clear()
        self._total_tokens = TokenUsage()


class AgentFlowTracker:
    """Tracks agent transitions and communication."""
    
    def __init__(self):
        self._transitions: List[Dict[str, Any]] = []
        self._current_agent: Optional[str] = None
    
    def record_transition(
        self,
        from_agent: Optional[str],
        to_agent: str,
        reason: str,
        context: Dict[str, Any]
    ):
        """Record an agent transition."""
        transition = {
            "timestamp": datetime.utcnow().isoformat(),
            "from_agent": from_agent,
            "to_agent": to_agent,
            "reason": reason,
            "context": context
        }
        self._transitions.append(transition)
        self._current_agent = to_agent
    
    def get_transitions(self) -> List[Dict[str, Any]]:
        """Get all recorded transitions."""
        return self._transitions
    
    def get_current_agent(self) -> Optional[str]:
        """Get current active agent."""
        return self._current_agent
    
    def reset(self):
        """Reset tracking."""
        self._transitions.clear()
        self._current_agent = None


class ReasoningTracker:
    """Tracks reasoning chain details."""
    
    def __init__(self):
        self._reasoning_steps: List[ReasoningStep] = []
        self._understanding: Dict[str, Any] = {}
        self._routing: Dict[str, Any] = {}
        self._memory_updates: Dict[str, Any] = {}
        self._response_guidance: Dict[str, Any] = {}
    
    def record_step(self, step_number: int, description: str, context: Dict[str, Any] = None):
        """Record a reasoning step."""
        step = ReasoningStep(
            step_number=step_number,
            description=description,
            context=context or {}
        )
        self._reasoning_steps.append(step)
    
    def set_understanding(self, understanding: Dict[str, Any]):
        """Set understanding result."""
        self._understanding = understanding
    
    def set_routing(self, routing: Dict[str, Any]):
        """Set routing decision."""
        self._routing = routing
    
    def set_memory_updates(self, memory_updates: Dict[str, Any]):
        """Set memory updates."""
        self._memory_updates = memory_updates
    
    def set_response_guidance(self, response_guidance: Dict[str, Any]):
        """Set response guidance."""
        self._response_guidance = response_guidance
    
    def get_details(self) -> ReasoningDetails:
        """Get complete reasoning details."""
        return ReasoningDetails(
            reasoning_chain=self._reasoning_steps,
            understanding=self._understanding,
            routing=self._routing,
            memory_updates=self._memory_updates,
            response_guidance=self._response_guidance
        )
    
    def reset(self):
        """Reset tracking."""
        self._reasoning_steps.clear()
        self._understanding.clear()
        self._routing.clear()
        self._memory_updates.clear()
        self._response_guidance.clear()


class ToolExecutionTracker:
    """Tracks tool executions with detailed information."""
    
    def __init__(self):
        self._executions: List[ToolExecutionDetail] = []
    
    def record_execution(
        self,
        tool_name: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        duration_seconds: float,
        success: bool = True,
        error: Optional[str] = None
    ):
        """Record a tool execution."""
        execution = ToolExecutionDetail(
            tool_name=tool_name,
            inputs=inputs,
            outputs=outputs,
            duration_seconds=duration_seconds,
            success=success,
            error=error
        )
        self._executions.append(execution)
    
    def get_executions(self) -> List[ToolExecutionDetail]:
        """Get all recorded executions."""
        return self._executions
    
    def reset(self):
        """Reset tracking."""
        self._executions.clear()


class ObservabilityLogger:
    """Main observability logging interface."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.token_tracker = TokenTracker()
        self.agent_flow_tracker = AgentFlowTracker()
        self.reasoning_tracker = ReasoningTracker()
        self.tool_tracker = ToolExecutionTracker()
        self.cost_calculator = CostCalculator()
        
        self._pipeline_steps: List[PipelineStep] = []
        self._llm_calls: List[LLMCall] = []
        self._agent_execution: Optional[AgentExecutionDetails] = None
        self._validation_details: Optional[ValidationDetails] = None
        self._finalization_details: Optional[FinalizationDetails] = None
        self._custom_metrics: List[Dict[str, Any]] = []
        
        self._start_time = time.time()
        self._enabled = settings.enable_observability
        self._broadcaster = get_observability_broadcaster() if settings.enable_observability else None
    
    def reset(self):
        """
        Reset observability logger for a new request.
        
        This clears all per-request data (tokens, steps, LLM calls, etc.)
        but preserves the logger instance for the session.
        Accumulative cost is tracked globally and not reset.
        """
        # Reset all trackers
        self.token_tracker.reset()
        self.agent_flow_tracker.reset()
        self.reasoning_tracker.reset()
        self.tool_tracker.reset()
        
        # Clear all per-request data
        self._pipeline_steps.clear()
        self._llm_calls.clear()
        self._agent_execution = None
        self._validation_details = None
        self._finalization_details = None
        self._custom_metrics.clear()
        
        # Reset start time for new request
        self._start_time = time.time()
        
        logger.debug(f"Reset observability logger for session {self.session_id}")
    
    def is_enabled(self) -> bool:
        """Check if observability is enabled."""
        return self._enabled
    
    def record_pipeline_step(
        self,
        step_number: int,
        step_name: str,
        component: str,
        inputs: Dict[str, Any] = None,
        outputs: Dict[str, Any] = None,
        duration_seconds: float = 0.0,
        error: Optional[str] = None,
        metadata: Dict[str, Any] = None
    ):
        """Record a pipeline step."""
        if not self._enabled:
            return
        
        step = PipelineStep(
            step_number=step_number,
            step_name=step_name,
            component=component,
            inputs=inputs or {},
            outputs=outputs or {},
            duration_seconds=duration_seconds,
            error=error,
            metadata=metadata or {}
        )
        self._pipeline_steps.append(step)
        
        # Broadcast to WebSocket clients (fire and forget)
        if self._broadcaster:
            try:
                import asyncio
                try:
                    loop = asyncio.get_running_loop()
                    # Schedule task without awaiting
                    task = loop.create_task(self._broadcaster.broadcast_pipeline_step(step.model_dump(mode='json')))
                    # Add error callback to log any errors
                    task.add_done_callback(lambda t: logger.error(f"Broadcast task failed: {t.exception()}") if t.exception() else None)
                except RuntimeError:
                    # No running loop, create new one
                    try:
                        asyncio.run(self._broadcaster.broadcast_pipeline_step(step.model_dump(mode='json')))
                    except Exception as run_error:
                        logger.error(f"Error running broadcast in new event loop: {run_error}", exc_info=True)
            except Exception as e:
                logger.error(f"Error broadcasting pipeline step: {e}", exc_info=True)
    
    @contextmanager
    def pipeline_step(
        self,
        step_number: int,
        step_name: str,
        component: str,
        inputs: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None
    ):
        """Context manager for timing pipeline steps."""
        start_time = time.time()
        error = None
        
        try:
            yield
        except Exception as e:
            error = str(e)
            raise
        finally:
            duration_seconds = time.time() - start_time
            self.record_pipeline_step(
                step_number=step_number,
                step_name=step_name,
                component=component,
                inputs=inputs or {},
                duration_seconds=duration_seconds,
                error=error,
                metadata=metadata or {}
            )
    
    def record_llm_call(
        self,
        component: str,
        provider: str,
        model: str,
        tokens: TokenUsage,
        duration_seconds: float,
        system_prompt_length: int = 0,
        messages_count: int = 0,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        error: Optional[str] = None
    ) -> LLMCall:
        """Record an LLM call."""
        if not self._enabled:
            return None
        
        # Calculate cost
        cost = self.cost_calculator.calculate_cost(tokens, model, provider) if settings.cost_tracking_enabled else CostInfo()
        
        # Track tokens
        self.token_tracker.record_tokens(component, tokens.input_tokens, tokens.output_tokens)
        
        llm_call = LLMCall(
            component=component,
            provider=provider,
            model=model,
            system_prompt_length=system_prompt_length,
            messages_count=messages_count,
            temperature=temperature,
            max_tokens=max_tokens,
            tokens=tokens,
            cost=cost,
            duration_seconds=duration_seconds,
            error=error
        )
        
        self._llm_calls.append(llm_call)
        
        # Broadcast to WebSocket clients (fire and forget)
        if self._broadcaster:
            try:
                import asyncio
                try:
                    loop = asyncio.get_running_loop()
                    # Schedule task without awaiting
                    task = loop.create_task(self._broadcaster.broadcast_llm_call(llm_call.model_dump(mode='json')))
                    # Add error callback to log any errors
                    task.add_done_callback(lambda t: logger.error(f"Broadcast task failed: {t.exception()}") if t.exception() else None)
                except RuntimeError:
                    # No running loop, create new one
                    try:
                        asyncio.run(self._broadcaster.broadcast_llm_call(llm_call.model_dump(mode='json')))
                    except Exception as run_error:
                        logger.error(f"Error running broadcast in new event loop: {run_error}", exc_info=True)
            except Exception as e:
                logger.error(f"Error broadcasting LLM call: {e}", exc_info=True)
        
        return llm_call
    
    def record_agent_context(
        self,
        agent_name: str,
        minimal_context: Dict[str, Any],
        conversation_history_length: int,
        system_prompt_preview: str
    ):
        """Record agent context."""
        if not self._enabled:
            return
        
        context = AgentContext(
            session_id=self.session_id,
            agent_name=agent_name,
            minimal_context=minimal_context,
            conversation_history_length=conversation_history_length,
            system_prompt_preview=system_prompt_preview
        )
        
        # Store in agent execution if it exists
        if self._agent_execution:
            self._agent_execution.context = context
    
    def record_tool_execution(
        self,
        tool_name: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        duration_seconds: float,
        success: bool = True,
        error: Optional[str] = None
    ):
        """Record a tool execution."""
        if not self._enabled:
            return
        
        self.tool_tracker.record_execution(
            tool_name=tool_name,
            inputs=inputs,
            outputs=outputs,
            duration_seconds=duration_seconds,
            success=success,
            error=error
        )
        
        # Broadcast to WebSocket clients (fire and forget)
        if self._broadcaster:
            try:
                import asyncio
                execution_dict = {
                    "tool_name": tool_name,
                    "inputs": inputs,
                    "outputs": outputs,
                    "duration_seconds": duration_seconds,
                    "success": success,
                    "error": error,
                    "timestamp": datetime.utcnow().isoformat()
                }
                try:
                    loop = asyncio.get_running_loop()
                    # Schedule task without awaiting
                    task = loop.create_task(self._broadcaster.broadcast_tool_execution(execution_dict))
                    # Add error callback to log any errors
                    task.add_done_callback(lambda t: logger.error(f"Broadcast task failed: {t.exception()}") if t.exception() else None)
                except RuntimeError:
                    # No running loop, create new one
                    try:
                        asyncio.run(self._broadcaster.broadcast_tool_execution(execution_dict))
                    except Exception as run_error:
                        logger.error(f"Error running broadcast in new event loop: {run_error}", exc_info=True)
            except Exception as e:
                logger.error(f"Error broadcasting tool execution: {e}", exc_info=True)
    
    def set_agent_execution(self, execution: AgentExecutionDetails):
        """Set agent execution details."""
        if not self._enabled:
            return
        
        self._agent_execution = execution
    
    def set_validation_details(self, validation: ValidationDetails):
        """Set validation details."""
        if not self._enabled:
            return
        
        self._validation_details = validation
    
    def set_finalization_details(self, finalization: FinalizationDetails):
        """Set finalization details."""
        if not self._enabled:
            return
        
        self._finalization_details = finalization
    
    def record_custom_metric(
        self,
        name: str,
        value: Any,
        tags: Optional[Dict[str, Any]] = None
    ):
        """Record a custom metric."""
        if not self._enabled:
            return
        
        metric = {
            "name": name,
            "value": value,
            "tags": tags or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        self._custom_metrics.append(metric)
        
        # Broadcast to WebSocket clients (fire and forget)
        if self._broadcaster:
            try:
                import asyncio
                try:
                    loop = asyncio.get_running_loop()
                    # Schedule task without awaiting
                    task = loop.create_task(self._broadcaster.broadcast_custom_metric(metric))
                    # Add error callback to log any errors
                    task.add_done_callback(lambda t: logger.error(f"Broadcast task failed: {t.exception()}") if t.exception() else None)
                except RuntimeError:
                    # No running loop, create new one
                    try:
                        asyncio.run(self._broadcaster.broadcast_custom_metric(metric))
                    except Exception as run_error:
                        logger.error(f"Error running broadcast in new event loop: {run_error}", exc_info=True)
            except Exception as e:
                logger.debug(f"Error broadcasting custom metric: {e}")
    
    def get_session_observability(self) -> SessionObservability:
        """Get complete observability data for the session."""
        if not self._enabled:
            return None
        
        # Calculate totals for THIS REQUEST
        total_tokens = self.token_tracker.get_total_tokens()
        total_cost = CostInfo()
        
        # Sum costs from all LLM calls in this request
        for llm_call in self._llm_calls:
            total_cost += llm_call.cost
        
        total_duration_seconds = time.time() - self._start_time
        
        # Get accumulative cost (global across all requests)
        accumulative_cost = get_accumulative_cost()
        
        # Build pipeline summary
        pipeline_summary = {
            "steps": [step.model_dump() for step in self._pipeline_steps],
            "total_tokens": {
                "input_tokens": total_tokens.input_tokens,
                "output_tokens": total_tokens.output_tokens,
                "total_tokens": total_tokens.total_tokens
            },
            "total_cost": {
                "input_cost_usd": total_cost.input_cost_usd,
                "output_cost_usd": total_cost.output_cost_usd,
                "total_cost_usd": total_cost.total_cost_usd
            },
            "duration_seconds": total_duration_seconds,
            "custom_metrics": self._custom_metrics
        }
        
        return SessionObservability(
            session_id=self.session_id,
            pipeline=pipeline_summary,
            reasoning=self.reasoning_tracker.get_details() if self.reasoning_tracker._reasoning_steps else None,
            agent=self._agent_execution,
            validation=self._validation_details,
            finalization=self._finalization_details,
            total_tokens=total_tokens,
            total_cost=total_cost,
            total_duration_seconds=total_duration_seconds,
            accumulative_cost=accumulative_cost
        )
    
    def _format_duration(self, seconds: float) -> str:
        """
        Format duration in human-readable format (minutes and seconds).

        Args:
            seconds: Duration in seconds

        Returns:
            Formatted string (e.g., "1m 23.5s" or "0.456s")
        """
        if seconds < 60:
            return f"{seconds:.3f}s"

        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"

    def _log_pipeline_cost_breakdown(self, observability: SessionObservability):
        """
        Log detailed per-step cost breakdown mapping LLM calls to pipeline steps.

        This provides a clear view of which orchestrator steps cost what, including
        per-iteration breakdown for agentic loops.
        """
        # Map component names to step numbers and names
        step_mapping = {
            "translation.detect_and_translate": (2, "Translation (Input) [OPTIMIZED]"),
            "memory.summarize": (3, "Add to Memory"),
            "reasoning": (4, "Reasoning"),
            "agent.": (8, "Agent Execution"),  # Prefix match for all agent calls
            "validation": (9, "Validation"),
            "finalization": (11, "Finalization"),
            "translation.translate_from_english": (13, "Translation (Output)")
        }

        # Group LLM calls by step
        step_costs = {}  # step_number -> {name, llm_calls, total_cost, total_tokens, duration}
        agent_calls = []  # Special handling for agent calls

        for llm_call in self._llm_calls:
            component = llm_call.component
            matched = False

            # Find matching step
            for prefix, (step_num, step_name) in step_mapping.items():
                if prefix == "agent." and component.startswith(prefix):
                    agent_calls.append(llm_call)
                    matched = True
                    break
                elif component == prefix:
                    if step_num not in step_costs:
                        step_costs[step_num] = {
                            "name": step_name,
                            "llm_calls": [],
                            "total_cost": 0.0,
                            "total_tokens": 0,
                            "duration": 0.0
                        }
                    step_costs[step_num]["llm_calls"].append(llm_call)
                    step_costs[step_num]["total_cost"] += llm_call.cost.total_cost_usd
                    step_costs[step_num]["total_tokens"] += llm_call.tokens.total_tokens
                    step_costs[step_num]["duration"] += llm_call.duration_seconds
                    matched = True
                    break

        # Get pipeline steps for duration info
        pipeline_steps = {step['step_number']: step for step in observability.pipeline.get('steps', [])}

        # Log each step with costs
        for step_num in sorted(step_costs.keys()):
            step_data = step_costs[step_num]
            pipeline_step = pipeline_steps.get(step_num, {})
            step_duration = pipeline_step.get('duration_seconds', step_data['duration'])

            logger.info(f"\nStep {step_num}: {step_data['name']}")
            logger.info(f"  Duration: {self._format_duration(step_duration)}")
            logger.info(f"  LLM Calls: {len(step_data['llm_calls'])}")

            for llm_call in step_data['llm_calls']:
                logger.info(f"    - {llm_call.component}")
                logger.info(f"      Tokens: {llm_call.tokens.input_tokens}/{llm_call.tokens.output_tokens} "
                           f"(total: {llm_call.tokens.total_tokens})")
                if settings.cost_tracking_enabled:
                    logger.info(f"      Cost: ${llm_call.cost.total_cost_usd:.6f}")

            if settings.cost_tracking_enabled:
                logger.info(f"  Step Total Cost: ${step_data['total_cost']:.6f}")

        # Special handling for agent execution with iteration breakdown
        if agent_calls:
            agent_step = pipeline_steps.get(8, {})
            logger.info(f"\nStep 7-8: Agent Activation & Execution")
            if observability.agent:
                logger.info(f"  Agent: {observability.agent.agent_name}")
            logger.info(f"  Duration: {self._format_duration(agent_step.get('duration_seconds', 0))}")
            logger.info(f"\n  AGENTIC LOOP BREAKDOWN:")

            # Group agent calls by type (think vs response generation)
            think_calls = [c for c in agent_calls if ".think" in c.component]
            response_calls = [c for c in agent_calls if "generate" in c.component or "response" in c.component]
            other_calls = [c for c in agent_calls if c not in think_calls and c not in response_calls]

            # Log iterations
            if think_calls:
                logger.info(f"\n  Thinking Iterations: {len(think_calls)}")
                for idx, llm_call in enumerate(think_calls, 1):
                    logger.info(f"\n    Iteration {idx}:")
                    logger.info(f"      Duration: {self._format_duration(llm_call.duration_seconds)}")
                    logger.info(f"      Tokens: {llm_call.tokens.input_tokens}/{llm_call.tokens.output_tokens} "
                               f"(total: {llm_call.tokens.total_tokens})")
                    if settings.cost_tracking_enabled:
                        logger.info(f"      Cost: ${llm_call.cost.total_cost_usd:.6f}")

            # Log response generation
            if response_calls:
                logger.info(f"\n  Final Response Generation:")
                for llm_call in response_calls:
                    logger.info(f"    Component: {llm_call.component}")
                    logger.info(f"    Duration: {self._format_duration(llm_call.duration_seconds)}")
                    logger.info(f"    Tokens: {llm_call.tokens.input_tokens}/{llm_call.tokens.output_tokens} "
                               f"(total: {llm_call.tokens.total_tokens})")
                    if settings.cost_tracking_enabled:
                        logger.info(f"    Cost: ${llm_call.cost.total_cost_usd:.6f}")

            # Other agent calls (verify, etc.)
            if other_calls:
                logger.info(f"\n  Other Agent Operations:")
                for llm_call in other_calls:
                    logger.info(f"    {llm_call.component}")
                    logger.info(f"      Tokens: {llm_call.tokens.total_tokens}")
                    if settings.cost_tracking_enabled:
                        logger.info(f"      Cost: ${llm_call.cost.total_cost_usd:.6f}")

            # Agent total
            if observability.agent and settings.cost_tracking_enabled:
                logger.info(f"\n  Agent Total Cost: ${observability.agent.total_cost.total_cost_usd:.6f}")

        # Steps with no LLM calls (just show duration)
        for step_num, step in sorted(pipeline_steps.items()):
            if step_num not in step_costs and step_num != 8:  # Skip agent execution (handled above)
                logger.info(f"\nStep {step_num}: {step['step_name']}")
                logger.info(f"  Duration: {self._format_duration(step['duration_seconds'])}")
                logger.info(f"  LLM Calls: 0")
                if settings.cost_tracking_enabled:
                    logger.info(f"  Cost: $0.000000")

    def log_summary(self):
        """Log a summary of the session."""
        if not self._enabled:
            return

        observability = self.get_session_observability()
        if not observability:
            return

        # Format based on output format setting
        output_format = settings.observability_output_format

        if output_format == "json":
            self._log_json(observability)
        elif output_format == "structured":
            self._log_structured(observability)
        else:  # detailed
            self._log_detailed(observability)

        # Broadcast summary to WebSocket clients (fire and forget)
        if self._broadcaster:
            try:
                import asyncio
                try:
                    loop = asyncio.get_running_loop()
                    # Schedule task without awaiting
                    task = loop.create_task(self._broadcaster.broadcast_session_summary(observability.to_dict()))
                    # Add error callback to log any errors
                    task.add_done_callback(lambda t: logger.error(f"Broadcast task failed: {t.exception()}") if t.exception() else None)
                except RuntimeError:
                    # No running loop, create new one
                    try:
                        asyncio.run(self._broadcaster.broadcast_session_summary(observability.to_dict()))
                    except Exception as run_error:
                        logger.error(f"Error running broadcast in new event loop: {run_error}", exc_info=True)
            except Exception as e:
                logger.error(f"Error broadcasting session summary: {e}", exc_info=True)
    
    def _log_json(self, observability: SessionObservability):
        """Log in JSON format."""
        data = observability.to_dict()
        json_str = json.dumps(data, indent=2, default=str)
        
        if settings.observability_log_to_file:
            with open(settings.observability_log_to_file, "a") as f:
                f.write(json_str + "\n")
        else:
            logger.info(f"OBSERVABILITY JSON:\n{json_str}")
    
    def _log_structured(self, observability: SessionObservability):
        """Log in structured format with per-request summaries and accumulated totals."""
        logger.info("=" * 80)
        logger.info(f"OBSERVABILITY SUMMARY - Session: {observability.session_id}")
        logger.info("=" * 80)
        
        # INDIVIDUAL REQUEST SUMMARIES
        logger.info("\n📋 INDIVIDUAL REQUEST SUMMARIES:")
        logger.info("-" * 80)
        
        # 1. Reasoning Request
        if observability.reasoning and observability.reasoning.llm_call:
            llm_call = observability.reasoning.llm_call
            logger.info(f"\n1. REASONING REQUEST:")
            logger.info(f"   Component: {llm_call.component}")
            logger.info(f"   Model: {llm_call.model}")
            logger.info(f"   Tokens: {llm_call.tokens.total_tokens} "
                       f"(Input: {llm_call.tokens.input_tokens}, "
                       f"Output: {llm_call.tokens.output_tokens})")
            logger.info(f"   Duration: {self._format_duration(llm_call.duration_seconds)}")
            if settings.cost_tracking_enabled:
                logger.info(f"   Cost: ${llm_call.cost.total_cost_usd:.6f}")
            logger.info(f"   Reasoning Steps: {len(observability.reasoning.reasoning_chain)}")
        
        # 2. Agent LLM Calls (individual requests)
        if observability.agent and observability.agent.llm_calls:
            logger.info(f"\n2. AGENT LLM CALLS ({len(observability.agent.llm_calls)} requests):")
            for idx, llm_call in enumerate(observability.agent.llm_calls, 1):
                logger.info(f"   {idx}. {llm_call.component} - {llm_call.model}")
                logger.info(f"      Tokens: {llm_call.tokens.total_tokens} "
                           f"(Input: {llm_call.tokens.input_tokens}, "
                           f"Output: {llm_call.tokens.output_tokens})")
                logger.info(f"      Duration: {self._format_duration(llm_call.duration_seconds)}")
                if settings.cost_tracking_enabled:
                    logger.info(f"      Cost: ${llm_call.cost.total_cost_usd:.6f}")

        # 3. Agent Tool Executions (individual requests)
        if observability.agent and observability.agent.tool_executions:
            logger.info(f"\n3. AGENT TOOL EXECUTIONS ({len(observability.agent.tool_executions)} requests):")
            for idx, tool in enumerate(observability.agent.tool_executions, 1):
                logger.info(f"   {idx}. {tool.tool_name}")
                logger.info(f"      Duration: {self._format_duration(tool.duration_seconds)}")
                logger.info(f"      Status: {'✓ Success' if tool.success else '✗ Failed'}")
                if tool.error:
                    logger.info(f"      Error: {tool.error}")
        
        # 4. Validation Request
        if observability.validation:
            logger.info(f"\n4. VALIDATION REQUEST:")
            logger.info(f"   Decision: {observability.validation.decision}")
            logger.info(f"   Confidence: {observability.validation.confidence:.2f}")
            logger.info(f"   Retries: {observability.validation.retry_count}")
            if observability.validation.llm_call:
                llm_call = observability.validation.llm_call
                logger.info(f"   Model: {llm_call.model}")
                logger.info(f"   Tokens: {llm_call.tokens.total_tokens}")
                logger.info(f"   Duration: {self._format_duration(llm_call.duration_seconds)}")
                if settings.cost_tracking_enabled:
                    logger.info(f"   Cost: ${llm_call.cost.total_cost_usd:.6f}")

        # 5. Finalization Request
        if observability.finalization:
            logger.info(f"\n5. FINALIZATION REQUEST:")
            logger.info(f"   Decision: {observability.finalization.decision}")
            logger.info(f"   Confidence: {observability.finalization.confidence:.2f}")
            logger.info(f"   Rewritten: {observability.finalization.was_rewritten}")
            if observability.finalization.llm_call:
                llm_call = observability.finalization.llm_call
                logger.info(f"   Model: {llm_call.model}")
                logger.info(f"   Tokens: {llm_call.tokens.total_tokens}")
                logger.info(f"   Duration: {self._format_duration(llm_call.duration_seconds)}")
                if settings.cost_tracking_enabled:
                    logger.info(f"   Cost: ${llm_call.cost.total_cost_usd:.6f}")

        # PIPELINE COST BREAKDOWN (Per Step)
        logger.info("\n" + "=" * 80)
        logger.info("💵 PIPELINE COST BREAKDOWN (Per Step):")
        logger.info("=" * 80)

        self._log_pipeline_cost_breakdown(observability)

        # PER-REQUEST TOTALS (This Request Only)
        logger.info("\n" + "=" * 80)
        logger.info("📊 PER-REQUEST TOTALS (This Request Only):")
        logger.info("=" * 80)

        # Pipeline summary
        logger.info(f"Pipeline Steps: {len(observability.pipeline.get('steps', []))}")
        logger.info(f"Total Duration: {self._format_duration(observability.total_duration_seconds)}")
        
        # Token usage totals (this request)
        logger.info(f"Total Tokens (This Request): {observability.total_tokens.total_tokens} "
                   f"(Input: {observability.total_tokens.input_tokens}, "
                   f"Output: {observability.total_tokens.output_tokens})")
        
        # Cost totals (this request)
        if settings.cost_tracking_enabled:
            logger.info(f"Request Cost: ${observability.total_cost.total_cost_usd:.6f}")
            logger.info(f"  - Input Cost: ${observability.total_cost.input_cost_usd:.6f}")
            logger.info(f"  - Output Cost: ${observability.total_cost.output_cost_usd:.6f}")
            logger.info(f"  - Model: {observability.total_cost.model}")
            logger.info(f"  - Provider: {observability.total_cost.provider}")
        
        # Agent summary
        if observability.agent:
            logger.info(f"\nAgent Summary:")
            logger.info(f"  Agent: {observability.agent.agent_name}")
            logger.info(f"  LLM Calls: {len(observability.agent.llm_calls)}")
            logger.info(f"  Tool Executions: {len(observability.agent.tool_executions)}")
            logger.info(f"  Agent Tokens: {observability.agent.total_tokens.total_tokens}")
            if settings.cost_tracking_enabled:
                logger.info(f"  Agent Cost: ${observability.agent.total_cost.total_cost_usd:.6f}")
        
        # ACCUMULATIVE COST (Since Service Start)
        if settings.cost_tracking_enabled and observability.accumulative_cost:
            logger.info("\n" + "=" * 80)
            logger.info("💰 ACCUMULATIVE COST (Since Service Start):")
            logger.info("=" * 80)
            logger.info(f"Total Accumulative Cost: ${observability.accumulative_cost.total_cost_usd:.6f}")
            logger.info(f"  - Input Cost: ${observability.accumulative_cost.input_cost_usd:.6f}")
            logger.info(f"  - Output Cost: ${observability.accumulative_cost.output_cost_usd:.6f}")
            logger.info(f"  - Model: {observability.accumulative_cost.model}")
            logger.info(f"  - Provider: {observability.accumulative_cost.provider}")
        
        logger.info("=" * 80)
    
    def _log_detailed(self, observability: SessionObservability):
        """Log in detailed format."""
        self._log_structured(observability)
        
        # Additional detailed information
        logger.info("\nDETAILED BREAKDOWN:")
        
        # Pipeline steps
        logger.info("\nPipeline Steps:")
        for step in observability.pipeline.get('steps', []):
            logger.info(f"  {step['step_number']}. {step['step_name']} ({step['component']}) - "
                       f"{self._format_duration(step['duration_seconds'])}")

        # LLM calls
        logger.info("\nLLM Calls:")
        for llm_call in self._llm_calls:
            logger.info(f"  {llm_call.component}: {llm_call.model} - "
                       f"{llm_call.tokens.total_tokens} tokens, "
                       f"{self._format_duration(llm_call.duration_seconds)}")
            if settings.cost_tracking_enabled:
                logger.info(f"    Cost: ${llm_call.cost.total_cost_usd:.6f}")

        # Tool executions
        if observability.agent and observability.agent.tool_executions:
            logger.info("\nTool Executions:")
            for tool in observability.agent.tool_executions:
                logger.info(f"  {tool.tool_name} - {self._format_duration(tool.duration_seconds)} "
                           f"({'success' if tool.success else 'failed'})")
        
        logger.info("")


# Global session observability loggers
_session_loggers: Dict[str, ObservabilityLogger] = {}

# Global accumulative cost tracker (persists across all requests)
_accumulative_cost: CostInfo = CostInfo()
_accumulative_cost_lock = None  # Will be initialized if threading is needed


def get_accumulative_cost() -> CostInfo:
    """Get accumulative cost across all requests since service start."""
    return _accumulative_cost


def _add_to_accumulative_cost(cost: CostInfo):
    """Add cost to global accumulative tracker."""
    global _accumulative_cost
    _accumulative_cost += cost


def get_observability_logger(session_id: str) -> ObservabilityLogger:
    """Get or create observability logger for a session."""
    if session_id not in _session_loggers:
        _session_loggers[session_id] = ObservabilityLogger(session_id)
    return _session_loggers[session_id]


def clear_observability_logger(session_id: str):
    """Clear observability logger for a session (resets per-request data)."""
    if session_id in _session_loggers:
        # Get the logger before deleting
        logger = _session_loggers[session_id]
        
        # Add this request's cost to accumulative before resetting
        if logger._enabled:
            request_cost = CostInfo()
            for llm_call in logger._llm_calls:
                request_cost += llm_call.cost
            if request_cost.total_cost_usd > 0:
                _add_to_accumulative_cost(request_cost)
        
        # Reset the logger (clears tokens, steps, etc. but keeps accumulative cost)
        logger.reset()
        
        # Optionally delete the logger (or keep it for session continuity)
        # For now, we'll keep it but reset it
        # del _session_loggers[session_id]

