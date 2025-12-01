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
        duration_ms: float,
        success: bool = True,
        error: Optional[str] = None
    ):
        """Record a tool execution."""
        execution = ToolExecutionDetail(
            tool_name=tool_name,
            inputs=inputs,
            outputs=outputs,
            duration_ms=duration_ms,
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
        
        self._start_time = time.time()
        self._enabled = settings.enable_observability
        self._broadcaster = get_observability_broadcaster() if settings.enable_observability else None
    
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
        duration_ms: float = 0.0,
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
            duration_ms=duration_ms,
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
            duration_ms = (time.time() - start_time) * 1000
            self.record_pipeline_step(
                step_number=step_number,
                step_name=step_name,
                component=component,
                inputs=inputs or {},
                duration_ms=duration_ms,
                error=error,
                metadata=metadata or {}
            )
    
    def record_llm_call(
        self,
        component: str,
        provider: str,
        model: str,
        tokens: TokenUsage,
        duration_ms: float,
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
            duration_ms=duration_ms,
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
        duration_ms: float,
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
            duration_ms=duration_ms,
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
                    "duration_ms": duration_ms,
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
    
    def get_session_observability(self) -> SessionObservability:
        """Get complete observability data for the session."""
        if not self._enabled:
            return None
        
        # Calculate totals
        total_tokens = self.token_tracker.get_total_tokens()
        total_cost = CostInfo()
        
        # Sum costs from all LLM calls
        for llm_call in self._llm_calls:
            total_cost += llm_call.cost
        
        total_duration_ms = (time.time() - self._start_time) * 1000
        
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
            "duration_ms": total_duration_ms
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
            total_duration_ms=total_duration_ms
        )
    
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
        """Log in structured format."""
        logger.info("=" * 80)
        logger.info(f"OBSERVABILITY SUMMARY - Session: {observability.session_id}")
        logger.info("=" * 80)
        
        # Pipeline summary
        logger.info(f"Pipeline Steps: {len(observability.pipeline.get('steps', []))}")
        logger.info(f"Total Duration: {observability.total_duration_ms:.2f}ms")
        
        # Token usage
        logger.info(f"Total Tokens: {observability.total_tokens.total_tokens} "
                   f"(Input: {observability.total_tokens.input_tokens}, "
                   f"Output: {observability.total_tokens.output_tokens})")
        
        # Cost
        if settings.cost_tracking_enabled:
            logger.info(f"Total Cost: ${observability.total_cost.total_cost_usd:.6f}")
        
        # Reasoning
        if observability.reasoning:
            logger.info(f"Reasoning Chain: {len(observability.reasoning.reasoning_chain)} steps")
            if observability.reasoning.llm_call:
                logger.info(f"Reasoning Tokens: {observability.reasoning.llm_call.tokens.total_tokens}")
                if settings.cost_tracking_enabled:
                    logger.info(f"Reasoning Cost: ${observability.reasoning.llm_call.cost.total_cost_usd:.6f}")
        
        # Agent
        if observability.agent:
            logger.info(f"Agent: {observability.agent.agent_name}")
            logger.info(f"Agent LLM Calls: {len(observability.agent.llm_calls)}")
            logger.info(f"Agent Tool Executions: {len(observability.agent.tool_executions)}")
            logger.info(f"Agent Tokens: {observability.agent.total_tokens.total_tokens}")
            if settings.cost_tracking_enabled:
                logger.info(f"Agent Cost: ${observability.agent.total_cost.total_cost_usd:.6f}")
        
        # Validation
        if observability.validation:
            logger.info(f"Validation: {observability.validation.decision} "
                       f"(confidence: {observability.validation.confidence:.2f}, "
                       f"retries: {observability.validation.retry_count})")
        
        # Finalization
        if observability.finalization:
            logger.info(f"Finalization: {observability.finalization.decision} "
                       f"(confidence: {observability.finalization.confidence:.2f}, "
                       f"rewritten: {observability.finalization.was_rewritten})")
        
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
                       f"{step['duration_ms']:.2f}ms")
        
        # LLM calls
        logger.info("\nLLM Calls:")
        for llm_call in self._llm_calls:
            logger.info(f"  {llm_call.component}: {llm_call.model} - "
                       f"{llm_call.tokens.total_tokens} tokens, "
                       f"{llm_call.duration_ms:.2f}ms")
            if settings.cost_tracking_enabled:
                logger.info(f"    Cost: ${llm_call.cost.total_cost_usd:.6f}")
        
        # Tool executions
        if observability.agent and observability.agent.tool_executions:
            logger.info("\nTool Executions:")
            for tool in observability.agent.tool_executions:
                logger.info(f"  {tool.tool_name} - {tool.duration_ms:.2f}ms "
                           f"({'success' if tool.success else 'failed'})")
        
        logger.info("")


# Global session observability loggers
_session_loggers: Dict[str, ObservabilityLogger] = {}


def get_observability_logger(session_id: str) -> ObservabilityLogger:
    """Get or create observability logger for a session."""
    if session_id not in _session_loggers:
        _session_loggers[session_id] = ObservabilityLogger(session_id)
    return _session_loggers[session_id]


def clear_observability_logger(session_id: str):
    """Clear observability logger for a session."""
    if session_id in _session_loggers:
        del _session_loggers[session_id]

