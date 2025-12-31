"""
Data models for system observability and telemetry.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class TokenUsage(BaseModel):
    """Token usage tracking with cache support."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    # Cache tracking (Anthropic-specific)
    cache_creation_input_tokens: int = 0  # Tokens written to cache (1.25x cost)
    cache_read_input_tokens: int = 0       # Tokens read from cache (0.1x cost)

    # Extended Thinking tracking (Anthropic-specific)
    thinking_tokens: int = 0  # Internal reasoning tokens (billed as output)

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate (0.0 to 1.0)."""
        total_cacheable = self.cache_creation_input_tokens + self.cache_read_input_tokens
        if total_cacheable == 0:
            return 0.0
        return self.cache_read_input_tokens / total_cacheable

    @property
    def regular_input_tokens(self) -> int:
        """Input tokens not involved in caching."""
        return self.input_tokens - self.cache_creation_input_tokens - self.cache_read_input_tokens

    @property
    def regular_output_tokens(self) -> int:
        """Output tokens excluding thinking tokens."""
        return self.output_tokens - self.thinking_tokens

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        """Add two TokenUsage instances."""
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cache_creation_input_tokens=self.cache_creation_input_tokens + other.cache_creation_input_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens + other.cache_read_input_tokens,
            thinking_tokens=self.thinking_tokens + other.thinking_tokens,
        )


class CostInfo(BaseModel):
    """Cost information for API calls with cache breakdown."""
    input_cost_usd: float = 0.0
    output_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    model: str = ""
    provider: str = ""

    # Cache cost breakdown
    cache_creation_cost_usd: float = 0.0  # Cost of writing to cache (1.25x)
    cache_read_cost_usd: float = 0.0       # Cost of reading from cache (0.1x)
    regular_input_cost_usd: float = 0.0    # Cost of non-cached input tokens

    # Extended Thinking cost breakdown
    thinking_cost_usd: float = 0.0  # Cost of thinking tokens (output rate)

    @property
    def cache_savings_usd(self) -> float:
        """Estimated savings from cache hits vs full price."""
        # If we had paid full price for cache_read tokens instead of 0.1x
        # Savings = (full_price - discounted_price) for cache reads
        if self.cache_read_cost_usd > 0:
            # cache_read is at 0.1x, so full price would be 10x
            full_price = self.cache_read_cost_usd * 10
            return full_price - self.cache_read_cost_usd
        return 0.0

    def __add__(self, other: "CostInfo") -> "CostInfo":
        """Add two CostInfo instances."""
        return CostInfo(
            input_cost_usd=self.input_cost_usd + other.input_cost_usd,
            output_cost_usd=self.output_cost_usd + other.output_cost_usd,
            total_cost_usd=self.total_cost_usd + other.total_cost_usd,
            model=self.model or other.model,
            provider=self.provider or other.provider,
            cache_creation_cost_usd=self.cache_creation_cost_usd + other.cache_creation_cost_usd,
            cache_read_cost_usd=self.cache_read_cost_usd + other.cache_read_cost_usd,
            regular_input_cost_usd=self.regular_input_cost_usd + other.regular_input_cost_usd,
            thinking_cost_usd=self.thinking_cost_usd + other.thinking_cost_usd,
        )


class LLMCall(BaseModel):
    """Information about a single LLM call."""
    call_id: str = Field(default_factory=lambda: f"llm_{datetime.utcnow().timestamp()}")
    component: str = ""  # "reasoning", "agent", "validation", "finalization"
    function_name: Optional[str] = None  # Specific function within component (e.g., "reason", "_think", "validate_response")
    provider: str = ""
    model: str = ""
    system_prompt_length: int = 0
    messages_count: int = 0
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    tokens: TokenUsage = Field(default_factory=TokenUsage)
    cost: CostInfo = Field(default_factory=CostInfo)
    duration_seconds: float = 0.0  # Changed from duration_ms to duration_seconds
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    error: Optional[str] = None


class ToolExecutionDetail(BaseModel):
    """Detailed information about a tool execution."""
    tool_name: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    duration_seconds: float  # Changed from duration_ms to duration_seconds
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    success: bool = True
    error: Optional[str] = None


class AgentContext(BaseModel):
    """Context information passed to an agent."""
    session_id: str
    agent_name: str
    minimal_context: Dict[str, Any] = Field(default_factory=dict)
    conversation_history_length: int = 0
    system_prompt_preview: str = ""  # First 200 chars
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ReasoningStep(BaseModel):
    """A single step in the reasoning chain."""
    step_number: int
    description: str
    context: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ReasoningDetails(BaseModel):
    """Detailed reasoning information."""
    reasoning_chain: List[ReasoningStep] = Field(default_factory=list)
    understanding: Dict[str, Any] = Field(default_factory=dict)
    routing: Dict[str, Any] = Field(default_factory=dict)
    memory_updates: Dict[str, Any] = Field(default_factory=dict)
    response_guidance: Dict[str, Any] = Field(default_factory=dict)
    llm_call: Optional[LLMCall] = None


class PipelineStep(BaseModel):
    """A single step in the orchestrator pipeline."""
    step_number: float  # Changed to float to support decimal steps like 4.5
    step_name: str
    component: str  # "orchestrator", "translation", "reasoning", "agent", etc.
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    duration_seconds: float = 0.0  # Changed from duration_ms to duration_seconds
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentExecutionDetails(BaseModel):
    """Detailed information about agent execution."""
    agent_name: str
    context: AgentContext
    llm_calls: List[LLMCall] = Field(default_factory=list)
    tool_executions: List[ToolExecutionDetail] = Field(default_factory=list)
    total_tokens: TokenUsage = Field(default_factory=TokenUsage)
    total_cost: CostInfo = Field(default_factory=CostInfo)
    duration_seconds: float = 0.0  # Changed from duration_ms to duration_seconds
    response_preview: str = ""  # First 200 chars of response
    tool_scoping_stats: Dict[str, Any] = Field(default_factory=dict)


class ValidationDetails(BaseModel):
    """Details about response validation."""
    is_valid: bool
    confidence: float
    decision: str  # "send", "retry", "redirect", "fallback"
    issues: List[str] = Field(default_factory=list)
    reasoning: List[str] = Field(default_factory=list)
    feedback_to_agent: str = ""
    llm_call: Optional[LLMCall] = None
    retry_count: int = 0


class FinalizationDetails(BaseModel):
    """Details about response finalization."""
    decision: str  # "send", "edit", "fallback"
    confidence: float
    was_rewritten: bool = False
    rewritten_response_preview: str = ""
    issues: List[str] = Field(default_factory=list)
    reasoning: List[str] = Field(default_factory=list)
    llm_call: Optional[LLMCall] = None


class SessionObservability(BaseModel):
    """Complete observability data for a session."""
    session_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    pipeline: Dict[str, Any] = Field(default_factory=dict)
    reasoning: Optional[ReasoningDetails] = None
    agent: Optional[AgentExecutionDetails] = None
    validation: Optional[ValidationDetails] = None
    finalization: Optional[FinalizationDetails] = None
    total_tokens: TokenUsage = Field(default_factory=TokenUsage)
    total_cost: CostInfo = Field(default_factory=CostInfo)
    total_duration_seconds: float = 0.0  # Changed from total_duration_ms to total_duration_seconds
    accumulative_cost: Optional[CostInfo] = None  # Accumulative cost since service start
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "pipeline": self.pipeline,
            "reasoning": self.reasoning.model_dump() if self.reasoning else None,
            "agent": self.agent.model_dump() if self.agent else None,
            "validation": self.validation.model_dump() if self.validation else None,
            "finalization": self.finalization.model_dump() if self.finalization else None,
            "total_tokens": {
                "input_tokens": self.total_tokens.input_tokens,
                "output_tokens": self.total_tokens.output_tokens,
                "total_tokens": self.total_tokens.total_tokens
            },
            "total_cost": {
                "input_cost_usd": self.total_cost.input_cost_usd,
                "output_cost_usd": self.total_cost.output_cost_usd,
                "total_cost_usd": self.total_cost.total_cost_usd,
                "model": self.total_cost.model,
                "provider": self.total_cost.provider
            },
            "total_duration_seconds": self.total_duration_seconds,
            "accumulative_cost": {
                "input_cost_usd": self.accumulative_cost.input_cost_usd if self.accumulative_cost else 0.0,
                "output_cost_usd": self.accumulative_cost.output_cost_usd if self.accumulative_cost else 0.0,
                "total_cost_usd": self.accumulative_cost.total_cost_usd if self.accumulative_cost else 0.0,
                "model": self.accumulative_cost.model if self.accumulative_cost else "",
                "provider": self.accumulative_cost.provider if self.accumulative_cost else ""
            } if self.accumulative_cost else None
        }





