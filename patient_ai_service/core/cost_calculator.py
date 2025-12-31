"""
Cost calculation for LLM API calls based on model and provider.
"""

import logging
from typing import Dict, Optional
from patient_ai_service.models.enums import LLMProvider, AnthropicModel, OpenAIModel
from patient_ai_service.models.observability import TokenUsage, CostInfo

logger = logging.getLogger(__name__)


# Model pricing per 1M tokens (as of implementation date)
# Prices are in USD
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    # Anthropic Claude models
    "claude-sonnet-4-5-20250929": {
        "input": 3.0,   # $3 per 1M input tokens
        "output": 15.0  # $15 per 1M output tokens
    },
    "claude-sonnet-4-20250514": {
        "input": 3.0,
        "output": 15.0
    },
    "claude-3-5-haiku-20241022": {
        "input": 0.25,  # $0.25 per 1M input tokens
        "output": 1.25  # $1.25 per 1M output tokens
    },
    # OpenAI models
    "gpt-4o-mini": {
        "input": 0.15,  # $0.15 per 1M input tokens
        "output": 0.6   # $0.6 per 1M output tokens
    },
    "gpt-4o": {
        "input": 2.5,
        "output": 10.0
    },
    "gpt-4-turbo": {
        "input": 10.0,
        "output": 30.0
    },
    "gpt-3.5-turbo": {
        "input": 0.5,
        "output": 1.5
    },
    # Default fallback pricing (use Haiku pricing as conservative estimate)
    "default": {
        "input": 0.25,
        "output": 1.25
    }
}


class CostCalculator:
    """Calculate costs for LLM API calls based on token usage and model."""
    
    @staticmethod
    def calculate_cost(
        tokens: TokenUsage,
        model: str,
        provider: str
    ) -> CostInfo:
        """
        Calculate cost for token usage.
        
        Args:
            tokens: Token usage information
            model: Model name
            provider: Provider name (anthropic, openai)
            
        Returns:
            CostInfo with calculated costs
        """
        # Normalize model name
        model_key = model.lower()
        
        # Get pricing for this model
        pricing = MODEL_PRICING.get(model_key, MODEL_PRICING["default"])
        
        # Calculate costs (prices are per 1M tokens)
        input_cost = (tokens.input_tokens / 1_000_000) * pricing["input"]
        output_cost = (tokens.output_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost
        
        return CostInfo(
            input_cost_usd=round(input_cost, 6),
            output_cost_usd=round(output_cost, 6),
            total_cost_usd=round(total_cost, 6),
            model=model,
            provider=provider
        )
    
    @staticmethod
    def get_model_pricing(model: str) -> Dict[str, float]:
        """
        Get pricing information for a model.
        
        Args:
            model: Model name
            
        Returns:
            Dictionary with input and output pricing per 1M tokens
        """
        model_key = model.lower()
        return MODEL_PRICING.get(model_key, MODEL_PRICING["default"])
    
    @staticmethod
    def estimate_cost_from_tokens(
        input_tokens: int,
        output_tokens: int,
        model: str,
        provider: str
    ) -> CostInfo:
        """
        Estimate cost from token counts.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model name
            provider: Provider name
            
        Returns:
            CostInfo with estimated costs
        """
        tokens = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens
        )
        return CostCalculator.calculate_cost(tokens, model, provider)
    
    @staticmethod
    def format_cost(cost_usd: float) -> str:
        """
        Format cost for display.
        
        Args:
            cost_usd: Cost in USD
            
        Returns:
            Formatted cost string
        """
        if cost_usd < 0.0001:
            return f"${cost_usd:.6f}"
        elif cost_usd < 0.01:
            return f"${cost_usd:.4f}"
        elif cost_usd < 1.0:
            return f"${cost_usd:.3f}"
        else:
            return f"${cost_usd:.2f}"


def get_cost_calculator() -> CostCalculator:
    """Get the global cost calculator instance."""
    return CostCalculator()


