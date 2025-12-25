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
    "gpt-5-mini-2025-08-07": {
        "input": 0.25,  # $0.25 per 1M input tokens
        "output": 2.0   # $2.00 per 1M output tokens
    },
    "gpt-5-nano-2025-08-07": {
        "input": 0.05,  # $0.05 per 1M input tokens
        "output": 0.4   # $0.40 per 1M output tokens
    },
    "o4-mini-2025-04-16": {
        "input": 1.10,  # $1.10 per 1M input tokens
        "output": 4.40  # $4.40 per 1M output tokens
    },
    # Default fallback pricing (use Haiku pricing as conservative estimate)
    "default": {
        "input": 0.25,
        "output": 1.25
    }
}

# Add/update the PRICING constant to include cache multipliers
ANTHROPIC_PRICING = {
    "claude-sonnet-4-5-20250929": {
        "input_per_million": 3.00,
        "output_per_million": 15.00,
        "cache_write_multiplier": 1.25,  # 25% premium for cache writes
        "cache_read_multiplier": 0.10,   # 90% discount for cache reads
    },
    "claude-3-5-sonnet-20241022": {
        "input_per_million": 3.00,
        "output_per_million": 15.00,
        "cache_write_multiplier": 1.25,
        "cache_read_multiplier": 0.10,
    },
    # Add other models as needed
}

OPENAI_PRICING = {
    "gpt-4o-mini": {
        "input_per_million": 0.15,
        "output_per_million": 0.60,
    },
    "gpt-4o": {
        "input_per_million": 2.50,
        "output_per_million": 10.00,
    },
    "gpt-4-turbo": {
        "input_per_million": 10.00,
        "output_per_million": 30.00,
    },
    "gpt-3.5-turbo": {
        "input_per_million": 0.50,
        "output_per_million": 1.50,
    },
    "gpt-5-mini-2025-08-07": {
        "input_per_million": 0.25,
        "output_per_million": 2.00,
    },
    "gpt-5-nano-2025-08-07": {
        "input_per_million": 0.05,
        "output_per_million": 0.40,
    },
    "o4-mini-2025-04-16": {
        "input_per_million": 1.10,
        "output_per_million": 4.40,
    },
}


class CostCalculator:
    """Calculate costs for LLM API calls with cache support."""
    
    def calculate_cost(
        self,
        tokens: TokenUsage,
        model: str,
        provider: str
    ) -> CostInfo:
        """
        Calculate cost with cache-aware pricing.
        
        Pricing breakdown:
        - Regular input tokens: standard input price
        - Cache creation tokens: input price × 1.25
        - Cache read tokens: input price × 0.10
        - Output tokens: standard output price
        """
        if provider.lower() == "anthropic":
            return self._calculate_anthropic_cost(tokens, model)
        elif provider.lower() == "openai":
            return self._calculate_openai_cost(tokens, model)
        else:
            return CostInfo(model=model, provider=provider)
    
    def _calculate_anthropic_cost(self, tokens: TokenUsage, model: str) -> CostInfo:
        """Calculate Anthropic costs with cache pricing."""
        pricing = ANTHROPIC_PRICING.get(model, ANTHROPIC_PRICING.get("claude-sonnet-4-5-20250929"))
        
        input_rate = pricing["input_per_million"] / 1_000_000
        output_rate = pricing["output_per_million"] / 1_000_000
        cache_write_mult = pricing.get("cache_write_multiplier", 1.25)
        cache_read_mult = pricing.get("cache_read_multiplier", 0.10)
        
        # Calculate each component
        regular_input_tokens = tokens.regular_input_tokens
        regular_input_cost = regular_input_tokens * input_rate
        
        cache_creation_cost = tokens.cache_creation_input_tokens * input_rate * cache_write_mult
        cache_read_cost = tokens.cache_read_input_tokens * input_rate * cache_read_mult
        
        output_cost = tokens.output_tokens * output_rate
        
        # Total input cost is sum of all input components
        total_input_cost = regular_input_cost + cache_creation_cost + cache_read_cost
        total_cost = total_input_cost + output_cost
        
        return CostInfo(
            input_cost_usd=round(total_input_cost, 6),
            output_cost_usd=round(output_cost, 6),
            total_cost_usd=round(total_cost, 6),
            model=model,
            provider="anthropic",
            cache_creation_cost_usd=round(cache_creation_cost, 6),
            cache_read_cost_usd=round(cache_read_cost, 6),
            regular_input_cost_usd=round(regular_input_cost, 6),
        )
    
    def _calculate_openai_cost(self, tokens: TokenUsage, model: str) -> CostInfo:
        """Calculate OpenAI costs (no caching support)."""
        # OpenAI doesn't have prompt caching, use standard pricing
        pricing = OPENAI_PRICING.get(model, {"input_per_million": 0.15, "output_per_million": 0.60})
        
        input_rate = pricing["input_per_million"] / 1_000_000
        output_rate = pricing["output_per_million"] / 1_000_000
        
        input_cost = tokens.input_tokens * input_rate
        output_cost = tokens.output_tokens * output_rate
        
        return CostInfo(
            input_cost_usd=round(input_cost, 6),
            output_cost_usd=round(output_cost, 6),
            total_cost_usd=round(input_cost + output_cost, 6),
            model=model,
            provider="openai",
            regular_input_cost_usd=round(input_cost, 6),
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
    
    def estimate_cost_from_tokens(
        self,
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
        return self.calculate_cost(tokens, model, provider)
    
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





