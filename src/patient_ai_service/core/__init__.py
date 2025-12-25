"""
Core components for the dental clinic AI system.
"""

from .config import settings, Settings
from .llm import (
    LLMClient,
    AnthropicClient,
    OpenAIClient,
    LLMFactory,
    get_llm_client,
    reset_llm_client,
)
from .state_manager import (
    StateManager,
    StateBackend,
    InMemoryBackend,
    RedisBackend,
    get_state_manager,
    reset_state_manager,
)
from .message_broker import (
    MessageBroker,
    get_message_broker,
    reset_message_broker,
)
from .conversation_memory import (
    ConversationMemory,
    ConversationMemoryManager,
    ConversationTurn,
    get_conversation_memory_manager,
    reset_conversation_memory_manager,
)
from .reasoning import (
    ReasoningEngine,
    ReasoningOutput,
    UnderstandingResult,
    RoutingResult,
    MemoryUpdate,
    ResponseGuidance,
    get_reasoning_engine,
    reset_reasoning_engine,
)
from .unified_reasoning import (
    UnifiedReasoning,
    get_unified_reasoning,
    reset_unified_reasoning,
)
from .llm_config import (
    LLMConfig,
    LLMConfigManager,
    get_llm_config_manager,
    get_llm_config,
    get_hierarchical_llm_client,
    reset_llm_config_manager,
)
from .prompt_cache import (
    get_universal_system_content,
    get_cache_config,
    set_cache_enabled,
    CacheConfig,
)

__all__ = [
    # Config
    "settings",
    "Settings",
    # LLM
    "LLMClient",
    "AnthropicClient",
    "OpenAIClient",
    "LLMFactory",
    "get_llm_client",
    "reset_llm_client",
    # State
    "StateManager",
    "StateBackend",
    "InMemoryBackend",
    "RedisBackend",
    "get_state_manager",
    "reset_state_manager",
    # Messaging
    "MessageBroker",
    "get_message_broker",
    "reset_message_broker",
    # Conversation Memory
    "ConversationMemory",
    "ConversationMemoryManager",
    "ConversationTurn",
    "get_conversation_memory_manager",
    "reset_conversation_memory_manager",
    # Reasoning
    "ReasoningEngine",
    "ReasoningOutput",
    "UnderstandingResult",
    "RoutingResult",
    "MemoryUpdate",
    "ResponseGuidance",
    "get_reasoning_engine",
    "reset_reasoning_engine",
    # Unified Reasoning
    "UnifiedReasoning",
    "get_unified_reasoning",
    "reset_unified_reasoning",
    # LLM Configuration
    "LLMConfig",
    "LLMConfigManager",
    "get_llm_config_manager",
    "get_llm_config",
    "get_hierarchical_llm_client",
    "reset_llm_config_manager",
    # Prompt Caching
    "get_universal_system_content",
    "get_cache_config",
    "set_cache_enabled",
    "CacheConfig",
]
