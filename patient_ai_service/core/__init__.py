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
]
