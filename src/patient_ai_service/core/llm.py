"""
LLM abstraction layer supporting multiple providers.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import anthropic
import openai

from .config import settings
from patient_ai_service.models.enums import LLMProvider
from patient_ai_service.models.observability import TokenUsage

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def create_message(
        self,
        system: str,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Create a message completion."""
        pass

    @abstractmethod
    def create_message_with_tools(
        self,
        system: str,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> tuple[str, Optional[Dict[str, Any]]]:
        """Create a message completion with tool calling support."""
        pass


class AnthropicClient(LLMClient):
    """Anthropic Claude client implementation."""

    def __init__(self, api_key: str, model: str):
        # Initialize Anthropic client without proxies to avoid httpx compatibility issues
        import httpx
        http_client = httpx.Client(timeout=30.0)
        self.client = anthropic.Anthropic(api_key=api_key, http_client=http_client)
        self.model = model
        logger.info(f"Initialized Anthropic client with model: {model}")

    def create_message(
        self,
        system: str,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Create a message using Claude."""
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries + 1):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens or settings.llm_max_tokens,
                    temperature=temperature or settings.llm_temperature,
                    system=system,
                    messages=messages,
                )

                # Extract text from response
                if response.content and len(response.content) > 0:
                    return response.content[0].text
                return ""

            except anthropic.OverloadedError as e:
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Anthropic API overloaded (attempt {attempt + 1}/{max_retries + 1}). Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"Anthropic API overloaded after {max_retries + 1} attempts: {e}")
                    raise
            except anthropic.RateLimitError as e:
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Anthropic API rate limit hit (attempt {attempt + 1}/{max_retries + 1}). Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"Anthropic API rate limit exceeded after {max_retries + 1} attempts: {e}")
                    raise
            except anthropic.APIError as e:
                logger.error(f"Anthropic API error: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error calling Anthropic: {e}")
                raise
    
    def create_message_with_usage(
        self,
        system: str,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[str, TokenUsage]:
        """Create a message using Claude and return text with token usage."""
        start_time = time.time()
        
        # Retry configuration for transient errors
        max_retries = 3
        base_delay = 1.0  # seconds
        
        for attempt in range(max_retries + 1):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens or settings.llm_max_tokens,
                    temperature=temperature or settings.llm_temperature,
                    system=system,
                    messages=messages,
                )

                # Extract token usage
                tokens = TokenUsage(
                    input_tokens=response.usage.input_tokens if response.usage else 0,
                    output_tokens=response.usage.output_tokens if response.usage else 0,
                    total_tokens=response.usage.input_tokens + response.usage.output_tokens if response.usage else 0
                )

                # Extract text from response
                text = ""
                if response.content and len(response.content) > 0:
                    text = response.content[0].text

                return text, tokens

            except anthropic.OverloadedError as e:
                if attempt < max_retries:
                    # Exponential backoff: 1s, 2s, 4s
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Anthropic API overloaded (attempt {attempt + 1}/{max_retries + 1}). Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"Anthropic API overloaded after {max_retries + 1} attempts: {e}")
                    raise
            except anthropic.RateLimitError as e:
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Anthropic API rate limit hit (attempt {attempt + 1}/{max_retries + 1}). Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"Anthropic API rate limit exceeded after {max_retries + 1} attempts: {e}")
                    raise
            except anthropic.APIError as e:
                logger.error(f"Anthropic API error: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error calling Anthropic: {e}")
                raise

    def create_message_with_tools(
        self,
        system: str,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> tuple[str, Optional[Dict[str, Any]]]:
        """Create a message with tool calling support."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens or settings.llm_max_tokens,
                temperature=temperature or settings.llm_temperature,
                system=system,
                messages=messages,
                tools=tools,
            )

            # Check if there's a tool use
            tool_use = None
            text_response = ""

            for block in response.content:
                if block.type == "text":
                    text_response += block.text
                elif block.type == "tool_use":
                    tool_use = {
                        "id": block.id,
                        "name": block.name,
                        "input": block.input
                    }

            return text_response, tool_use

        except anthropic.APIError as e:
            logger.error(f"Anthropic API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error calling Anthropic: {e}")
            raise
    
    def create_message_with_tools_and_usage(
        self,
        system: str,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[str, Optional[Dict[str, Any]], TokenUsage]:
        """Create a message with tool calling support and return token usage."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens or settings.llm_max_tokens,
                temperature=temperature or settings.llm_temperature,
                system=system,
                messages=messages,
                tools=tools,
            )

            # Extract token usage
            tokens = TokenUsage(
                input_tokens=response.usage.input_tokens if response.usage else 0,
                output_tokens=response.usage.output_tokens if response.usage else 0,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens if response.usage else 0
            )

            # Check if there's a tool use
            tool_use = None
            text_response = ""

            for block in response.content:
                if block.type == "text":
                    text_response += block.text
                elif block.type == "tool_use":
                    tool_use = {
                        "id": block.id,
                        "name": block.name,
                        "input": block.input
                    }

            return text_response, tool_use, tokens

        except anthropic.APIError as e:
            logger.error(f"Anthropic API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error calling Anthropic: {e}")
            raise


class OpenAIClient(LLMClient):
    """OpenAI GPT client implementation."""

    def __init__(self, api_key: str, model: str):
        openai.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        logger.info(f"Initialized OpenAI client with model: {model}")

    def create_message(
        self,
        system: str,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Create a message using GPT."""
        try:
            # Prepend system message
            full_messages = [{"role": "system", "content": system}] + messages

            response = self.client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                max_tokens=max_tokens or settings.llm_max_tokens,
                temperature=temperature or settings.llm_temperature,
            )

            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content or ""
            return ""

        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error calling OpenAI: {e}")
            raise
    
    def create_message_with_usage(
        self,
        system: str,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[str, TokenUsage]:
        """Create a message using GPT and return text with token usage."""
        try:
            # Prepend system message
            full_messages = [{"role": "system", "content": system}] + messages

            response = self.client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                max_tokens=max_tokens or settings.llm_max_tokens,
                temperature=temperature or settings.llm_temperature,
            )

            # Extract token usage
            tokens = TokenUsage(
                input_tokens=response.usage.prompt_tokens if response.usage else 0,
                output_tokens=response.usage.completion_tokens if response.usage else 0,
                total_tokens=response.usage.total_tokens if response.usage else 0
            )

            text = ""
            if response.choices and len(response.choices) > 0:
                text = response.choices[0].message.content or ""

            return text, tokens

        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error calling OpenAI: {e}")
            raise

    def create_message_with_tools(
        self,
        system: str,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> tuple[str, Optional[Dict[str, Any]]]:
        """Create a message with tool calling support (function calling)."""
        try:
            # Convert Anthropic tool format to OpenAI function format
            functions = self._convert_tools_to_functions(tools)

            # Prepend system message
            full_messages = [{"role": "system", "content": system}] + messages

            response = self.client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                functions=functions,
                max_tokens=max_tokens or settings.llm_max_tokens,
                temperature=temperature or settings.llm_temperature,
            )

            message = response.choices[0].message

            # Check for function call
            tool_use = None
            if message.function_call:
                import json
                tool_use = {
                    "name": message.function_call.name,
                    "input": json.loads(message.function_call.arguments)
                }

            return message.content or "", tool_use

        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error calling OpenAI: {e}")
            raise
    
    def create_message_with_tools_and_usage(
        self,
        system: str,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[str, Optional[Dict[str, Any]], TokenUsage]:
        """Create a message with tool calling support and return token usage."""
        try:
            # Convert Anthropic tool format to OpenAI function format
            functions = self._convert_tools_to_functions(tools)

            # Prepend system message
            full_messages = [{"role": "system", "content": system}] + messages

            response = self.client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                functions=functions,
                max_tokens=max_tokens or settings.llm_max_tokens,
                temperature=temperature or settings.llm_temperature,
            )

            # Extract token usage
            tokens = TokenUsage(
                input_tokens=response.usage.prompt_tokens if response.usage else 0,
                output_tokens=response.usage.completion_tokens if response.usage else 0,
                total_tokens=response.usage.total_tokens if response.usage else 0
            )

            message = response.choices[0].message

            # Check for function call
            tool_use = None
            if message.function_call:
                import json
                tool_use = {
                    "name": message.function_call.name,
                    "input": json.loads(message.function_call.arguments)
                }

            return message.content or "", tool_use, tokens

        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error calling OpenAI: {e}")
            raise

    def _convert_tools_to_functions(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert Anthropic tool format to OpenAI function format."""
        functions = []
        for tool in tools:
            function = {
                "name": tool.get("name"),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {})
            }
            functions.append(function)
        return functions


class LLMFactory:
    """Factory for creating LLM clients."""

    @staticmethod
    def create_client(
        provider: Optional[LLMProvider] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ) -> LLMClient:
        """Create an LLM client based on provider."""
        provider = provider or settings.llm_provider
        api_key = api_key or settings.get_llm_api_key()
        model = model or settings.get_llm_model()

        if not api_key:
            raise ValueError(f"API key required for provider: {provider}")

        if provider == LLMProvider.ANTHROPIC:
            return AnthropicClient(api_key=api_key, model=model)
        elif provider == LLMProvider.OPENAI:
            return OpenAIClient(api_key=api_key, model=model)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")


# Global LLM client instance
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get or create the global LLM client instance."""
    global _llm_client
    if _llm_client is None:
        settings.validate_llm_config()
        _llm_client = LLMFactory.create_client()
    return _llm_client


def reset_llm_client():
    """Reset the global LLM client (useful for testing)."""
    global _llm_client
    _llm_client = None
