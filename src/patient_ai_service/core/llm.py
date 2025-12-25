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

    def create_message_with_cache(
        self,
        system_cached: str,
        system_dynamic: str,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Create a message with prompt caching enabled.
        
        Args:
            system_cached: Static system content to cache (universal guides, tools)
            system_dynamic: Dynamic system content (agent-specific, not cached)
            messages: User messages
            max_tokens: Maximum output tokens
            temperature: Sampling temperature
            
        Returns:
            Response text
        """
        max_retries = 3
        base_delay = 1.0
        
        # Build system content with cache control
        system_content = [
            {
                "type": "text",
                "text": system_cached,
                "cache_control": {"type": "ephemeral"}
            }
        ]
        
        # Only add dynamic content if non-empty
        if system_dynamic and system_dynamic.strip():
            system_content.append({
                "type": "text",
                "text": system_dynamic
            })
        
        for attempt in range(max_retries + 1):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens or settings.llm_max_tokens,
                    temperature=temperature or settings.llm_temperature,
                    system=system_content,
                    messages=messages,
                )
                
                if response.content and len(response.content) > 0:
                    return response.content[0].text
                return ""
                
            except anthropic.OverloadedError as e:
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Anthropic API overloaded (attempt {attempt + 1}/{max_retries + 1}). Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    raise
            except anthropic.RateLimitError as e:
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Anthropic API rate limit hit (attempt {attempt + 1}/{max_retries + 1}). Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    raise
            except anthropic.APIError as e:
                logger.error(f"Anthropic API error: {e}")
                raise

    def create_message_with_cache_and_usage(
        self,
        system_cached: str,
        system_dynamic: str,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[str, TokenUsage]:
        """
        Create a message with prompt caching and return token usage details.
        
        Returns:
            Tuple of (response_text, TokenUsage with cache breakdown)
        """
        max_retries = 3
        base_delay = 1.0
        
        # Build system content with cache control
        system_content = [
            {
                "type": "text",
                "text": system_cached,
                "cache_control": {"type": "ephemeral"}
            }
        ]
        
        if system_dynamic and system_dynamic.strip():
            system_content.append({
                "type": "text",
                "text": system_dynamic
            })
        
        for attempt in range(max_retries + 1):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens or settings.llm_max_tokens,
                    temperature=temperature or settings.llm_temperature,
                    system=system_content,
                    messages=messages,
                )
                
                # Extract token usage with cache details
                usage = response.usage
                if not usage:
                    tokens = TokenUsage()
                    logger.warning("⚠️ No usage object in API response")
                else:
                    # VALIDATION: Check if usage object has cache-related attributes
                    usage_dict = usage.model_dump() if hasattr(usage, 'model_dump') else {}
                    usage_attrs = {k: getattr(usage, k, None) for k in dir(usage) if not k.startswith('_') and not callable(getattr(usage, k, None))}
                    
                    logger.info(f"🔍 [Cache Validation] Usage object type: {type(usage)}")
                    logger.info(f"🔍 [Cache Validation] Usage object dict: {usage_dict}")
                    logger.info(f"🔍 [Cache Validation] Usage object attributes: {usage_attrs}")
                    
                    # Extract cache tokens - try multiple methods
                    cache_creation = None
                    cache_read = None
                    
                    # Method 1: Direct attribute access
                    if hasattr(usage, 'cache_creation_input_tokens'):
                        cache_creation = getattr(usage, 'cache_creation_input_tokens')
                        logger.info(f"🔍 [Cache Validation] Found cache_creation_input_tokens attribute: {cache_creation}")
                    elif 'cache_creation_input_tokens' in usage_dict:
                        cache_creation = usage_dict['cache_creation_input_tokens']
                        logger.info(f"🔍 [Cache Validation] Found cache_creation_input_tokens in dict: {cache_creation}")
                    
                    if hasattr(usage, 'cache_read_input_tokens'):
                        cache_read = getattr(usage, 'cache_read_input_tokens')
                        logger.info(f"🔍 [Cache Validation] Found cache_read_input_tokens attribute: {cache_read}")
                    elif 'cache_read_input_tokens' in usage_dict:
                        cache_read = usage_dict['cache_read_input_tokens']
                        logger.info(f"🔍 [Cache Validation] Found cache_read_input_tokens in dict: {cache_read}")
                    
                    # Check if cache tokens are in a nested structure (e.g., usage.cache)
                    if hasattr(usage, 'cache'):
                        cache_obj = getattr(usage, 'cache')
                        logger.info(f"🔍 [Cache Validation] Found cache object: {cache_obj}")
                        if hasattr(cache_obj, 'creation_input_tokens'):
                            cache_creation = getattr(cache_obj, 'creation_input_tokens')
                            logger.info(f"🔍 [Cache Validation] Found cache.creation_input_tokens: {cache_creation}")
                        if hasattr(cache_obj, 'read_input_tokens'):
                            cache_read = getattr(cache_obj, 'read_input_tokens')
                            logger.info(f"🔍 [Cache Validation] Found cache.read_input_tokens: {cache_read}")
                    
                    # Convert None to 0, but preserve actual 0 values
                    cache_creation_tokens = cache_creation if cache_creation is not None else 0
                    cache_read_tokens = cache_read if cache_read is not None else 0
                    
                    tokens = TokenUsage(
                        input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
                        total_tokens=usage.input_tokens + usage.output_tokens,
                        cache_creation_input_tokens=cache_creation_tokens,
                        cache_read_input_tokens=cache_read_tokens,
                    )
                    
                    # Log cache token extraction for debugging
                    logger.info(f"📊 [Cache Validation] Token extraction - Input: {usage.input_tokens}, Output: {usage.output_tokens}")
                    logger.info(f"📊 [Cache Validation] Cache tokens - Creation: {cache_creation_tokens}, Read: {cache_read_tokens}")
                    logger.info(f"📊 [Cache Validation] Regular input tokens (calculated): {tokens.regular_input_tokens}")
                    
                    # Verify: input_tokens should equal regular + cache_creation + cache_read
                    expected_regular = usage.input_tokens - cache_creation_tokens - cache_read_tokens
                    if tokens.regular_input_tokens != expected_regular:
                        logger.warning(f"⚠️ [Cache Validation] Token calculation mismatch! Expected regular: {expected_regular}, got: {tokens.regular_input_tokens}")
                    
                    # Warn if cache tokens are 0 but we're using caching
                    if cache_creation_tokens == 0 and cache_read_tokens == 0:
                        logger.warning(f"⚠️ [Cache Validation] Cache tokens are both 0! This might indicate:")
                        logger.warning(f"⚠️ [Cache Validation]   1. Caching is not enabled in the API request")
                        logger.warning(f"⚠️ [Cache Validation]   2. This is the first request (cache not created yet)")
                        logger.warning(f"⚠️ [Cache Validation]   3. Cache tokens are not being returned by the API")
                        logger.warning(f"⚠️ [Cache Validation]   4. Cache token extraction is failing")
                
                text = ""
                if response.content and len(response.content) > 0:
                    text = response.content[0].text
                
                # Log cache status
                if tokens.cache_read_input_tokens > 0:
                    logger.info(f"🎯 Cache HIT: {tokens.cache_read_input_tokens} tokens read from cache")
                elif tokens.cache_creation_input_tokens > 0:
                    logger.info(f"📝 Cache WRITE: {tokens.cache_creation_input_tokens} tokens written to cache")
                
                return text, tokens
                
            except anthropic.OverloadedError as e:
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Anthropic API overloaded (attempt {attempt + 1}/{max_retries + 1}). Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    raise
            except anthropic.RateLimitError as e:
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Anthropic API rate limit hit (attempt {attempt + 1}/{max_retries + 1}). Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    raise
            except anthropic.APIError as e:
                logger.error(f"Anthropic API error: {e}")
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
        # Initialize OpenAI client with httpx client to avoid compatibility issues
        import httpx
        http_client = httpx.Client(timeout=30.0)
        self.client = openai.OpenAI(api_key=api_key, http_client=http_client)
        self.model = model
        logger.info(f"Initialized OpenAI client with model: {model}")

    def _requires_max_completion_tokens(self) -> bool:
        """
        Check if this model requires 'max_completion_tokens' instead of 'max_tokens'.
        
        Newer OpenAI models (gpt-5 series, o4 series) use max_completion_tokens.
        Older models (gpt-4o, gpt-4o-mini, etc.) use max_tokens.
        """
        # Models that require max_completion_tokens
        models_requiring_completion_tokens = [
            "gpt-5-mini-2025-08-07",
            "gpt-5-nano-2025-08-07",
            "o4-mini-2025-04-16",
            # Add other models that require max_completion_tokens as needed
        ]
        
        # Check if model name matches exactly or starts with gpt-5- or o4-
        if self.model in models_requiring_completion_tokens:
            return True
        
        # Check for gpt-5- prefix (for any gpt-5 model)
        if self.model.startswith("gpt-5-"):
            return True
        
        # Check for o4- prefix (for any o4 model)
        if self.model.startswith("o4-"):
            return True
        
        return False
    
    def _get_max_tokens_param(self, max_tokens: Optional[int]) -> Dict[str, Any]:
        """
        Get the appropriate max tokens parameter for this model.
        
        Returns empty dict - max_tokens is not sent with requests.
        """
        # Completely remove max_tokens from requests
        return {}
    
    def _requires_default_temperature_only(self) -> bool:
        """
        Check if this model only supports the default temperature value of 1.0.
        
        Newer OpenAI models (gpt-5 series, o4 series) only support temperature=1.0.
        Older models (gpt-4o, gpt-4o-mini, etc.) support any temperature value.
        """
        # Models that only support default temperature (1.0)
        models_requiring_default_temp = [
            "gpt-5-mini-2025-08-07",
            "gpt-5-nano-2025-08-07",
            "o4-mini-2025-04-16",
            # Add other models that require default temperature only as needed
        ]
        
        # Check if model name matches exactly
        if self.model in models_requiring_default_temp:
            return True
        
        # Check for gpt-5- prefix (for any gpt-5 model)
        if self.model.startswith("gpt-5-"):
            return True
        
        # Check for o4- prefix (for any o4 model)
        if self.model.startswith("o4-"):
            return True
        
        return False
    
    def _get_temperature_param(self, temperature: Optional[float]) -> Dict[str, Any]:
        """
        Get the appropriate temperature parameter for this model.
        
        For models that only support default temperature (1.0):
        - Always omit temperature parameter (return empty dict) to use default
        
        For other models:
        - Return temperature dict with the provided value or default from settings.
        """
        # Check if model only supports default temperature
        if self._requires_default_temperature_only():
            # Omit temperature parameter - model will use default (1.0)
            # This avoids errors when non-1.0 values are requested
            return {}
        
        # Get the actual temperature value to use for other models
        if temperature is None:
            temperature = settings.llm_temperature
        
        # Other models support any temperature value
        return {"temperature": temperature}

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

            # Get appropriate max tokens parameter for this model
            max_tokens_param = self._get_max_tokens_param(max_tokens)
            # Get appropriate temperature parameter for this model
            temperature_param = self._get_temperature_param(temperature)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                **temperature_param,
                **max_tokens_param
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

            # Get appropriate max tokens parameter for this model
            max_tokens_param = self._get_max_tokens_param(max_tokens)
            # Get appropriate temperature parameter for this model
            temperature_param = self._get_temperature_param(temperature)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                **temperature_param,
                **max_tokens_param
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

            # Get appropriate max tokens parameter for this model
            max_tokens_param = self._get_max_tokens_param(max_tokens)
            # Get appropriate temperature parameter for this model
            temperature_param = self._get_temperature_param(temperature)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                functions=functions,
                **temperature_param,
                **max_tokens_param
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

            # Get appropriate max tokens parameter for this model
            max_tokens_param = self._get_max_tokens_param(max_tokens)
            # Get appropriate temperature parameter for this model
            temperature_param = self._get_temperature_param(temperature)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                functions=functions,
                **temperature_param,
                **max_tokens_param
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
