"""
Base Agent class for all specialized agents.

Provides common functionality including:
- LLM interaction
- Tool execution
- State management
- Conversation history
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime

from patient_ai_service.core import get_llm_client, get_state_manager
from patient_ai_service.core.llm import LLMClient
from patient_ai_service.core.state_manager import StateManager
from patient_ai_service.core.observability import get_observability_logger
from patient_ai_service.core.config import settings
from patient_ai_service.models.validation import ExecutionLog, ToolExecution
from patient_ai_service.models.observability import (
    LLMCall,
    ToolExecutionDetail,
    AgentExecutionDetails,
    AgentContext,
    TokenUsage,
    CostInfo
)

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all agents.

    Handles:
    - LLM communication
    - Tool/action execution
    - Conversation history management
    - State integration
    """

    def __init__(
        self,
        agent_name: str,
        llm_client: Optional[LLMClient] = None,
        state_manager: Optional[StateManager] = None
    ):
        self.agent_name = agent_name
        self.llm_client = llm_client or get_llm_client()
        self.state_manager = state_manager or get_state_manager()

        # Conversation history per session
        self.conversation_history: Dict[str, List[Dict[str, str]]] = {}

        # Minimal context from reasoning engine (per session)
        self._context: Dict[str, Dict[str, Any]] = {}

        # Temporary storage for execution log during tool execution
        # Note: Execution log is now passed from orchestrator, not created here
        # This dict is only used temporarily to pass log to _execute_tool()
        self._execution_log: Dict[str, ExecutionLog] = {}

        # Tool registry
        self._tools: Dict[str, Callable] = {}
        self._tool_schemas: List[Dict[str, Any]] = []

        # Register agent-specific tools
        self._register_tools()

        logger.info(f"Initialized {self.agent_name} agent")

    async def on_activated(self, session_id: str, reasoning: Any):
        """
        Called when agent is selected for a session.
        Override in subclasses to set up necessary state.

        Args:
            session_id: Session identifier
            reasoning: ReasoningOutput from reasoning engine

        Default implementation does nothing.
        """
        pass

    def set_context(self, session_id: str, context: Dict[str, Any]):
        """
        Set minimal context for this session.

        Args:
            session_id: Session identifier
            context: Minimal context dict from reasoning engine
        """
        self._context[session_id] = context
        logger.debug(f"Set context for {self.agent_name} session {session_id}: {context}")
        
        # Record agent context in observability
        if settings.enable_observability:
            obs_logger = get_observability_logger(session_id)
            system_prompt = self._get_system_prompt(session_id)
            obs_logger.record_agent_context(
                agent_name=self.agent_name,
                minimal_context=context,
                conversation_history_length=len(self.conversation_history.get(session_id, [])),
                system_prompt_preview=system_prompt[:200] if system_prompt else ""
            )

    def _get_context_note(self, session_id: str) -> str:
        """
        Generate a brief context note for the system prompt.

        IMPORTANT: Now includes language context received from reasoning engine.

        Args:
            session_id: Session identifier

        Returns:
            Brief context note string, or empty string if no context
        """
        context = self._context.get(session_id, {})
        if not context:
            return ""

        # Build minimal context note - just essentials
        parts = []

        if "user_wants" in context:
            parts.append(f"User wants: {context['user_wants']}")

        if "action" in context:
            parts.append(f"Suggested action: {context['action']}")

        if "prior_context" in context:
            parts.append(f"Context: {context['prior_context']}")

        # [NEW] Language context (received from reasoning engine via minimal_context)
        current_language = context.get("current_language")
        current_dialect = context.get("current_dialect")

        if current_language:
            lang_display = f"{current_language}"
            if current_dialect:
                lang_display += f"-{current_dialect}"

            parts.append(f"User's language: {lang_display}")

            # Check if this is a language switch
            # Note: We could check global_state.language_context.language_history
            # but for simplicity, we just note the current language
            if current_language != "en":
                parts.append(
                    f"Note: User speaks {lang_display}. "
                    f"Messages you receive are translated to English. "
                    f"Your responses will be translated back to {lang_display}."
                )

        if not parts:
            return ""

        # Return formatted context note
        return "\n[CONVERSATION CONTEXT]\n" + "\n".join(parts) + "\n"

    @abstractmethod
    def _get_system_prompt(self, session_id: str) -> str:
        """
        Generate system prompt with current context.

        Must be implemented by subclasses to provide agent-specific
        instructions and context.
        """
        pass

    @abstractmethod
    def _register_tools(self):
        """
        Register agent-specific tools.

        Must be implemented by subclasses to define available actions.
        """
        pass

    def register_tool(
        self,
        name: str,
        function: Callable,
        description: str,
        parameters: Dict[str, Any]
    ):
        """
        Register a tool/action for this agent.

        Args:
            name: Tool name
            function: Python function to execute
            description: Tool description for LLM
            parameters: JSON schema for parameters
        """
        self._tools[name] = function

        # Create tool schema for LLM
        schema = {
            "name": name,
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": parameters,
                "required": list(parameters.keys())
            }
        }
        self._tool_schemas.append(schema)

        logger.debug(f"Registered tool '{name}' for {self.agent_name}")

    def _should_auto_book_appointment(
        self,
        session_id: str,
        tool_name: str,
        tool_result: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Check if automatic appointment booking should be triggered.
        
        This enforces booking at the code level when check_availability
        returns MANDATORY_ACTION, preventing false confirmations.
        
        Args:
            session_id: Session identifier
            tool_name: Name of the tool that was just executed
            tool_result: Result from the tool execution
        
        Returns:
            Booking parameters if auto-booking should happen, None otherwise.
        """
        # Only trigger for check_availability tool
        if tool_name != "check_availability":
            logger.debug(f"Auto-booking check: tool_name is '{tool_name}', not 'check_availability'")
            return None
        
        # Check if tool result has MANDATORY_ACTION
        logger.info(f"🔍 Auto-booking check: tool_result keys = {list(tool_result.keys())[:10]}")
        mandatory_action = tool_result.get("MANDATORY_ACTION")
        available_at_time = tool_result.get("available_at_requested_time")
        logger.info(f"🔍 Auto-booking check: MANDATORY_ACTION = '{mandatory_action}', available_at_requested_time = {available_at_time}")
        if mandatory_action != "CALL book_appointment TOOL IMMEDIATELY":
            logger.info(f"⚠️ Auto-booking skipped: MANDATORY_ACTION mismatch (got '{mandatory_action}')")
            return None
        
        # Check if patient is registered
        global_state = self.state_manager.get_global_state(session_id)
        patient = global_state.patient_profile
        
        if not patient or not patient.patient_id:
            logger.info("Auto-booking skipped: Patient not registered")
            return None
        
        # Extract required parameters from tool result
        required_params = tool_result.get("required_parameters", {})
        
        # Get doctor_id from recent list_doctors result in conversation history
        # Patient mentions doctor by name (e.g., "Dr. Smith"), not by ID
        # So we need to find the list_doctors result and match the doctor name
        logger.info(f"🔍 Auto-booking: Looking for doctor_id from list_doctors result...")
        doctor_id = None
        history = self.conversation_history.get(session_id, [])
        logger.info(f"🔍 Auto-booking: Checking {len(history)} messages in history")

        # First, find the most recent list_doctors result
        doctors_list = None
        for msg in reversed(history):
            if "Tool result:" in msg.get("content", ""):
                try:
                    result_str = msg["content"].replace("Tool result: ", "")
                    result_data = json.loads(result_str)
                    # Check if this is a list_doctors result
                    if result_data.get("success") and "doctors" in result_data:
                        doctors_list = result_data.get("doctors", [])
                        logger.info(f"✅ Auto-booking: Found list_doctors result with {len(doctors_list)} doctors")
                        break
                except Exception as e:
                    logger.debug(f"Error parsing tool result: {e}")
                    continue

        if not doctors_list:
            logger.warning("⚠️ Auto-booking skipped: Could not find list_doctors result in history")
            return None

        # Now, extract doctor name from user messages
        # Look for doctor name in recent user messages
        doctor_name_mentioned = None
        for msg in reversed(history):
            if msg.get("role") == "user":
                content = msg.get("content", "").lower()
                # Split content into words for word-boundary matching
                content_words = content.split()

                # Check each doctor in the list to see if their name is mentioned
                for doctor in doctors_list:
                    doctor_full_name = doctor.get("name", "").lower()
                    matched = False

                    # Extract first and last name from "Dr. FirstName LastName"
                    if " " in doctor_full_name:
                        parts = doctor_full_name.split()
                        last_name = parts[-1]
                        # First name is the word after "dr." if it exists
                        first_name = parts[1] if len(parts) > 1 else ""

                        # Check if user mentioned the doctor's name (word-boundary matching)
                        # This prevents false positives like "ali" matching "allergy"
                        if (last_name in content_words or
                            first_name in content_words or
                            doctor_full_name in content):
                            matched = True
                    else:
                        # Single name doctor (no space) - check full name
                        if doctor_full_name in content_words or doctor_full_name in content:
                            matched = True

                    if matched:
                        doctor_name_mentioned = doctor.get("name")
                        doctor_id = doctor.get("id")
                        logger.info(f"✅ Auto-booking: Found doctor '{doctor_name_mentioned}' (ID: {doctor_id}) in user message: '{content[:100]}'")
                        break

                if doctor_id:
                    break

        if not doctor_id:
            logger.warning("⚠️ Auto-booking skipped: Could not match doctor name from user messages to list_doctors result")
            return None
        
        # Build booking parameters
        booking_params = {
            "session_id": session_id,
            "patient_id": patient.patient_id,
            "doctor_id": doctor_id,
            "date": tool_result.get("date"),
            "time": tool_result.get("requested_time"),
            "reason": "general consultation"  # Default reason
        }
        
        # Validate all required parameters are present
        if not all([
            booking_params["patient_id"],
            booking_params["doctor_id"],
            booking_params["date"],
            booking_params["time"]
        ]):
            logger.warning(f"Auto-booking skipped: Missing parameters: {booking_params}")
            return None
        
        logger.info(f"🤖 AUTO-BOOKING TRIGGERED: {booking_params}")
        return booking_params

    def _is_likely_english(self, text: str) -> bool:
        """
        Quick heuristic check if text is likely English.

        This is a safety check to catch obvious non-English text that
        might have bypassed translation.

        Args:
            text: Text to check

        Returns:
            True if likely English, False otherwise
        """
        # Check for non-Latin scripts (Arabic, Chinese, etc.)
        # These would indicate translation failed
        non_latin_ranges = [
            (0x0600, 0x06FF),  # Arabic
            (0x0750, 0x077F),  # Arabic Supplement
            (0x4E00, 0x9FFF),  # CJK Unified Ideographs
            (0x3040, 0x309F),  # Hiragana
            (0x30A0, 0x30FF),  # Katakana
            (0x0400, 0x04FF),  # Cyrillic
        ]

        for char in text[:100]:  # Check first 100 chars
            code_point = ord(char)
            for start, end in non_latin_ranges:
                if start <= code_point <= end:
                    return False

        # If we get here, text uses Latin/common scripts
        return True

    async def _emergency_translate(self, session_id: str, text: str) -> str:
        """
        Emergency fallback translation if non-English text reaches agent.

        This should rarely be needed if orchestrator works correctly.

        Args:
            session_id: Session identifier
            text: Text to translate

        Returns:
            Translated English text, or original if translation fails
        """
        logger.error(
            f"TRANSLATION BARRIER BREACH: Non-English text reached {self.agent_name} agent. "
            f"Text: '{text[:100]}...'"
        )

        try:
            # Get translation agent from orchestrator
            # Note: This is a fallback - ideally shouldn't happen
            from patient_ai_service.agents import TranslationAgent
            translation_agent = TranslationAgent()

            # Detect and translate
            detected_lang, detected_dialect = await translation_agent.detect_language_and_dialect(text)
            if detected_lang != "en":
                english_text = await translation_agent.translate_to_english_with_dialect(
                    text,
                    detected_lang,
                    detected_dialect
                )
                logger.info(f"Emergency translation succeeded: {detected_lang}-{detected_dialect} → en")
                return english_text
            else:
                # False alarm - text was English
                return text

        except Exception as e:
            logger.error(f"Emergency translation failed: {e}")
            # Return original and hope for the best
            return text

    async def process_message_with_log(
        self,
        session_id: str,
        user_message: str,
        execution_log: ExecutionLog
    ) -> Tuple[str, ExecutionLog]:
        """
        Process message and append tool executions to provided execution log.

        This method receives an execution log from the orchestrator, appends
        tool executions during message processing, and returns the updated log.

        Args:
            session_id: Session identifier
            user_message: User's input message
            execution_log: Execution log to append tool executions to

        Returns:
            Tuple of (response, execution_log with appended tools)
        """
        # Store execution_log temporarily for _execute_tool() access
        # This is needed because _execute_tool() is called from process_message()
        # which doesn't have direct access to execution_log parameter
        self._execution_log[session_id] = execution_log

        # Call existing process_message (which calls _execute_tool)
        response = await self.process_message(session_id, user_message)

        # Return the log (now with tools appended)
        return response, execution_log

    async def process_message(self, session_id: str, user_message: str) -> str:
        """
        Process a user message and return a response.

        This is the main entry point for agent interaction.

        IMPORTANT: This method expects user_message to be in ENGLISH.
        Translation should happen at the orchestrator level before calling this method.

        Args:
            session_id: Session identifier
            user_message: User's input message (expected to be English)

        Returns:
            Agent's response message
        """
        agent_start_time = time.time()
        llm_calls = []
        tool_executions = []
        obs_logger = get_observability_logger(session_id) if settings.enable_observability else None

        try:
            # [NEW] Translation barrier validation
            # Check if message is English (safety check)
            if not self._is_likely_english(user_message):
                logger.warning(
                    f"Non-English message detected in {self.agent_name} agent: {user_message[:50]}..."
                )
                # Attempt emergency translation
                user_message = await self._emergency_translate(session_id, user_message)

            # Initialize conversation history if needed
            if session_id not in self.conversation_history:
                self.conversation_history[session_id] = []

            # Add user message to history
            self.conversation_history[session_id].append({
                "role": "user",
                "content": user_message
            })

            # Get system prompt with context
            system_prompt = self._get_system_prompt(session_id)

            # Call LLM with token tracking
            llm_start_time = time.time()
            if self._tool_schemas:
                # Use tools if available
                if hasattr(self.llm_client, 'create_message_with_tools_and_usage'):
                    response_text, tool_use, tokens = self.llm_client.create_message_with_tools_and_usage(
                        system=system_prompt,
                        messages=self.conversation_history[session_id],
                        tools=self._tool_schemas
                    )
                else:
                    response_text, tool_use = self.llm_client.create_message_with_tools(
                        system=system_prompt,
                        messages=self.conversation_history[session_id],
                        tools=self._tool_schemas
                    )
                    tokens = TokenUsage()  # Fallback if method not available
            else:
                # No tools
                if hasattr(self.llm_client, 'create_message_with_usage'):
                    response_text, tokens = self.llm_client.create_message_with_usage(
                        system=system_prompt,
                        messages=self.conversation_history[session_id]
                    )
                    tool_use = None
                else:
                    response_text = self.llm_client.create_message(
                        system=system_prompt,
                        messages=self.conversation_history[session_id]
                    )
                    tokens = TokenUsage()  # Fallback if method not available
                    tool_use = None
            
            llm_duration_ms = (time.time() - llm_start_time) * 1000
            
            # Record LLM call
            if obs_logger:
                llm_call = obs_logger.record_llm_call(
                    component="agent",
                    provider=settings.llm_provider.value,
                    model=settings.get_llm_model(),
                    tokens=tokens,
                    duration_ms=llm_duration_ms,
                    system_prompt_length=len(system_prompt),
                    messages_count=len(self.conversation_history[session_id]),
                    temperature=settings.llm_temperature,
                    max_tokens=settings.llm_max_tokens
                )
                if llm_call:
                    llm_calls.append(llm_call)

            # Handle tool calls
            if tool_use:
                logger.info(f"Tool call requested: {tool_use.get('name')}")

                # Add assistant response to history
                # Only add if there's actual text - don't add "Using tool: X" messages
                # System prompts handle preventing premature responses, so we trust the LLM output
                if response_text and response_text.strip() and not response_text.startswith("Using tool:"):
                    self.conversation_history[session_id].append({
                        "role": "assistant",
                        "content": response_text
                    })
                # If no response_text, don't add anything - the tool result will be added next

                # Execute tool
                tool_result = await self._execute_tool(
                    session_id=session_id,
                    tool_name=tool_use.get('name'),
                    tool_input=tool_use.get('input', {})
                )

                # Add tool result to history
                tool_result_message = {
                    "role": "user",
                    "content": f"Tool result: {json.dumps(tool_result)}"
                }
                self.conversation_history[session_id].append(tool_result_message)

                # ========== AUTO-BOOKING ENFORCEMENT ==========
                # Check if automatic booking should be triggered
                logger.info(f"🔍 AUTO-BOOKING CHECK: tool_name='{tool_use.get('name')}', checking if auto-booking should trigger...")
                auto_booking_params = self._should_auto_book_appointment(
                    session_id=session_id,
                    tool_name=tool_use.get('name'),
                    tool_result=tool_result
                )
                logger.info(f"🔍 AUTO-BOOKING RESULT: {auto_booking_params is not None} (params: {auto_booking_params})")

                if auto_booking_params:
                    # Automatically execute book_appointment without waiting for LLM
                    logger.info("🤖 ENFORCING AUTOMATIC BOOKING (LLM override)")
                    
                    # Force the next tool call to be book_appointment
                    next_tool_use = {
                        "name": "book_appointment",
                        "input": auto_booking_params
                    }
                    
                    # Skip LLM call for next tool decision
                    final_response = ""
                else:
                    # Normal flow: Ask LLM if another tool call is needed
                    llm_start_time = time.time()
                    if self._tool_schemas:
                        if hasattr(self.llm_client, 'create_message_with_tools_and_usage'):
                            final_response, next_tool_use, tokens = self.llm_client.create_message_with_tools_and_usage(
                                system=system_prompt,
                                messages=self.conversation_history[session_id],
                                tools=self._tool_schemas
                            )
                        else:
                            final_response, next_tool_use = self.llm_client.create_message_with_tools(
                                system=system_prompt,
                                messages=self.conversation_history[session_id],
                                tools=self._tool_schemas
                            )
                            tokens = TokenUsage()
                    else:
                        if hasattr(self.llm_client, 'create_message_with_usage'):
                            final_response, tokens = self.llm_client.create_message_with_usage(
                                system=system_prompt,
                                messages=self.conversation_history[session_id]
                            )
                            next_tool_use = None
                        else:
                            final_response = self.llm_client.create_message(
                                system=system_prompt,
                                messages=self.conversation_history[session_id]
                            )
                            tokens = TokenUsage()
                            next_tool_use = None
                    
                    llm_duration_ms = (time.time() - llm_start_time) * 1000
                    if obs_logger:
                        llm_call = obs_logger.record_llm_call(
                            component="agent",
                            provider=settings.llm_provider.value,
                            model=settings.get_llm_model(),
                            tokens=tokens,
                            duration_ms=llm_duration_ms,
                            system_prompt_length=len(system_prompt),
                            messages_count=len(self.conversation_history[session_id]),
                            temperature=settings.llm_temperature,
                            max_tokens=settings.llm_max_tokens
                        )
                        if llm_call:
                            llm_calls.append(llm_call)
                # ========== END AUTO-BOOKING ENFORCEMENT ==========

                # If another tool is needed, execute it (chained tool call)
                if next_tool_use:
                    logger.info(f"Chained tool call requested: {next_tool_use.get('name')}")
                    
                    # Don't add intermediate responses that contain tool results - they're for internal use only
                    # Only add natural language responses that don't expose tool internals
                    # System prompts handle preventing premature responses, so we trust the LLM output
                    if final_response and final_response.strip() and not final_response.startswith("Using tool") and "Tool result:" not in final_response:
                        self.conversation_history[session_id].append({
                            "role": "assistant",
                            "content": final_response
                        })
                    
                    # Execute next tool
                    next_tool_result = await self._execute_tool(
                        session_id=session_id,
                        tool_name=next_tool_use.get('name'),
                        tool_input=next_tool_use.get('input', {})
                    )
                    
                    # Add next tool result
                    next_tool_result_message = {
                        "role": "user",
                        "content": f"Tool result: {json.dumps(next_tool_result)}"
                    }
                    self.conversation_history[session_id].append(next_tool_result_message)
                    
                    # ========== AUTO-BOOKING ENFORCEMENT (for chained tool calls) ==========
                    # Check if automatic booking should be triggered after chained tool call
                    logger.info(f"🔍 AUTO-BOOKING CHECK (chained): tool_name='{next_tool_use.get('name')}', checking if auto-booking should trigger...")
                    auto_booking_params_chained = self._should_auto_book_appointment(
                        session_id=session_id,
                        tool_name=next_tool_use.get('name'),
                        tool_result=next_tool_result
                    )
                    logger.info(f"🔍 AUTO-BOOKING RESULT (chained): {auto_booking_params_chained is not None} (params: {auto_booking_params_chained})")
                    
                    if auto_booking_params_chained:
                        # Automatically execute book_appointment without waiting for LLM
                        logger.info("🤖 ENFORCING AUTOMATIC BOOKING (LLM override) - chained tool call")
                        book_appointment_result = await self._execute_tool(
                            session_id=session_id,
                            tool_name="book_appointment",
                            tool_input=auto_booking_params_chained
                        )
                        # Add auto-booking tool result to history
                        self.conversation_history[session_id].append({
                            "role": "user",
                            "content": f"Tool result: {json.dumps(book_appointment_result)}"
                        })
                        # Force LLM to generate a final confirmation message after auto-booking
                        clean_messages = [
                            msg for msg in self.conversation_history[session_id]
                            if not msg.get("content", "").startswith("Tool result:")
                        ]
                        assistant_message = self.llm_client.create_message(
                            system=system_prompt + "\n\nCRITICAL: Provide ONLY a natural language confirmation response for the appointment. Do NOT include tool results, JSON, or technical details. Just provide a friendly confirmation message.",
                            messages=clean_messages
                        )
                        # Skip further processing as booking is complete
                        return assistant_message
                    # ========== END AUTO-BOOKING ENFORCEMENT (chained) ==========
                    
                    # Get final response after all tools
                    # Filter out tool result messages from conversation history for final response
                    clean_messages = [
                        msg for msg in self.conversation_history[session_id]
                        if not msg.get("content", "").startswith("Tool result:")
                    ]
                    
                    llm_start_time = time.time()
                    if hasattr(self.llm_client, 'create_message_with_usage'):
                        assistant_message, tokens = self.llm_client.create_message_with_usage(
                            system=system_prompt + "\n\nCRITICAL: Provide ONLY a natural language response. Do NOT include tool results, JSON, or technical details. Just provide a friendly confirmation message.",
                            messages=clean_messages
                        )
                    else:
                        assistant_message = self.llm_client.create_message(
                            system=system_prompt + "\n\nCRITICAL: Provide ONLY a natural language response. Do NOT include tool results, JSON, or technical details. Just provide a friendly confirmation message.",
                            messages=clean_messages
                        )
                        tokens = TokenUsage()
                    
                    llm_duration_ms = (time.time() - llm_start_time) * 1000
                    if obs_logger:
                        llm_call = obs_logger.record_llm_call(
                            component="agent",
                            provider=settings.llm_provider.value,
                            model=settings.get_llm_model(),
                            tokens=tokens,
                            duration_ms=llm_duration_ms,
                            system_prompt_length=len(system_prompt),
                            messages_count=len(clean_messages),
                            temperature=settings.llm_temperature,
                            max_tokens=settings.llm_max_tokens
                        )
                        if llm_call:
                            llm_calls.append(llm_call)
                    
                    # Clean up any tool result JSON that might have leaked into the response
                    if "Tool result:" in assistant_message:
                        # Extract only the natural language part before "Tool result:"
                        parts = assistant_message.split("Tool result:")
                        if parts:
                            assistant_message = parts[0].strip()
                            # If there's no natural language, get a clean response
                            if not assistant_message:
                                assistant_message = self.llm_client.create_message(
                                    system=system_prompt + "\n\nIMPORTANT: Provide ONLY a natural language response. Do NOT include tool results, JSON, or technical details in your response. Just provide a friendly confirmation message.",
                                    messages=clean_messages
                                )
                else:
                    assistant_message = final_response
            else:
                assistant_message = response_text

            # Add assistant response to history
            self.conversation_history[session_id].append({
                "role": "assistant",
                "content": assistant_message
            })

            # Limit history size (keep last 20 messages)
            if len(self.conversation_history[session_id]) > 20:
                self.conversation_history[session_id] = \
                    self.conversation_history[session_id][-20:]

            # Build agent execution details for observability
            if obs_logger:
                agent_duration_ms = (time.time() - agent_start_time) * 1000
                
                # Get tool executions from tracker
                tool_executions = obs_logger.tool_tracker.get_executions()
                
                # Calculate totals
                total_tokens = TokenUsage()
                total_cost = CostInfo()
                for llm_call in llm_calls:
                    total_tokens += llm_call.tokens
                    total_cost += llm_call.cost
                
                # Get agent context
                context = self._context.get(session_id, {})
                system_prompt = self._get_system_prompt(session_id)
                agent_context = AgentContext(
                    session_id=session_id,
                    agent_name=self.agent_name,
                    minimal_context=context,
                    conversation_history_length=len(self.conversation_history[session_id]),
                    system_prompt_preview=system_prompt[:200] if system_prompt else ""
                )
                
                agent_execution = AgentExecutionDetails(
                    agent_name=self.agent_name,
                    context=agent_context,
                    llm_calls=llm_calls,
                    tool_executions=tool_executions,
                    total_tokens=total_tokens,
                    total_cost=total_cost,
                    duration_ms=agent_duration_ms,
                    response_preview=assistant_message[:200] if assistant_message else ""
                )
                
                obs_logger.set_agent_execution(agent_execution)

            return assistant_message

        except Exception as e:
            logger.error(f"Error in {self.agent_name}.process_message: {e}", exc_info=True)
            return self._get_error_response(str(e))

    async def _execute_tool(
        self,
        session_id: str,
        tool_name: str,
        tool_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a tool/action.

        Args:
            session_id: Session identifier
            tool_name: Name of tool to execute
            tool_input: Tool parameters

        Returns:
            Tool execution result
        """
        tool_start_time = time.time()
        obs_logger = get_observability_logger(session_id) if settings.enable_observability else None
        
        if tool_name not in self._tools:
            logger.error(f"Unknown tool: {tool_name}")
            error_result = {"error": f"Unknown tool: {tool_name}"}
            if obs_logger:
                obs_logger.record_tool_execution(
                    tool_name=tool_name,
                    inputs=tool_input,
                    outputs=error_result,
                    duration_ms=(time.time() - tool_start_time) * 1000,
                    success=False,
                    error=f"Unknown tool: {tool_name}"
                )
            return error_result

        try:
            tool_function = self._tools[tool_name]

            # Add session_id to tool input
            tool_input['session_id'] = session_id

            # Execute tool (handle both sync and async)
            import asyncio
            if asyncio.iscoroutinefunction(tool_function):
                result = await tool_function(**tool_input)
            else:
                result = tool_function(**tool_input)

            logger.info(f"Tool '{tool_name}' executed successfully")
            
            tool_duration_ms = (time.time() - tool_start_time) * 1000
            result_dict = result if isinstance(result, dict) else {"result": result}

            # Log tool execution for observability
            if obs_logger:
                obs_logger.record_tool_execution(
                    tool_name=tool_name,
                    inputs=tool_input,
                    outputs=result_dict,
                    duration_ms=tool_duration_ms,
                    success=True
                )

            # Log tool execution for validation
            if session_id in self._execution_log:
                tool_execution = ToolExecution(
                    tool_name=tool_name,
                    inputs=tool_input,
                    outputs=result_dict,
                    timestamp=datetime.utcnow()
                )
                # APPEND to log (don't replace) - log is passed from orchestrator
                self._execution_log[session_id].tools_used.append(tool_execution)
                logger.debug(f"Appended tool execution to log: {tool_name}")
            else:
                logger.warning(f"⚠️ No execution_log found for session {session_id} - tool {tool_name} not logged!")

            return result_dict

        except Exception as e:
            logger.error(f"Error executing tool '{tool_name}': {e}", exc_info=True)
            error_result = {"error": str(e)}
            tool_duration_ms = (time.time() - tool_start_time) * 1000

            # Log failed tool execution for observability
            if obs_logger:
                obs_logger.record_tool_execution(
                    tool_name=tool_name,
                    inputs=tool_input,
                    outputs=error_result,
                    duration_ms=tool_duration_ms,
                    success=False,
                    error=str(e)
                )

            # Log failed tool execution for validation
            if session_id in self._execution_log:
                tool_execution = ToolExecution(
                    tool_name=tool_name,
                    inputs=tool_input,
                    outputs=error_result,
                    timestamp=datetime.utcnow()
                )
                # APPEND to log (don't replace) - log is passed from orchestrator
                self._execution_log[session_id].tools_used.append(tool_execution)
                logger.debug(f"Appended failed tool execution to log: {tool_name} (error: {str(e)})")
            else:
                logger.warning(f"⚠️ No execution_log found for session {session_id} - failed tool {tool_name} not logged!")

            return error_result

    def _get_error_response(self, error: str) -> str:
        """Generate user-friendly error response."""
        return (
            "I'm sorry, I encountered an error while processing your request. "
            "Please try again or contact support if the issue persists."
        )

    def clear_history(self, session_id: str):
        """Clear conversation history for a session."""
        if session_id in self.conversation_history:
            del self.conversation_history[session_id]
            logger.info(f"Cleared history for {self.agent_name}, session: {session_id}")

    def get_history_length(self, session_id: str) -> int:
        """Get conversation history length."""
        return len(self.conversation_history.get(session_id, []))
