"""Language Selection Agent - Handles initial language preference collection."""

import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class LanguageSelectionAgent:
    """
    Multilingual agent that welcomes patients and collects language preference.

    This agent:
    - Welcomes new patients warmly in both English and Arabic
    - Detects their language preference from their first message
    - Stores the preference in LanguageContext
    - Continues conversation naturally or routes to appropriate agent
    """

    def __init__(self, llm_config_manager):
        self.name = "language_selection"
        self.llm_config_manager = llm_config_manager
        logger.info("[Language Selection Agent] Initialized")

    async def process_message(
        self,
        session_id: str,
        message: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process message and collect language preference.

        Args:
            session_id: Session identifier
            message: User's message (may be in any language)
            context: Full context including conversation history, patient info

        Returns:
            Dict with:
                - detected_language: "en" or "ar"
                - detected_dialect: Optional dialect code
                - response: Welcome message in selected language
                - should_store_preference: bool
                - language_selected: bool
        """
        logger.info(f"[Language Selection] Processing message for session: {session_id}")

        # Detect language from message
        detected_lang, detected_dialect = await self._detect_language_preference(message)

        logger.info(f"[Language Selection] Detected: {detected_lang}-{detected_dialect}")

        # Determine if this is first message or continuation
        recent_turns = context.get("recent_turns", [])
        is_new_session = len(recent_turns) <= 1

        logger.info(f"[Language Selection] Is new session: {is_new_session}, Recent turns: {len(recent_turns)}")

        # Generate response using multilingual LLM
        if is_new_session:
            response = await self._generate_welcome_response(
                message, detected_lang, detected_dialect, context
            )
        else:
            response = await self._generate_continuation_response(
                message, detected_lang, detected_dialect, context
            )

        logger.info(f"[Language Selection] Generated response in {detected_lang}")

        return {
            "detected_language": detected_lang,
            "detected_dialect": detected_dialect,
            "response": response,
            "should_store_preference": True,
            "language_selected": True
        }

    async def _detect_language_preference(self, message: str) -> Tuple[str, Optional[str]]:
        """Detect language preference from user's message."""
        msg_lower = message.strip().lower()

        # Explicit English selection
        if msg_lower in ["1", "english", "en", "one"]:
            logger.info("[Language Selection] Explicit English selection detected")
            return ("en", None)

        # Explicit Arabic selection
        if msg_lower in ["2", "arabic", "ar", "two", "العربية", "عربي"]:
            logger.info("[Language Selection] Explicit Arabic selection detected")
            return ("ar", "ae")

        # Check for Arabic script
        from patient_ai_service.core.script_detector import ScriptDetector
        if ScriptDetector._has_arabic_letters(message):
            logger.info("[Language Selection] Arabic script detected")
            return ("ar", "ae")

        # Default to English
        logger.info("[Language Selection] Defaulting to English")
        return ("en", None)

    async def _generate_welcome_response(
        self,
        message: str,
        language: str,
        dialect: Optional[str],
        context: Dict[str, Any]
    ) -> str:
        """Generate warm welcome message and ask for language preference."""
        logger.info(f"[Language Selection] Generating welcome and language prompt")

        patient_info = context.get("patient_info", {})
        patient_name = patient_info.get("first_name", "")

        # Build system prompt - focus on asking for language preference
        system_prompt = """You are a warm, professional dental clinic receptionist in Dubai, UAE.

YOUR MAIN TASK:
Collect the patient's language preference (English or Arabic) in a natural, friendly way.

INSTRUCTIONS:
1. Welcome them warmly in BOTH English and Arabic
2. Acknowledge their initial message naturally if relevant
3. Clearly ask them to choose their preferred language (English or Arabic)
4. Keep it conversational and friendly

RESPONSE FORMAT:
- Start with bilingual greeting
- Briefly acknowledge their message if it makes sense
- Ask them to select their language preference
- Make it clear and easy to choose

TONE:
- Warm and welcoming
- Professional but friendly
- Natural and conversational
- Make language selection feel effortless

IMPORTANT:
- Always respond in BOTH English and Arabic in this first message
- Make the language choice clear and explicit
- Keep it brief and to the point"""

        # Build user prompt
        if patient_name:
            user_prompt = f"Patient name: {patient_name}\nPatient's initial message: {message}\n\nGenerate your bilingual welcome message asking for language preference:"
        else:
            user_prompt = f"Patient's initial message: {message}\n\nGenerate your bilingual welcome message asking for language preference:"

        # Call LLM
        try:
            llm_config = self.llm_config_manager.get_config("language_selection")

            # Use invoke method with system and user prompts
            response_obj = await llm_config.invoke(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )

            # Extract response text
            if hasattr(response_obj, 'content'):
                response = response_obj.content
            elif isinstance(response_obj, str):
                response = response_obj
            else:
                response = str(response_obj)

            logger.info(f"[Language Selection] LLM response generated successfully")
            return response.strip()

        except Exception as e:
            logger.error(f"[Language Selection] Error generating welcome response: {e}")
            # Fallback bilingual response
            return """Welcome to our dental clinic! 🦷
يا هلا فيك في عيادتنا! 🦷

I'd be happy to help you! To better assist you, please let me know your preferred language:
حياك الله! عشان أقدر أساعدك بشكل أفضل، وضح لي لغتك المفضلة:

• English
• العربية (Arabic)

Which language would you prefer? / أي لغة تفضل؟"""

    async def _generate_continuation_response(
        self,
        message: str,
        language: str,
        dialect: Optional[str],
        context: Dict[str, Any]
    ) -> str:
        """Generate continuation response after language preference set."""
        logger.info(f"[Language Selection] Generating continuation response in {language}")

        # Simple acknowledgment
        if language == "en":
            return "Great! I'll communicate with you in English. How can I help you today?"
        else:
            return "تمام! راح أتواصل معاك بالعربي. كيف أقدر أساعدك اليوم؟"
