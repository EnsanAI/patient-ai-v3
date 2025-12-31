"""
Translation Agent.

Handles language detection and translation for multi-language support.
"""

import logging
from typing import Dict, Any, Optional, Tuple

from .base_agent import BaseAgent
from patient_ai_service.models.enums import Language
from patient_ai_service.core.config import settings

logger = logging.getLogger(__name__)


class TranslationAgent(BaseAgent):
    """
    Agent for language detection and translation.

    Features:
    - Auto-detect user language
    - Translate input to English for processing
    - Translate output to user's preferred language
    - Cache translations for performance
    """

    SUPPORTED_LANGUAGES = {
        "en": "English",
        "ar": "Arabic",
        "es": "Spanish",
        "fr": "French",
        "hi": "Hindi",
        "zh": "Chinese",
        "pt": "Portuguese",
        "ru": "Russian"
    }

    def __init__(self, **kwargs):
        super().__init__(agent_name="Translation", **kwargs)

    def _register_tools(self):
        """Translation agent uses LLM directly, no additional tools needed."""
        pass

    def _get_system_prompt(self, session_id: str) -> str:
        """Generate translation system prompt."""
        return """You are a professional translation service for a dental clinic.

Your responsibilities:
1. Detect the language of input text
2. Translate text accurately while preserving meaning and tone
3. Maintain medical/dental terminology accuracy
4. Keep translations natural and conversational

Supported languages:
- English (en)
- Arabic (ar)
- Spanish (es)
- French (fr)
- Hindi (hi)
- Chinese (zh)
- Portuguese (pt)
- Russian (ru)

Guidelines:
- Preserve proper nouns and names
- Keep medical terms accurate
- Maintain professional but friendly tone
- Use gender-neutral language when possible
- Preserve formatting and structure"""

    async def detect_language(self, text: str) -> str:
        """
        Detect the language of the input text.

        Args:
            text: Input text to analyze

        Returns:
            ISO 639-1 language code (e.g., 'en', 'ar', 'es')

        Note: This is the legacy method. Use detect_language_and_dialect() for dialect support.
        """
        try:
            prompt = f"""Detect the language of this text and respond with ONLY the ISO 639-1 language code (e.g., 'en', 'ar', 'es', 'fr', 'hi', 'zh', 'pt', 'ru').

Text: "{text}"

Language code:"""

            response = self.llm_client.create_message(
                system=self._get_system_prompt("detection"),
                messages=[{"role": "user", "content": prompt}],
                temperature=settings.translation_temperature
            )

            # Extract language code
            lang_code = response.strip().lower()[:2]

            if lang_code in self.SUPPORTED_LANGUAGES:
                logger.info(f"Detected language: {lang_code}")
                return lang_code
            else:
                logger.warning(f"Unknown language code: {lang_code}, defaulting to 'en'")
                return "en"

        except Exception as e:
            logger.error(f"Error detecting language: {e}")
            return "en"  # Default to English

    async def detect_language_and_dialect(self, text: str) -> Tuple[str, Optional[str]]:
        """
        Detect language AND dialect of input text.

        Args:
            text: Input text to analyze

        Returns:
            Tuple of (language_code, dialect_code)
            Example: ("ar", "EG"), ("en", "US"), ("es", "MX")
        """
        try:
            prompt = f"""Detect the language and regional dialect of this text.

Text: "{text}"

Respond with ONLY a JSON object in this format:
{{
    "language": "ISO 639-1 code (e.g., en, ar, es)",
    "dialect": "Region/dialect code (e.g., US, GB, EG, SA, MX) or null if unknown",
    "confidence": "high/medium/low"
}}

Examples:
- "I want to book an appointment" → {{"language": "en", "dialect": "US", "confidence": "high"}}
- "أريد حجز موعد" (Egyptian accent) → {{"language": "ar", "dialect": "EG", "confidence": "medium"}}
- "Quiero reservar una cita" (Mexican) → {{"language": "es", "dialect": "MX", "confidence": "medium"}}
- "مرحبا، كيف حالك؟" (Gulf accent) → {{"language": "ar", "dialect": "SA", "confidence": "medium"}}

Response:"""

            response = self.llm_client.create_message(
                system=self._get_system_prompt("detection"),
                messages=[{"role": "user", "content": prompt}],
                temperature=settings.translation_temperature
            )

            # Parse JSON response - extract from markdown if needed
            import json
            import re

            # Try to extract JSON from response (handles markdown code blocks)
            # Claude often returns JSON wrapped like: ```json\n{...}\n```
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                logger.error(f"No JSON found in language detection response: {response[:200]}")
                return "en", None

            result = json.loads(json_match.group())

            language = result.get("language", "en")
            dialect = result.get("dialect")

            # Validate language is supported
            if language not in self.SUPPORTED_LANGUAGES:
                logger.warning(f"Unknown language: {language}, defaulting to 'en'")
                language = "en"
                dialect = None

            logger.info(f"Detected: {language}-{dialect if dialect else 'unknown'} (confidence: {result.get('confidence', 'unknown')})")
            return language, dialect

        except Exception as e:
            logger.error(f"Error detecting language/dialect: {e}")
            logger.debug(f"Response was: {response[:200] if 'response' in locals() else 'N/A'}")
            return "en", None

    async def translate_to_english(self, text: str, source_lang: str) -> str:
        """
        Translate text from source language to English.

        Args:
            text: Text to translate
            source_lang: Source language code

        Returns:
            Translated English text

        Note: This is the legacy method. Use translate_to_english_with_dialect() for dialect support.
        """
        if source_lang == "en":
            return text  # Already English

        try:
            prompt = f"""Translate this {self.SUPPORTED_LANGUAGES.get(source_lang, 'text')} to English.
Preserve medical/dental terminology accurately.
Provide ONLY the translation, no explanations.

Text: "{text}"

Translation:"""

            response = self.llm_client.create_message(
                system=self._get_system_prompt("translation"),
                messages=[{"role": "user", "content": prompt}],
                temperature=settings.translation_temperature
            )

            translated = response.strip()
            logger.info(f"Translated from {source_lang} to English")
            return translated

        except Exception as e:
            logger.error(f"Error translating to English: {e}")
            return text  # Return original if translation fails

    async def translate_to_english_with_dialect(
        self,
        text: str,
        source_lang: str,
        source_dialect: Optional[str] = None
    ) -> str:
        """
        Translate text from source language to English with dialect awareness.

        Args:
            text: Text to translate
            source_lang: Source language code
            source_dialect: Optional dialect code (e.g., "EG", "SA", "MX")

        Returns:
            Translated English text
        """
        if source_lang == "en":
            return text  # Already English

        try:
            dialect_note = ""
            if source_dialect:
                dialect_note = f" ({source_dialect} dialect)"

            prompt = f"""Translate this {self.SUPPORTED_LANGUAGES.get(source_lang)}{dialect_note} text to English.

IMPORTANT:
- Preserve medical/dental terminology accurately
- Adapt region-specific terms to US English medical standard
- Maintain the user's intent and tone
- Provide ONLY the translation, no explanations

Text: "{text}"

Translation:"""

            response = self.llm_client.create_message(
                system=self._get_system_prompt("translation"),
                messages=[{"role": "user", "content": prompt}],
                temperature=settings.translation_temperature
            )

            translated = response.strip()
            logger.info(f"Translated from {source_lang}-{source_dialect or 'unknown'} to English")
            return translated

        except Exception as e:
            logger.error(f"Error translating to English: {e}")
            return text  # Return original if translation fails

    async def translate_from_english(self, text: str, target_lang: str) -> str:
        """
        Translate text from English to target language.

        Args:
            text: English text to translate
            target_lang: Target language code

        Returns:
            Translated text

        Note: This is the legacy method. Use translate_from_english_with_dialect() for dialect support.
        """
        if target_lang == "en":
            return text  # Already English

        try:
            prompt = f"""Translate this English text to {self.SUPPORTED_LANGUAGES.get(target_lang, target_lang)}.
Keep it natural and conversational.
Preserve medical/dental terms accurately.
Provide ONLY the translation, no explanations.

Text: "{text}"

Translation:"""

            response = self.llm_client.create_message(
                system=self._get_system_prompt("translation"),
                messages=[{"role": "user", "content": prompt}],
                temperature=settings.translation_temperature
            )

            translated = response.strip()
            logger.info(f"Translated from English to {target_lang}")
            return translated

        except Exception as e:
            logger.error(f"Error translating from English: {e}")
            return text  # Return original if translation fails

    async def translate_from_english_with_dialect(
        self,
        text: str,
        target_lang: str,
        target_dialect: Optional[str] = None
    ) -> str:
        """
        Translate text from English to target language with dialect awareness.

        Args:
            text: English text to translate
            target_lang: Target language code
            target_dialect: Optional dialect code (e.g., "EG", "SA", "MX")

        Returns:
            Translated text
        """
        if target_lang == "en":
            return text  # Already English

        try:
            dialect_note = ""
            if target_dialect:
                dialect_note = f" ({target_dialect} dialect)"

            prompt = f"""Translate this English text to {self.SUPPORTED_LANGUAGES.get(target_lang)}{dialect_note}.

IMPORTANT:
- Use natural, conversational language
- Adapt to regional dialect if specified
- Preserve medical/dental terms accurately
- Be culturally appropriate
- Provide ONLY the translation, no explanations

Text: "{text}"

Translation:"""

            response = self.llm_client.create_message(
                system=self._get_system_prompt("translation"),
                messages=[{"role": "user", "content": prompt}],
                temperature=settings.translation_temperature
            )

            translated = response.strip()
            logger.info(f"Translated from English to {target_lang}-{target_dialect or 'unknown'}")
            return translated

        except Exception as e:
            logger.error(f"Error translating from English: {e}")
            return text  # Return original if translation fails

    async def process_input(self, session_id: str, message: str) -> Tuple[str, str]:
        """
        Process user input: detect language/dialect and translate to English.

        Args:
            session_id: Session identifier
            message: User's message

        Returns:
            Tuple of (english_text, detected_language)

        Note: This method now detects dialect internally but returns only language code
        for backward compatibility. The dialect is stored in GlobalState.language_context.
        """
        # Get or detect language and dialect
        translation_state = self.state_manager.get_translation_state(session_id)

        if translation_state.auto_detect:
            # Use new dialect-aware detection
            detected_lang, detected_dialect = await self.detect_language_and_dialect(message)

            # Update translation state (legacy)
            self.state_manager.update_translation_state(
                session_id,
                source_language=detected_lang
            )
        else:
            # Use configured language (no dialect detection)
            detected_lang = translation_state.source_language
            detected_dialect = None

        # Translate to English if needed (using dialect-aware method)
        if detected_lang != "en":
            english_text = await self.translate_to_english_with_dialect(
                message,
                detected_lang,
                detected_dialect
            )
        else:
            english_text = message

        # NOTE: This update to global state is deprecated.
        # The orchestrator should update language_context directly.
        # Keeping this for backward compatibility.
        self.state_manager.update_global_state(
            session_id,
            detected_language=detected_lang
        )

        # Return language code only for backward compatibility
        # Orchestrator will access dialect from detect_language_and_dialect if needed
        return english_text, detected_lang

    async def process_output(self, session_id: str, message: str) -> str:
        """
        Process system output: translate from English to user's language with dialect.

        Args:
            session_id: Session identifier
            message: English message to translate

        Returns:
            Translated message
        """
        global_state = self.state_manager.get_global_state(session_id)

        # Get language and dialect from language_context
        target_lang = global_state.language_context.current_language
        target_dialect = global_state.language_context.current_dialect

        if target_lang != "en":
            # Use dialect-aware translation
            translated = await self.translate_from_english_with_dialect(
                message,
                target_lang,
                target_dialect
            )
            return translated
        else:
            return message
