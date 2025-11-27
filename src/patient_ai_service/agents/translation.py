"""
Translation Agent.

Handles language detection and translation for multi-language support.
"""

import logging
from typing import Dict, Any, Optional, Tuple

from .base_agent import BaseAgent
from patient_ai_service.models.enums import Language

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
        """
        try:
            prompt = f"""Detect the language of this text and respond with ONLY the ISO 639-1 language code (e.g., 'en', 'ar', 'es', 'fr', 'hi', 'zh', 'pt', 'ru').

Text: "{text}"

Language code:"""

            response = self.llm_client.create_message(
                system=self._get_system_prompt("detection"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
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

    async def translate_to_english(self, text: str, source_lang: str) -> str:
        """
        Translate text from source language to English.

        Args:
            text: Text to translate
            source_lang: Source language code

        Returns:
            Translated English text
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
                temperature=0.3
            )

            translated = response.strip()
            logger.info(f"Translated from {source_lang} to English")
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
                temperature=0.3
            )

            translated = response.strip()
            logger.info(f"Translated from English to {target_lang}")
            return translated

        except Exception as e:
            logger.error(f"Error translating from English: {e}")
            return text  # Return original if translation fails

    async def process_input(self, session_id: str, message: str) -> Tuple[str, str]:
        """
        Process user input: detect language and translate to English.

        Args:
            session_id: Session identifier
            message: User's message

        Returns:
            Tuple of (english_text, detected_language)
        """
        # Get or detect language
        translation_state = self.state_manager.get_translation_state(session_id)

        if translation_state.auto_detect:
            detected_lang = await self.detect_language(message)
            self.state_manager.update_translation_state(
                session_id,
                source_language=detected_lang
            )
        else:
            detected_lang = translation_state.source_language

        # Translate to English if needed
        if detected_lang != "en":
            english_text = await self.translate_to_english(message, detected_lang)
        else:
            english_text = message

        # Update global state
        self.state_manager.update_global_state(
            session_id,
            detected_language=detected_lang
        )

        return english_text, detected_lang

    async def process_output(self, session_id: str, message: str) -> str:
        """
        Process system output: translate from English to user's language.

        Args:
            session_id: Session identifier
            message: English message to translate

        Returns:
            Translated message
        """
        global_state = self.state_manager.get_global_state(session_id)
        target_lang = global_state.detected_language

        if target_lang != "en":
            translated = await self.translate_from_english(message, target_lang)
            return translated
        else:
            return message
