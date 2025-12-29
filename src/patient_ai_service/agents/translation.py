"""
Translation Agent.

Handles language detection and translation for multi-language support.
"""

import logging
import re
from typing import Dict, Any, Optional, Tuple

from .base_agent import BaseAgent
from patient_ai_service.models.enums import Language
from patient_ai_service.core.config import settings
from patient_ai_service.core.script_detector import fast_detect_language
from patient_ai_service.core.llm_config import get_llm_config_manager

logger = logging.getLogger(__name__)


class LanguageDetector:
    """
    Enhanced language detector with Franco-Arabic support.

    Uses Unicode ranges for Arabic script detection and keyword matching
    for Franco-Arabic (e.g., "3ayez maw3ed").
    """

    def __init__(self):
        # Franco-Arabic number substitutions
        self.franco_numbers = re.compile(r'[23578]')

        # Common Franco-Arabic words
        self.franco_keywords = {
            'ya', 'ya3ni', 'ana', 'enta', 'enti', 'ehna', 'homma',
            'eh', 'fe', 'mesh', 'msh', 'la2', 'wala', 'bas', 'aw',
            '3ayez', '3ayz', 'awz', 'awza', 'ayez',
            'ro7', 'aro7', 'trou7', 'yro7',
            'geet', 'ga', 'igi', '2olt', 'olt', 'ba2ol',
            '3amel', '3amlt', 'kont', 'kan', 'tamam', 'tmm', 'aked', 'akid', '5alas', 'khalas', 'inshallah',
            'bokra', 'embare7', 'delwa2ty', 'dlw2ty', 'nharda',
            'naharda', 'elyom', 'alyoum', 'wa2t',
            'maw3ed', 'mwa3eed', 'sa3a', 'sa3t',
            'momken', 'momkn', 'law sama7t', 'lw sm7t',
            'min fadlak', 'mn fdlk', 'ynfa3', 'yenfa3',
            'aiwa', 'aywa', 'ah', 'aah', 'tamam', 'tmm',
            'mashi', 'mashy', 'khalas', '5alas',
            'inshallah', 'nshallah', 'wallahi',
            'mafeesh', 'mafesh', 'mafish',
            '3afwan', 'shokran',
            'ezayak', 'ezzayak', 'izzayak', 'zayak',
            '3andak', '3andek', '3ando', 'm3ak', 'ma3ak',
            'ta3ban', '3ayyan', 'waga3', 'wg3',
            '3eyada', 'kashf', 'ta2meen'
        }

        # Common English short words (whitelist)
        self.english_short_words = {
            'yes', 'no', 'ok', 'okay', 'sure', 'hi', 'hello',
            'bye', 'thanks', 'when', 'where', 'why', 'how',
            'what', 'who', 'can', 'will', 'would', 'could',
            'please', 'thank', 'you', 'me', 'my', 'i',
            'exactly', 'right', 'wrong', 'good', 'bad',
            'great', 'fine', 'well', 'very', 'much'
        }

    def is_non_english(self, text: str) -> Tuple[bool, str]:
        """
        Detect if text is non-English.

        Returns:
            Tuple of (is_non_english: bool, detected_language: str)
        """
        if not text or len(text.strip()) < 1:
            return False, 'unknown'

        text_clean = text.strip().lower()

        # 1. Arabic script detection
        if re.search(r'[\u0600-\u06FF]', text):
            return True, 'ar'

        # 2. Franco-Arabic detection
        words = set(re.findall(r'\b\w+\b', text_clean))
        franco_score = 0

        # Check for number substitutions in words
        if self.franco_numbers.search(text_clean):
            franco_score += 2

        # Check for common Franco-Arabic words
        if words & self.franco_keywords:
            franco_score += 3

        if franco_score >= 3:
            return True, 'ar-franco'

        # 3. SHORT TEXT HANDLING - Check whitelist first
        word_count = len(words)
        if word_count <= 2:
            if words.issubset(self.english_short_words):
                return False, 'en'

        # 4. Standard language detection (only for longer text)
        # Use langdetect ONLY to identify non-English, not to determine specific language
        if len(text_clean) >= 10:
            try:
                from langdetect import detect, LangDetectException
                lang = detect(text)
                if lang != 'en':
                    # Non-English detected - return 'other' to trigger LLM-based detection
                    # This ensures we don't try to handle unsupported languages in fast path
                    return True, 'other'
                else:
                    return False, 'en'
            except (ImportError, Exception):
                # langdetect not available or failed - default to English
                return False, 'en'

        # 5. Default to English for very short unrecognized text
        return False, 'en'

    def should_translate(self, text: str) -> bool:
        """Simple boolean check for translation need."""
        is_non_eng, _ = self.is_non_english(text)
        return is_non_eng


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
        self.llm_config_manager = get_llm_config_manager()
        # Initialize enhanced language detector with Franco-Arabic support
        self._language_detector = LanguageDetector()

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

            # Get hierarchical config for detect_language function
            llm_config = self.llm_config_manager.get_config(
                agent_name="translation",
                function_name="detect_language"
            )
            llm_client = self.llm_config_manager.get_client(
                agent_name="translation",
                function_name="detect_language"
            )

            response = llm_client.create_message(
                system=self._get_system_prompt("detection"),
                messages=[{"role": "user", "content": prompt}],
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens
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

    async def detect_and_translate(self, text: str, session_id: Optional[str] = None) -> Tuple[str, str, Optional[str], bool]:
        """
        OPTIMIZED: Single LLM call for language detection AND translation.

        This is the preferred method for the translation layer - combines detection
        and translation into one fast call to reduce latency.

        Args:
            text: Input text to analyze and translate
            session_id: Optional session ID for observability tracking

        Returns:
            Tuple of (english_text, language_code, dialect_code, translation_succeeded)
            - english_text: Translated text (or original if already English)
            - language_code: ISO 639-1 code (e.g., "ar", "en", "es")
            - dialect_code: Region code (e.g., "EG", "SA", "MX") or None
            - translation_succeeded: True if translation worked, False if fallback used
        """
        import json
        import re
        import time
        from patient_ai_service.core.observability import get_observability_logger
        from patient_ai_service.models.observability import TokenUsage

        # ═══════════════════════════════════════════════════════════════════
        # FAST PATH: Enhanced language detection with Franco-Arabic support
        # ═══════════════════════════════════════════════════════════════════
        is_non_english, detected_lang = self._language_detector.is_non_english(text)

        if not is_non_english:
            # English - no translation needed
            logger.info(f"[FAST PATH] LanguageDetector: English, skipping LLM")
            return text, "en", None, True

        if detected_lang == 'ar' or detected_lang == 'ar-franco':
            # Arabic (script or Franco) - translate with Emirati dialect
            logger.info(f"[FAST PATH] LanguageDetector: {detected_lang}, translating to English")
            english_text = await self._translate_arabic_to_english(text, session_id)
            return english_text, "ar", "ae", True
        
        # ═══════════════════════════════════════════════════════════════════
        # NORMAL PATH: LLM-based detection and translation
        # ═══════════════════════════════════════════════════════════════════
        try:
            prompt = f"""Analyze this text and perform detection + translation in ONE response.

Text: "{text}"

Tasks:
1. Detect the language (ISO 639-1 code: en, ar, es, fr, hi, zh, pt, ru)
2. Detect the regional dialect ONLY for non-English languages (e.g., EG for Egyptian Arabic, SA for Gulf Arabic, MX for Mexican Spanish)
3. If NOT English, translate to English while preserving medical/dental terminology

IMPORTANT: For English text, always set dialect to null.

Respond with ONLY a JSON object:
{{
    "language": "ISO 639-1 code",
    "dialect": "region code or null (ALWAYS null for English)",
    "is_english": true/false,
    "english_text": "the English translation OR original text if already English",
    "confidence": "high/medium/low"
}}

Examples:
- "I want to book an appointment" → {{"language": "en", "dialect": null, "is_english": true, "english_text": "I want to book an appointment", "confidence": "high"}}
- "عايز احجز ميعاد" → {{"language": "ar", "dialect": "EG", "is_english": false, "english_text": "I want to book an appointment", "confidence": "high"}}
- "Quiero reservar una cita" → {{"language": "es", "dialect": "MX", "is_english": false, "english_text": "I want to book an appointment", "confidence": "high"}}

Response:"""

            # Get hierarchical config for detect_and_translate function
            llm_config = self.llm_config_manager.get_config(
                agent_name="translation",
                function_name="detect_and_translate"
            )
            llm_client = self.llm_config_manager.get_client(
                agent_name="translation",
                function_name="detect_and_translate"
            )
            
            # Track LLM call with token usage
            llm_start_time = time.time()
            if hasattr(llm_client, 'create_message_with_usage'):
                response, tokens = llm_client.create_message_with_usage(
                    system=self._get_system_prompt("detection_and_translation"),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=llm_config.temperature,
                    max_tokens=llm_config.max_tokens
                )
            else:
                response = llm_client.create_message(
                    system=self._get_system_prompt("detection_and_translation"),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=llm_config.temperature,
                    max_tokens=llm_config.max_tokens
                )
                tokens = TokenUsage()

            llm_duration_seconds = time.time() - llm_start_time

            # Record LLM call for observability
            if session_id and settings.enable_observability:
                obs_logger = get_observability_logger(session_id)
                if obs_logger:
                    obs_logger.record_llm_call(
                        component="translation.detect_and_translate",
                        provider=llm_config.provider,
                        model=llm_config.model,
                        tokens=tokens,
                        duration_seconds=llm_duration_seconds,
                        system_prompt_length=len(self._get_system_prompt("detection_and_translation")),
                        messages_count=1,
                        temperature=llm_config.temperature,
                        max_tokens=llm_config.max_tokens,
                        function_name="detect_and_translate"
                    )

            # Parse JSON response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                logger.error(f"No JSON found in detect_and_translate response: {response[:200]}")
                return text, "en", None, False  # Fallback: assume English, mark as failed

            result = json.loads(json_match.group())

            language = result.get("language", "en")
            dialect = result.get("dialect")
            english_text = result.get("english_text", text)
            is_english = result.get("is_english", True)

            # Validate language is supported
            if language not in self.SUPPORTED_LANGUAGES:
                logger.warning(f"Unsupported language '{language}', defaulting to 'en'")
                language = "en"
                dialect = None
                english_text = text

            # IMPORTANT: Ensure dialect is None for English
            if language == "en":
                dialect = None

            # Validate we got a translation if needed
            if not is_english and (not english_text or english_text.strip() == ""):
                logger.error(f"Translation returned empty for non-English text")
                return text, language, dialect, False  # Return original, mark as failed

            logger.info(
                f"[FAST] Detected: {language}-{dialect or 'unknown'}, "
                f"translated: {not is_english}, "
                f"confidence: {result.get('confidence', 'unknown')}"
            )

            return english_text, language, dialect, True

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error in detect_and_translate: {e}")
            return text, "en", None, False
        except Exception as e:
            logger.error(f"Error in detect_and_translate: {e}")
            return text, "en", None, False  # Fallback: return original, mark as failed

    async def _translate_arabic_to_english(self, text: str, session_id: Optional[str] = None) -> str:
        """
        Translate Arabic text to English without dialect detection.
        Used when script detection already identified Arabic.
        
        Args:
            text: Arabic text to translate
            session_id: Optional session ID for observability
            
        Returns:
            English translation
        """
        import time
        from patient_ai_service.core.observability import get_observability_logger
        from patient_ai_service.models.observability import TokenUsage
        
        prompt = f"""Arabic Text: "{text}"

English translation:"""

        try:
            # Get hierarchical config for translate_text function (closest match)
            llm_config = self.llm_config_manager.get_config(
                agent_name="translation",
                function_name="translate_text"
            )
            llm_client = self.llm_config_manager.get_client(
                agent_name="translation",
                function_name="translate_text"
            )
            
            llm_start_time = time.time()
            
            system_prompt = "You are an Arabic to English translator for a clinic. Translate naturally. Handle slang/colloquialisms with English equivalents (NEVER translate them literally). Preserve medical terms. Output: 'English translation: [text]' only."
            
            if hasattr(llm_client, 'create_message_with_usage'):
                response, tokens = llm_client.create_message_with_usage(
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=llm_config.temperature,
                    max_tokens=llm_config.max_tokens
                )
            else:
                response = llm_client.create_message(
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=llm_config.temperature,
                    max_tokens=llm_config.max_tokens
                )
                tokens = TokenUsage()
            
            llm_duration_seconds = time.time() - llm_start_time
            
            # Record LLM call for observability
            if session_id and settings.enable_observability:
                obs_logger = get_observability_logger(session_id)
                if obs_logger:
                    obs_logger.record_llm_call(
                        component="translation.arabic_to_english_fast",
                        provider=llm_config.provider,
                        model=llm_config.model,
                        tokens=tokens,
                        duration_seconds=llm_duration_seconds,
                        system_prompt_length=len(system_prompt),
                        messages_count=1,
                        temperature=llm_config.temperature,
                        max_tokens=llm_config.max_tokens,
                        function_name="arabic_to_english_fast"
                    )
            
            # Clean up response - extract only the translation text
            translation = response.strip()
            
            # Extract text after "English translation:" marker
            if "English translation:" in translation:
                parts = translation.split("English translation:", 1)
                if len(parts) > 1:
                    translation = parts[1].strip()
            
            # Remove markdown code blocks
            if translation.startswith("```"):
                lines = translation.split("\n")
                in_code = False
                cleaned = []
                for line in lines:
                    if line.strip().startswith("```"):
                        in_code = not in_code
                        continue
                    if not in_code:
                        cleaned.append(line)
                translation = "\n".join(cleaned).strip()
            
            # Extract quoted text if present (common LLM format: "translation text")
            import re
            quoted_match = re.search(r'"([^"]+)"', translation)
            if quoted_match:
                translation = quoted_match.group(1)
            else:
                # Remove headers, notes, horizontal rules, and stop at notes
                lines = translation.split("\n")
                cleaned_lines = []
                for line in lines:
                    line_stripped = line.strip()
                    # Stop at notes
                    if line_stripped.startswith("**Note:**") or line_stripped.startswith("Note:") or line_stripped.startswith("*Note:"):
                        break
                    # Skip headers and separators
                    if line_stripped.startswith("#") or line_stripped == "---" or line_stripped == "":
                        continue
                    # Skip lines that are just formatting
                    if line_stripped.lower().startswith("english translation"):
                        continue
                    cleaned_lines.append(line)
                translation = "\n".join(cleaned_lines).strip()
            
            # Final cleanup: remove any remaining quotes or extra whitespace
            translation = translation.strip('"').strip("'").strip()
            
            return translation
            
        except Exception as e:
            logger.error(f"Error in Arabic translation: {e}")
            return text  # Return original if translation fails

    async def detect_language_and_dialect(self, text: str) -> Tuple[str, Optional[str]]:
        """
        Detect language AND dialect of input text.

        NOTE: For better performance, consider using detect_and_translate() which
        combines detection and translation into a single LLM call.

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
- "عايز احجز ميعاد" (Egyptian accent) → {{"language": "ar", "dialect": "EG", "confidence": "medium"}}
- "Quiero reservar una cita" (Mexican) → {{"language": "es", "dialect": "MX", "confidence": "medium"}}
- "مرحبا، كيف حالك؟" (Gulf accent) → {{"language": "ar", "dialect": "SA", "confidence": "medium"}}

Response:"""

            # Get hierarchical config for detect_language_and_dialect function
            llm_config = self.llm_config_manager.get_config(
                agent_name="translation",
                function_name="detect_language_and_dialect"
            )
            llm_client = self.llm_config_manager.get_client(
                agent_name="translation",
                function_name="detect_language_and_dialect"
            )

            response = llm_client.create_message(
                system=self._get_system_prompt("detection"),
                messages=[{"role": "user", "content": prompt}],
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens
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

            # Get hierarchical config for translate_to_english function
            llm_config = self.llm_config_manager.get_config(
                agent_name="translation",
                function_name="translate_to_english"
            )
            llm_client = self.llm_config_manager.get_client(
                agent_name="translation",
                function_name="translate_to_english"
            )

            response = llm_client.create_message(
                system=self._get_system_prompt("translation"),
                messages=[{"role": "user", "content": prompt}],
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens
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

            # Get hierarchical config for translate_to_english_with_dialect function
            llm_config = self.llm_config_manager.get_config(
                agent_name="translation",
                function_name="translate_to_english_with_dialect"
            )
            llm_client = self.llm_config_manager.get_client(
                agent_name="translation",
                function_name="translate_to_english_with_dialect"
            )

            response = llm_client.create_message(
                system=self._get_system_prompt("translation"),
                messages=[{"role": "user", "content": prompt}],
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens
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

            # Get hierarchical config for translate_from_english function
            llm_config = self.llm_config_manager.get_config(
                agent_name="translation",
                function_name="translate_from_english"
            )
            llm_client = self.llm_config_manager.get_client(
                agent_name="translation",
                function_name="translate_from_english"
            )

            response = llm_client.create_message(
                system=self._get_system_prompt("translation"),
                messages=[{"role": "user", "content": prompt}],
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens
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
        target_dialect: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> str:
        """
        Translate text from English to target language with dialect awareness.

        Args:
            text: English text to translate
            target_lang: Target language code
            target_dialect: Optional dialect code (e.g., "EG", "SA", "MX")
            session_id: Optional session ID for observability tracking

        Returns:
            Translated text
        """
        if target_lang == "en":
            return text  # Already English

        try:
            import time
            from patient_ai_service.core.observability import get_observability_logger
            from patient_ai_service.models.observability import TokenUsage

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

            # Get hierarchical config for translate_text function
            llm_config = self.llm_config_manager.get_config(
                agent_name="translation",
                function_name="translate_text"
            )
            llm_client = self.llm_config_manager.get_client(
                agent_name="translation",
                function_name="translate_text"
            )
            
            # Track LLM call with token usage
            llm_start_time = time.time()
            if hasattr(llm_client, 'create_message_with_usage'):
                response, tokens = llm_client.create_message_with_usage(
                    system=self._get_system_prompt("translation"),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=llm_config.temperature,
                    max_tokens=llm_config.max_tokens
                )
            else:
                response = llm_client.create_message(
                    system=self._get_system_prompt("translation"),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=llm_config.temperature,
                    max_tokens=llm_config.max_tokens
                )
                tokens = TokenUsage()

            llm_duration_seconds = time.time() - llm_start_time

            # Record LLM call for observability
            if session_id and settings.enable_observability:
                obs_logger = get_observability_logger(session_id)
                if obs_logger:
                    obs_logger.record_llm_call(
                        component="translation.translate_from_english",
                        provider=llm_config.provider,
                        model=llm_config.model,
                        tokens=tokens,
                        duration_seconds=llm_duration_seconds,
                        system_prompt_length=len(self._get_system_prompt("translation")),
                        messages_count=1,
                        temperature=llm_config.temperature,
                        max_tokens=llm_config.max_tokens,
                        function_name="translate_from_english_with_dialect"
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
            # IMPORTANT: Force Emirati Arabic (ae) dialect for all Arabic responses
            if target_lang == "ar":
                target_dialect = "ae"
                logger.info(f"[Translation] Forcing Emirati Arabic dialect (ae) for Arabic response")

            # Use dialect-aware translation
            translated = await self.translate_from_english_with_dialect(
                message,
                target_lang,
                target_dialect,
                session_id
            )
            return translated
        else:
            return message
