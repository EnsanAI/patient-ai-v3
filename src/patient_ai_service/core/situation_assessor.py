"""
Situation Assessor - Lightweight LLM-Powered Message Classification

This component runs on EVERY message to determine:
1. What type of situation this is
2. How much reasoning is needed
3. What entities were mentioned/changed

Design principles:
- Always LLM-powered (no keyword shortcuts)
- Minimal context for speed (~300 tokens)
- Low confidence triggers comprehensive reasoning
"""

import json
import logging
import time
from typing import Optional, Dict, Any, Tuple, List

from patient_ai_service.core.llm import LLMClient, get_llm_client
from patient_ai_service.core.config import settings
from patient_ai_service.core.state_manager import StateManager, get_state_manager
from patient_ai_service.core.observability import get_observability_logger
from patient_ai_service.models.situation_assessment import (
    SituationAssessment,
    SituationType,
    ReasoningNeeds,
    ExtractedEntities,
    EntityChanges
)
from patient_ai_service.models.reasoning_state import ReasoningState
from patient_ai_service.models.observability import TokenUsage

logger = logging.getLogger(__name__)


# Confidence threshold - below this triggers comprehensive reasoning
CONFIDENCE_THRESHOLD = 0.7


class SituationAssessor:
    """
    Lightweight LLM-based situation assessor.
    
    Runs on every message to determine how to handle it,
    without the overhead of full reasoning.
    """
    
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        state_manager: Optional[StateManager] = None
    ):
        self.llm_client = llm_client or get_llm_client()
        self.state_manager = state_manager or get_state_manager()
        
        logger.info("SituationAssessor initialized")
    
    async def assess(
        self,
        session_id: str,
        message: str,
        patient_info: Optional[Dict[str, Any]] = None,
        continuation_context: Optional[Dict[str, Any]] = None,  # NEW
        active_agent: Optional[str] = None  # NEW
    ) -> SituationAssessment:
        """
        Assess the situation with awareness of what we're awaiting.
        
        Args:
            session_id: Session identifier
            message: User's message (in English)
            patient_info: Patient profile info
            continuation_context: What we're awaiting (if any)
            active_agent: Currently active agent (if any)
        """
        start_time = time.time()
        
        logger.info("🦾 [SituationAssessor] ════════════════════════════════════════════════════════")
        logger.info(f"🦾 [SituationAssessor] Assessing message for session: {session_id}")
        logger.info(f"🦾 [SituationAssessor] Message: '{message[:100]}{'...' if len(message) > 100 else ''}'")
        
        # Extract awaiting context
        awaiting = None
        awaiting_options = []
        resolved_entities = {}
        original_request = None
        
        if continuation_context:
            awaiting = continuation_context.get('awaiting')
            awaiting_options = continuation_context.get('presented_options', [])
            resolved_entities = continuation_context.get('resolved_entities', {})
            original_request = continuation_context.get('original_request')
            logger.info(f"🦾 [SituationAssessor] ⏳ Awaiting: {awaiting}")
            logger.info(f"🦾 [SituationAssessor]   Active agent: {active_agent or 'None'}")
        
        # Get reasoning state for context
        reasoning_state = self._get_reasoning_state(session_id)
        logger.info(f"🎯 [ReasoningState] Active agent: {reasoning_state.active_agent or 'None'}")
        logger.info(f"🎯 [ReasoningState] Awaiting: {reasoning_state.awaiting or 'None'}")
        logger.info(f"🎯 [ReasoningState] Established intent: {reasoning_state.established_intent or 'None'}")
        
        # Get entity state for current preferences
        entity_state = self.state_manager.get_entity_state(session_id)
        prefs = entity_state.patient.to_context_dict()
        has_prefs = any(v for v in prefs.values() if v)
        logger.info(f"📦 [EntityState] Has preferences: {has_prefs}")
        if has_prefs:
            logger.info(f"📦 [EntityState] Preferences: {[k for k, v in prefs.items() if v]}")
        
        # Build the assessment prompt with awaiting context
        prompt = self._build_assessment_prompt(
            message=message,
            reasoning_state=reasoning_state,
            patient_info=patient_info,
            current_preferences=prefs,
            awaiting=awaiting,
            awaiting_options=awaiting_options,
            resolved_entities=resolved_entities,
            original_request=original_request,
            active_agent=active_agent
        )
        
        logger.info(f"🦾 [SituationAssessor] Prompt length: {len(prompt)} chars")
        
        # Get observability logger
        obs_logger = get_observability_logger(session_id) if settings.enable_observability else None
        
        try:
            # Make LLM call with low temperature for consistency
            llm_start = time.time()
            
            if hasattr(self.llm_client, 'create_message_with_usage'):
                response, tokens = self.llm_client.create_message_with_usage(
                    system=self._get_system_prompt(),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,  # Very low for consistent classification
                    max_tokens=500    # Keep response small
                )
            else:
                response = self.llm_client.create_message(
                    system=self._get_system_prompt(),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=500
                )
                tokens = TokenUsage()
            
            llm_duration_ms = (time.time() - llm_start) * 1000
            llm_duration_seconds = llm_duration_ms / 1000
            
            # Record LLM call
            if obs_logger:
                llm_call = obs_logger.record_llm_call(
                    component="situation_assessor",
                    provider=settings.llm_provider.value,
                    model=settings.get_llm_model(),
                    tokens=tokens,
                    duration_seconds=llm_duration_seconds,
                    system_prompt_length=len(self._get_system_prompt()),
                    messages_count=1,
                    temperature=0.1,
                    max_tokens=500
                )
                
                # Full observability logging for token calculation
                logger.info("🦾 [SituationAssessor] 📊 Token Usage Details:")
                logger.info(f"🦾 [SituationAssessor]   Input tokens: {tokens.input_tokens}")
                logger.info(f"🦾 [SituationAssessor]   Output tokens: {tokens.output_tokens}")
                logger.info(f"🦾 [SituationAssessor]   Total tokens: {tokens.total_tokens}")
                logger.info(f"🦾 [SituationAssessor]   LLM Duration: {llm_duration_ms:.2f}ms ({llm_duration_seconds:.3f}s)")
                logger.info(f"🦾 [SituationAssessor]   Model: {settings.get_llm_model()}")
                logger.info(f"🦾 [SituationAssessor]   Provider: {settings.llm_provider.value}")
                logger.info(f"🦾 [SituationAssessor]   Temperature: 0.1")
                logger.info(f"🦾 [SituationAssessor]   Max tokens: 500")
                logger.info(f"🦾 [SituationAssessor]   System prompt length: {len(self._get_system_prompt())} chars")
                logger.info(f"🦾 [SituationAssessor]   User prompt length: {len(prompt)} chars")
                
                # Log cost if available
                if llm_call and llm_call.cost and settings.cost_tracking_enabled:
                    logger.info(f"🦾 [SituationAssessor]   Cost: ${llm_call.cost.total_cost_usd:.6f}")
                    logger.info(f"🦾 [SituationAssessor]     - Input cost: ${llm_call.cost.input_cost_usd:.6f}")
                    logger.info(f"🦾 [SituationAssessor]     - Output cost: ${llm_call.cost.output_cost_usd:.6f}")
                
                # Record tokens in token tracker for component-level tracking
                obs_logger.token_tracker.record_tokens(
                    component="situation_assessor",
                    input_tokens=tokens.input_tokens,
                    output_tokens=tokens.output_tokens
                )
            
            # Parse response
            assessment = self._parse_assessment_response(response, message)
            
            # Add timing
            assessment.assessment_duration_ms = (time.time() - start_time) * 1000
            
            # Update reasoning state based on assessment
            self._update_reasoning_state(session_id, reasoning_state, assessment)
            
            # Enhanced logging
            logger.info("🦾 [SituationAssessor] ════════════════════════════════════════════════════════")
            logger.info(f"🦾 [SituationAssessor] ✅ Assessment complete in {assessment.assessment_duration_ms:.0f}ms")
            logger.info(f"🦾 [SituationAssessor]   Type: {assessment.situation_type.value}")
            logger.info(f"🦾 [SituationAssessor]   Confidence: {assessment.confidence:.2f}")
            logger.info(f"🦾 [SituationAssessor]   Reasoning needs: {assessment.reasoning_needs.value}")
            logger.info(f"🦾 [SituationAssessor]   Key understanding: {assessment.key_understanding}")
            
            # Log extracted entities
            ext = assessment.extracted_entities
            if ext.doctor_name or ext.date or ext.time or ext.procedure:
                logger.info("🦾 [SituationAssessor]   📦 Extracted entities:")
                if ext.doctor_name:
                    logger.info(f"🦾 [SituationAssessor]      • Doctor: {ext.doctor_name}")
                if ext.date:
                    logger.info(f"🦾 [SituationAssessor]      • Date: {ext.date}")
                if ext.time:
                    logger.info(f"🦾 [SituationAssessor]      • Time: {ext.time}")
                if ext.procedure:
                    logger.info(f"🦾 [SituationAssessor]      • Procedure: {ext.procedure}")
            
            # Log routing decision
            needs_comprehensive = assessment.needs_comprehensive_reasoning()
            routing_path = "COMPREHENSIVE" if needs_comprehensive else "FOCUSED"
            logger.info(f"🦾 [SituationAssessor]   ⚡ Routing: {routing_path}")
            if assessment.continue_with_active_agent:
                logger.info(f"🦾 [SituationAssessor]   🔄 Continue with active agent: {reasoning_state.active_agent}")
            
            return assessment
            
        except Exception as e:
            logger.error(f"🦾 [SituationAssessor] ❌ Error in situation assessment: {e}", exc_info=True)
            # Return safe fallback - comprehensive reasoning
            fallback = self._fallback_assessment(message)
            logger.warning(f"🦾 [SituationAssessor] ⚠️  Using fallback assessment (triggers comprehensive)")
            return fallback
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for assessor - kept minimal but precise."""
        return """You are a fast situation classifier for a dental clinic AI.

Your job: Quickly classify the user's message and extract key info.

Output ONLY valid JSON with this structure:
{
    "situation_type": "greeting|farewell|thanks|pleasantry|direct_continuation|selection|confirmation|rejection|modification|new_intent|topic_shift|ambiguous|emergency",
    "confidence": 0.0-1.0,
    "key_understanding": "What user means in 1 sentence",
    "reasoning_needs": "none|minimal|focused|comprehensive",
    "extracted": {
        "doctor": "name if mentioned",
        "date": "if mentioned",
        "time": "if mentioned",
        "procedure": "if mentioned",
        "selected_option": "if selecting from options"
    },
    "changes": ["list of changed preferences if any"],
    "continue_with_active_agent": true/false,
    "is_response_to_options": true/false
}

═══════════════════════════════════════════════════════════════════════════
SITUATION TYPE DEFINITIONS (READ CAREFULLY)
═══════════════════════════════════════════════════════════════════════════

"new_intent": User wants INFORMATION or ACTION
  ✓ EXAMPLES:
    - "what are your hours?" → Wants clinic hours
    - "where are you located?" → Wants address
    - "how much is X?" → Wants pricing
    - "I want to book an appointment" → Wants action
    - "do you accept insurance?" → Wants information
    - "can I see Dr. Smith?" → Wants action
    - "what services do you offer?" → Wants information
  → reasoning_needs: "comprehensive"

"emergency": Medical/dental emergency
  ✓ EXAMPLES: "severe pain", "bleeding won't stop", "tooth knocked out"
  → reasoning_needs: "comprehensive"

"pleasantry": Pure social exchange, NO information request
  ✓ EXAMPLES:
    - "how are you?" → Social greeting, no info needed
    - "I'm fine" → Social response, no info needed
    - "كيف حالك؟" → Social greeting in Arabic
    - "hope you're well" → Social pleasantry
  → reasoning_needs: "none"
  ✗ NOT: "how are your hours?" (that's new_intent - wants info)

═══════════════════════════════════════════════════════════════════════════
REASONING NEEDS RULES
═══════════════════════════════════════════════════════════════════════════

- "none": greeting, farewell, thanks, pleasantry
- "minimal": clear selections, simple confirmations ("yes", "the 3pm one")
- "focused": modifications with clear new values
- "comprehensive": new_intent, emergency, ambiguous, complex requests

confidence < 0.7 triggers full reasoning as safety fallback.

═══════════════════════════════════════════════════════════════════════════
CONTINUATION TYPES (when we're awaiting user input)
═══════════════════════════════════════════════════════════════════════════

"direct_answer": User directly answers what we're awaiting
  ✓ EXAMPLES (when awaiting="gender"):
    - "male" → Direct answer
    - "female" → Direct answer
    - "I'm a man" → Direct answer to gender question
  ✓ EXAMPLES (when awaiting="visit_reason"):
    - "cleaning" → Direct answer
    - "checkup" → Direct answer
    - "I have a toothache" → Direct answer
  → reasoning_needs: "minimal"
  → is_response_to_awaiting: true
  → continue_with_active_agent: true

"selection": User picks from options we presented
  ✓ EXAMPLES (when awaiting="time_selection", presented_options=[...]):
    - "the 3pm one" → Selection
    - "first option" → Selection
    - "option 2" → Selection
  → reasoning_needs: "minimal"
  → is_response_to_awaiting: true
  → is_response_to_options: true

"confirmation": User confirms a proposed action
  ✓ EXAMPLES (when awaiting="confirmation"):
    - "yes" → Confirmation
    - "go ahead" → Confirmation
    - "confirm" → Confirmation
    - "sure" → Confirmation
  → reasoning_needs: "minimal"
  → is_response_to_awaiting: true

"rejection": User rejects/declines
  ✓ EXAMPLES (when awaiting="confirmation"):
    - "no" → Rejection
    - "cancel" → Rejection
    - "never mind" → Rejection
  → reasoning_needs: "minimal"
  → is_response_to_awaiting: true

"direct_continuation": User continues the flow with related information
  ✓ EXAMPLES (when active_agent="registration", no specific awaiting):
    - User volunteers additional info related to registration
  → reasoning_needs: "focused"
  → continue_with_active_agent: true

WHEN IN DOUBT: If you're unsure whether something is "pleasantry" or "new_intent",
choose "new_intent" - it's safer to use the agent than to give a wrong fast response.

Be fast and accurate. Output ONLY JSON."""
    
    def _build_assessment_prompt(
        self,
        message: str,
        reasoning_state: ReasoningState,
        patient_info: Optional[Dict[str, Any]],
        current_preferences: Dict[str, Any],
        awaiting: Optional[str] = None,
        awaiting_options: Optional[List[Any]] = None,
        resolved_entities: Optional[Dict[str, Any]] = None,
        original_request: Optional[str] = None,
        active_agent: Optional[str] = None
    ) -> str:
        """Build the prompt for situation assessment."""
        
        # Base context
        prompt = f"""Assess this user message in a dental clinic context.

USER MESSAGE: "{message}"

PATIENT INFO:
- Patient ID: {patient_info.get('patient_id', 'Not registered') if patient_info else 'Not registered'}
- Name: {patient_info.get('first_name', 'Unknown') if patient_info else 'Unknown'} {patient_info.get('last_name', '') if patient_info else ''}
"""

        # Add awaiting context if present
        if awaiting:
            prompt += f"""
═══════════════════════════════════════════════════════════════
⏳ AWAITING CONTEXT - We asked the user something!
═══════════════════════════════════════════════════════════════
We are currently WAITING for: {awaiting}
Active agent: {active_agent or 'None'}
Original request: {original_request or 'Unknown'}

Already resolved:
{json.dumps(resolved_entities, indent=2) if resolved_entities else '  (nothing yet)'}
"""
            if awaiting_options:
                prompt += f"""
Options we presented:
{json.dumps(awaiting_options[:5], indent=2)}{'... and more' if len(awaiting_options) > 5 else ''}
"""
            
            prompt += """
CRITICAL: Determine if the user's message is:
1. direct_answer - Directly answers what we're awaiting (e.g., we asked for gender, they said "male")
2. selection - Picks from the options we presented (e.g., "the 3pm one", "option 2")
3. confirmation - Confirms a pending action (e.g., "yes", "go ahead", "confirm")
4. rejection - Rejects/declines (e.g., "no", "cancel", "never mind")
5. modification - Wants to change something already resolved (e.g., "actually different doctor")
6. direct_continuation - Related info without specific awaiting (e.g., volunteering extra details)
7. new_intent - Completely different request requiring different agent (e.g., "what are your hours?")

If we're awaiting something, lean toward interpreting the message as a response unless it's clearly unrelated.
"""
        else:
            prompt += """
No pending awaiting context - this is a fresh message or continuation of general conversation.
"""

        prompt += """
═══════════════════════════════════════════════════════════════
RESPOND WITH JSON:
═══════════════════════════════════════════════════════════════
{
    "situation_type": "GREETING | FAREWELL | CONFIRMATION | REJECTION | SELECTION | 
                       DIRECT_ANSWER | MODIFICATION | CLARIFICATION_RESPONSE | 
                       NEW_INTENT | EMERGENCY | PLEASANTRY | PIVOT_SAME_FLOW",
    "confidence": 0.0-1.0,
    "key_understanding": "What the user means/wants",
    "reasoning_needs": "NONE | MINIMAL | FULL",
    "continue_with_active_agent": true/false,
    "is_response_to_awaiting": true/false,
    "user_sentiment": "positive | negative | neutral",
        "extracted_entities": {
        "doctor_name": null or "Dr. Smith",
        "date": null or "2025-12-15",
        "time": null or "14:00",
        "procedure": null or "cleaning",
        "selected_option": null or "the selected option value"
        },
        "suggested_agent": null or "appointment_manager | registration | general_assistant | emergency_response",
        "suggested_tone": "a short natural-language tone for how the assistant should respond (for example: warm and professional, concise and neutral, friendly and reassuring, urgent and direct). Choose whatever tone best fits the user and situation."
        }
"""
        return prompt
    
    def _parse_assessment_response(
        self,
        response: str,
        original_message: str
    ) -> SituationAssessment:
        """Parse LLM response into SituationAssessment."""
        try:
            # Extract JSON from response
            json_str = self._extract_json(response)
            if not json_str:
                logger.warning("No JSON found in assessor response")
                return self._fallback_assessment(original_message)
            
            data = json.loads(json_str)
            
            # Parse situation type
            situation_type_str = data.get("situation_type", "ambiguous")
            try:
                situation_type = SituationType(situation_type_str)
            except ValueError:
                situation_type = SituationType.AMBIGUOUS
            
            # Parse reasoning needs
            reasoning_needs_str = data.get("reasoning_needs", "comprehensive")
            try:
                reasoning_needs = ReasoningNeeds(reasoning_needs_str)
            except ValueError:
                reasoning_needs = ReasoningNeeds.COMPREHENSIVE
            
            # Parse extracted entities (support both old "extracted" and new "extracted_entities" format)
            extracted_data = data.get("extracted_entities") or data.get("extracted", {})
            extracted_entities = ExtractedEntities(
                doctor_name=extracted_data.get("doctor_name") or extracted_data.get("doctor"),
                date=extracted_data.get("date"),
                time=extracted_data.get("time"),
                procedure=extracted_data.get("procedure"),
                selected_option=extracted_data.get("selected_option")
            )
            
            # Parse entity changes
            changes_list = data.get("changes", [])
            entity_changes = EntityChanges(
                changed_fields=changes_list
            )
            
            # Build assessment
            assessment = SituationAssessment(
                situation_type=situation_type,
                confidence=float(data.get("confidence", 0.5)),
                key_understanding=data.get("key_understanding", original_message),
                reasoning_needs=reasoning_needs,
                extracted_entities=extracted_entities,
                entity_changes=entity_changes,
                continue_with_active_agent=data.get("continue_with_active_agent", False),
                is_response_to_options=data.get("situation_type") == 'SELECTION',
                is_response_to_awaiting=data.get("is_response_to_awaiting", False),  # NEW
                user_sentiment=data.get("user_sentiment", "neutral"),
                suggested_agent=data.get("suggested_agent"),
                suggested_tone=data.get("suggested_tone")
            )
            
            return assessment
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error in assessor: {e}")
            return self._fallback_assessment(original_message)
        except Exception as e:
            logger.error(f"Error parsing assessment: {e}")
            return self._fallback_assessment(original_message)
    
    def _extract_json(self, response: str) -> Optional[str]:
        """Extract JSON from response."""
        response = response.strip()
        
        # Try direct parse
        if response.startswith("{"):
            return response
        
        # Look for JSON in response
        start = response.find("{")
        end = response.rfind("}") + 1
        
        if start != -1 and end > start:
            return response[start:end]
        
        return None
    
    def _fallback_assessment(self, message: str) -> SituationAssessment:
        """Return safe fallback assessment that triggers comprehensive reasoning."""
        return SituationAssessment(
            situation_type=SituationType.AMBIGUOUS,
            confidence=0.3,  # Low confidence triggers comprehensive
            key_understanding=message,
            reasoning_needs=ReasoningNeeds.COMPREHENSIVE,
            reasoning_reason="Fallback due to assessment error"
        )
    
    def _get_reasoning_state(self, session_id: str) -> ReasoningState:
        """Get or create reasoning state for session."""
        key = f"session:{session_id}:reasoning_state"
        data = self.state_manager.backend.get(key)
        state: ReasoningState
        
        if data:
            try:
                state_dict = json.loads(data)
                state = ReasoningState(**state_dict)
            except Exception as e:
                logger.warning(f"Error loading reasoning state: {e}")
                state = ReasoningState(session_id=session_id)
        else:
            state = ReasoningState(session_id=session_id)

        #region agent log
        try:
            import time as _time, json as _json
            with open("/Users/omar/Downloads/The Future/carebot_dev/.cursor/debug.log", "a") as _f:
                _f.write(_json.dumps({
                    "sessionId": session_id,
                    "runId": "pre-fix",
                    "hypothesisId": "H1",
                    "location": "situation_assessor._get_reasoning_state",
                    "message": "loaded_reasoning_state",
                    "data": {
                        "active_agent": state.active_agent,
                        "awaiting": state.awaiting,
                        "established_intent": state.established_intent,
                    },
                    "timestamp": int(_time.time() * 1000),
                }) + "\n")
        except Exception:
            pass
        #endregion

        return state
    
    def _save_reasoning_state(self, session_id: str, state: ReasoningState):
        """Save reasoning state."""
        key = f"session:{session_id}:reasoning_state"
        data = json.dumps(state.model_dump(), default=str)
        self.state_manager.backend.set(key, data, ttl=self.state_manager.ttl)
    
    def _update_reasoning_state(
        self,
        session_id: str,
        state: ReasoningState,
        assessment: SituationAssessment
    ):
        """Update reasoning state based on assessment."""
        logger.info("🎯 [ReasoningState] ════════════════════════════════════════════════════════")
        logger.info(f"🎯 [ReasoningState] Updating reasoning state for session: {session_id}")
        
        #region agent log
        try:
            import time as _time, json as _json
            with open("/Users/omar/Downloads/The Future/carebot_dev/.cursor/debug.log", "a") as _f:
                _f.write(_json.dumps({
                    "sessionId": session_id,
                    "runId": "pre-fix",
                    "hypothesisId": "H1",
                    "location": "situation_assessor._update_reasoning_state",
                    "message": "before_update",
                    "data": {
                        "awaiting_before": state.awaiting,
                        "presented_options_before": len(state.presented_options or []),
                        "situation_type": getattr(assessment.situation_type, "value", str(assessment.situation_type)),
                        "is_response_to_options": getattr(assessment, "is_response_to_options", None),
                    },
                    "timestamp": int(_time.time() * 1000),
                }) + "\n")
        except Exception:
            pass
        #endregion
        
        # Increment turn counter
        state.increment_turn()
        logger.info(f"🎯 [ReasoningState] Turn count: {state.turns_since_intent_established}")
        
        # Update based on situation type
        if assessment.situation_type == SituationType.NEW_INTENT:
            # Clear established intent for new intent
            logger.info("🎯 [ReasoningState] 🆕 New intent detected - clearing established state")
            state.established_intent = None
            state.active_agent = None
            state.clear_awaiting()
        
        elif assessment.situation_type in [
            SituationType.DIRECT_ANSWER,     # NEW - Add this type
            SituationType.DIRECT_CONTINUATION,
            SituationType.SELECTION,
            SituationType.CONFIRMATION
        ]:
            # User responded to what we're awaiting
            if assessment.is_response_to_awaiting:  # Changed from is_response_to_options
                logger.info("🎯 [ReasoningState] ✅ User responded to awaiting - clearing")
                state.clear_awaiting()
        
        elif assessment.situation_type == SituationType.REJECTION:
            # User rejected our proposal
            logger.info("🎯 [ReasoningState] ❌ User rejected proposal - clearing awaiting")
            state.clear_awaiting()
        
        #region agent log
        try:
            import time as _time, json as _json
            with open("/Users/omar/Downloads/The Future/carebot_dev/.cursor/debug.log", "a") as _f:
                _f.write(_json.dumps({
                    "sessionId": session_id,
                    "runId": "pre-fix",
                    "hypothesisId": "H1",
                    "location": "situation_assessor._update_reasoning_state",
                    "message": "after_update",
                    "data": {
                        "awaiting_after": state.awaiting,
                        "presented_options_after": len(state.presented_options or []),
                    },
                    "timestamp": int(_time.time() * 1000),
                }) + "\n")
        except Exception:
            pass
        #endregion

        # Save updated state
        self._save_reasoning_state(session_id, state)
        logger.info(f"🎯 [ReasoningState] 💾 State saved")


# Global assessor instance
_situation_assessor: Optional[SituationAssessor] = None


def get_situation_assessor() -> SituationAssessor:
    """Get or create the global situation assessor instance."""
    global _situation_assessor
    if _situation_assessor is None:
        _situation_assessor = SituationAssessor()
    return _situation_assessor


def reset_situation_assessor():
    """Reset the global situation assessor (for testing)."""
    global _situation_assessor
    _situation_assessor = None

