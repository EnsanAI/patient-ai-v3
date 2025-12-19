"""
Unified Reasoning Engine.

Single-pass LLM call that replaces SituationAssessor + ReasoningEngine
for message classification, routing, and plan lifecycle decisions.
"""

import json
import logging
import time
from typing import Optional, Dict, Any, List

from patient_ai_service.core.llm import LLMClient, get_llm_client
from patient_ai_service.core.config import settings
from patient_ai_service.core.observability import get_observability_logger
from patient_ai_service.models.unified_reasoning import (
    UnifiedReasoningOutput,
    RouteType,
    PlanDecision,
)
from patient_ai_service.models.situation_assessment import SituationType
from patient_ai_service.models.agent_plan import AgentPlan
from patient_ai_service.models.observability import TokenUsage

logger = logging.getLogger(__name__)


# Agent capabilities for plan context
AGENT_CAPABILITIES = {
    "registration": "Registers new patients (name, phone, DOB, gender). Cannot book appointments.",
    "appointment_manager": "Books, reschedules, cancels appointments. Checks doctor availability.",
    "emergency_response": "Handles urgent medical situations requiring immediate attention.",
    "general_assistant": "Answers general questions about the clinic, hours, location, services.",
    "medical_inquiry": "Answers medical and dental health questions.",
}


class UnifiedReasoning:
    """
    Single-pass reasoning engine.

    Combines situation assessment, routing, and plan lifecycle decisions
    into one LLM call.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or get_llm_client()
        logger.info("UnifiedReasoning initialized")

    async def reason(
        self,
        session_id: str,
        message: str,
        patient_info: Dict[str, Any],
        active_agent: Optional[str],
        awaiting: Optional[str],
        awaiting_context: Optional[str],              # NEW
        pending_action: Optional[Dict[str, Any]],     # NEW
        recent_turns: List[Dict[str, str]],
        existing_plan: Optional[AgentPlan]
    ) -> UnifiedReasoningOutput:
        """
        Perform unified reasoning in a single LLM call.

        Args:
            session_id: Session identifier
            message: User's message (in English)
            patient_info: Patient profile info
            active_agent: Currently active agent (if any)
            awaiting: What we're waiting for (if any)
            awaiting_context: Human-readable summary of what we're awaiting (NEW)
            pending_action: Structured pending action details for confirmations (NEW)
            recent_turns: Recent conversation turns
            existing_plan: Current active plan (if any)

        Returns:
            UnifiedReasoningOutput with routing decision
        """
        start_time = time.time()

        logger.info("=" * 80)
        logger.info(f"[UnifiedReasoning] Session: {session_id}")
        logger.info(f"[UnifiedReasoning] Message: {message[:100]}...")
        logger.info(f"[UnifiedReasoning] Active agent: {active_agent}")
        logger.info(f"[UnifiedReasoning] Awaiting: {awaiting}")
        logger.info(f"[UnifiedReasoning] Awaiting context: {awaiting_context}")  # NEW
        logger.info(f"[UnifiedReasoning] Has pending action: {pending_action is not None}")  # NEW
        logger.info(f"[UnifiedReasoning] Has plan: {existing_plan is not None}")

        # Get observability logger
        obs_logger = get_observability_logger(session_id) if settings.enable_observability else None

        # Build prompt
        prompt = self._build_prompt(
            message=message,
            patient_info=patient_info,
            active_agent=active_agent,
            awaiting=awaiting,
            awaiting_context=awaiting_context,    # NEW
            pending_action=pending_action,         # NEW
            recent_turns=recent_turns,
            existing_plan=existing_plan
        )

        # Log the built prompt
        logger.info("=" * 80)
        logger.info("[UnifiedReasoning] BUILT PROMPT:")
        logger.info("=" * 80)
        logger.info(f"[UnifiedReasoning] System Prompt:\n{self._get_system_prompt()}")
        logger.info("-" * 80)
        logger.info(f"[UnifiedReasoning] User Prompt:\n{prompt}")
        logger.info("=" * 80)

        try:
            # Make LLM call with token tracking
            llm_start_time = time.time()
            system_prompt = self._get_system_prompt()
            
            if hasattr(self.llm_client, 'create_message_with_usage'):
                response_text, tokens = self.llm_client.create_message_with_usage(
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=600
                )
            else:
                # Fallback if method doesn't exist - use create_message
                response_text = self.llm_client.create_message(
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=600
                )
                tokens = TokenUsage()

            llm_duration_seconds = time.time() - llm_start_time

            # Log the raw LLM response
            logger.info("=" * 80)
            logger.info("[UnifiedReasoning] RAW LLM RESPONSE:")
            logger.info("=" * 80)
            logger.info(f"[UnifiedReasoning] Response Content:\n{response_text}")
            logger.info("=" * 80)

            # Record LLM call in observability
            if obs_logger:
                obs_logger.record_llm_call(
                    component="unified_reasoning",
                    provider=settings.llm_provider.value,
                    model=settings.get_llm_model(),
                    tokens=tokens,
                    duration_seconds=llm_duration_seconds,
                    system_prompt_length=len(system_prompt),
                    messages_count=1,
                    temperature=0.1,
                    max_tokens=600
                )

            # Parse response
            output = self._parse_response(response_text)

            duration_ms = (time.time() - start_time) * 1000
            logger.info(f"[UnifiedReasoning] Completed in {duration_ms:.0f}ms")
            logger.info(f"[UnifiedReasoning] Route: {output.route_type.value}")
            logger.info(f"[UnifiedReasoning] Situation: {output.situation_type.value}")
            if output.agent:
                logger.info(f"[UnifiedReasoning] Agent: {output.agent}")
                logger.info(f"[UnifiedReasoning] Plan decision: {output.plan_decision.value if output.plan_decision else 'N/A'}")
            logger.info("=" * 80)

            return output

        except Exception as e:
            logger.error(f"[UnifiedReasoning] Error: {e}", exc_info=True)
            # Fallback: route to general_assistant
            return UnifiedReasoningOutput(
                route_type=RouteType.AGENT,
                situation_type=SituationType.AMBIGUOUS,
                confidence=0.3,
                agent="general_assistant",
                plan_decision=PlanDecision.NO_PLAN,
                plan_reasoning="Fallback due to reasoning error",
                what_user_means=message,
                objective=""
            )

    def _get_system_prompt(self) -> str:
        """Get system prompt for unified reasoning."""
        return """You are the unified reasoning engine for a clinic AI.

═══════════════════════════════════════════════════════════════════════════════
SITUATION TYPES
═══════════════════════════════════════════════════════════════════════════════

FAST-PATH (route_type="fast_path", plan_decision="no_plan"):
- greeting, farewell, thanks, pleasantry

CONTINUATION (route_type="agent", usually resume plan):
- direct_answer: Answers what we're awaiting
- selection: Picks from presented options
- confirmation: Confirms proposed action
- rejection: Declines/cancels
- modification: Changes something already set

NEW INTENT (route_type="agent", plan_decision="create_new" or "abandon_create"):
- new_intent: New request OR confirmation of something active agent CANNOT do
- topic_shift: Changing subjects entirely

SPECIAL:
- emergency: Route to emergency_response immediately
- ambiguous: Route to general_assistant for clarification

═══════════════════════════════════════════════════════════════════════════════
PLAN DECISIONS
═══════════════════════════════════════════════════════════════════════════════

no_plan: IF routing decision IS fast_path or general_assistant
create_new: New intent, no existing plan
resume: Continue existing plan unchanged
abandon_create: New intent/topic_shift with existing plan to abandon

═══════════════════════════════════════════════════════════════════════════════
TASK
═══════════════════════════════════════════════════════════════════════════════

Analyze the user message in context. Determine:
1. What type of situation is this?
2. Does this need an agent, and if so, which one can fulfill what the user actually wants?
3. What should happen with the current plan (if any)?
4. What is the user trying to achieve?

Consider: What the user wants may differ from what the active agent can provide.

═══════════════════════════════════════════════════════════════════════════════
OUTPUT
═══════════════════════════════════════════════════════════════════════════════

FOR FAST-PATH (greeting, farewell, thanks, pleasantry):
{"route_type": "fast_path", "situation_type": "..."}

FOR AGENT ROUTING:
{
  "route_type": "agent",
  "situation_type": "...",
  "confidence": 0.0-1.0,
  "agent": "registration|appointment_manager|emergency_response|general_assistant|medical_inquiry",
  "plan_decision": "no_plan|create_new|resume|abandon_create",
  "plan_reasoning": "Why this plan decision",
  "what_user_means": "Plain English explanation of what user actually wants",
  "objective": "Goal for the agent (empty if resume)",
  "is_continuation": true|false,
  "continuation_type": "selection|confirmation|direct_answer|rejection|modification|null",
  "routing_action": "execute_confirmed_action|null"
}

IMPORTANT: When a pending action is awaiting confirmation:
- If user confirms (yes, yeah, sure, okay, etc.) → routing_action MUST be "execute_confirmed_action"
- If user rejects (no, cancel, etc.) → routing_action is null (agent will think normally)
- If user modifies (yes but..., change to..., etc.) → routing_action is null (agent will think normally)
- Otherwise → routing_action is null

Note: Only explicit confirmations get special routing. Rejections and modifications
go through normal agent thinking where _think() will decide the appropriate response.

JSON only. No explanation outside JSON."""

    def _build_prompt(
        self,
        message: str,
        patient_info: Dict[str, Any],
        active_agent: Optional[str],
        awaiting: Optional[str],
        awaiting_context: Optional[str],              # NEW
        pending_action: Optional[Dict[str, Any]],     # NEW
        recent_turns: List[Dict[str, str]],
        existing_plan: Optional[AgentPlan]
    ) -> str:
        """Build the unified reasoning prompt."""

        # Format patient info
        patient_id = patient_info.get("patient_id", "Not registered")
        patient_name = f"{patient_info.get('first_name', 'Unknown')} {patient_info.get('last_name', '')}".strip()
        is_registered = bool(patient_info.get("patient_id"))

        # Format recent turns
        if recent_turns:
            turns_formatted = "\n".join([
                f"{'User' if t['role'] == 'user' else 'Assistant'}: {t['content'][:200]}"
                for t in recent_turns[-6:]
            ])
        else:
            turns_formatted = "(No previous messages)"

        # Build plan context
        plan_context = self._build_plan_context(existing_plan, active_agent)

        # Build registration status section - ONLY when patient is NOT registered
        registration_status_section = ""
        if not is_registered:
            registration_status_section = """
═══════════════════════════════════════════════════════════════════════════════
REGISTRATION STATUS - IMPORTANT
═══════════════════════════════════════════════════════════════════════════════

⚠️ PATIENT IS NOT REGISTERED

The patient can ONLY inquire about the clinic through the general_assistant agent.

For any other actions (booking appointments, managing appointments, etc.),
the patient MUST register first through the registration agent.

ROUTING RULES FOR UNREGISTERED PATIENTS:
- General inquiries about the clinic → general_assistant (allowed)
- Appointment booking, management, or any patient-specific actions → registration (must register first)
- Medical inquiries → registration (must register first)
- Emergency situations → emergency_response (always allowed)

"""

        # NEW: Build pending action section for confirmations
        pending_action_section = ""
        if awaiting == "confirmation" and pending_action:
            pending_action_section = f"""
═══════════════════════════════════════════════════════════════════════════════
PENDING ACTION AWAITING CONFIRMATION
═══════════════════════════════════════════════════════════════════════════════

ACTION TYPE: {pending_action.get('action_type', 'unknown')}
SUMMARY: {pending_action.get('summary', awaiting_context or 'Pending action')}
TOOL TO EXECUTE: {pending_action.get('tool', 'unknown')}
TOOL PARAMETERS: {json.dumps(pending_action.get('tool_input', {}), default=str)}

ROUTING RULES FOR THIS CONFIRMATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

If user CONFIRMS (yes, yeah, sure, okay, do it, proceed, confirm, go ahead, yep, yup, absolutely, please do):
  → routing_action: "execute_confirmed_action"
  → situation_type: "confirmation"

If user REJECTS (no, nope, cancel, nevermind, don't, stop, forget it, actually no):
  → routing_action: null (let agent think normally)
  → situation_type: "rejection"
  → continuation_type: "rejection"
  → plan_decision: "resume"

If user MODIFIES (yes but at 4pm, change to Dr. X, different day, make it earlier):
  → routing_action: null (let agent think normally)
  → situation_type: "modification"
  → continuation_type: "modification"
  → plan_decision: "resume"
  → Extract the modification in objective/what_user_means

If user asks something UNRELATED (what are your hours?, who is Dr. X?, something completely different):
  → routing_action: null
  → situation_type: "new_intent" or "topic_shift"
  → Route to appropriate agent (this abandons the pending action)

"""

        return f"""═══════════════════════════════════════════════════════════════════════════════
CONTEXT
═══════════════════════════════════════════════════════════════════════════════

USER MESSAGE: "{message}"

PATIENT: {patient_name} (ID: {patient_id}, Registered: {"Yes" if is_registered else "No"})

ACTIVE AGENT: {active_agent or "None"}
AWAITING: {awaiting or "Nothing"}
AWAITING CONTEXT: {awaiting_context or "N/A"}

RECENT CONVERSATION:
{turns_formatted}
{registration_status_section}
{pending_action_section}
{plan_context}

Analyze and respond with JSON only."""

    def _build_plan_context(
        self,
        existing_plan: Optional[AgentPlan],
        active_agent: Optional[str]
    ) -> str:
        """Build plan context with agent capabilities."""

        lines = ["═══════════════════════════════════════════════════════════════════════════════",
                 "PLAN STATE",
                 "═══════════════════════════════════════════════════════════════════════════════"]

        if existing_plan:
            lines.append(f"Plan exists: YES")
            lines.append(f"Plan agent: {existing_plan.agent_name}")
            lines.append(f"Plan objective: {existing_plan.objective}")
            lines.append(f"Plan status: {'BLOCKED - waiting for: ' + (existing_plan.awaiting_info or 'user input') if existing_plan.is_blocked() else existing_plan.status.value}")
            if existing_plan.resolved_entities:
                lines.append(f"Resolved entities: {existing_plan.resolved_entities}")
        else:
            lines.append("Plan exists: NO")

        # Add agent capability context (critical for the handoff bug fix)
        if active_agent and active_agent in AGENT_CAPABILITIES:
            lines.append("")
            lines.append(f"ACTIVE AGENT CAPABILITY:")
            lines.append(f"{active_agent}: {AGENT_CAPABILITIES[active_agent]}")
            lines.append("")
            lines.append("⚠️  If user wants something this agent CANNOT do → treat as new_intent")

        return "\n".join(lines)

    def _parse_response(self, response: str) -> UnifiedReasoningOutput:
        """Parse LLM response to structured output."""

        # Extract JSON from response
        response = response.strip()

        # Handle markdown code blocks
        if response.startswith("```"):
            lines = response.split("\n")
            json_lines = []
            in_json = False
            for line in lines:
                if line.startswith("```") and not in_json:
                    in_json = True
                    continue
                elif line.startswith("```") and in_json:
                    break
                elif in_json:
                    json_lines.append(line)
            response = "\n".join(json_lines)

        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"[UnifiedReasoning] JSON parse error: {e}")
            logger.error(f"[UnifiedReasoning] Response was: {response[:500]}")
            raise

        # Handle fast-path (minimal output)
        if data.get("route_type") == "fast_path":
            return UnifiedReasoningOutput(
                route_type=RouteType.FAST_PATH,
                situation_type=SituationType(data.get("situation_type", "greeting"))
            )

        # Handle full agent routing output
        return UnifiedReasoningOutput(
            route_type=RouteType(data.get("route_type", "agent")),
            situation_type=SituationType(data.get("situation_type", "new_intent")),
            confidence=data.get("confidence", 0.8),
            agent=data.get("agent", "general_assistant"),
            plan_decision=PlanDecision(data.get("plan_decision", "no_plan")) if data.get("plan_decision") else PlanDecision.NO_PLAN,
            plan_reasoning=data.get("plan_reasoning", ""),
            what_user_means=data.get("what_user_means", ""),
            objective=data.get("objective", ""),
            is_continuation=data.get("is_continuation", False),
            continuation_type=data.get("continuation_type"),
            routing_action=data.get("routing_action")  # NEW
        )


# Singleton management
_unified_reasoning: Optional[UnifiedReasoning] = None


def get_unified_reasoning() -> UnifiedReasoning:
    """Get singleton UnifiedReasoning instance."""
    global _unified_reasoning
    if _unified_reasoning is None:
        _unified_reasoning = UnifiedReasoning()
    return _unified_reasoning


def reset_unified_reasoning():
    """Reset singleton (for testing)."""
    global _unified_reasoning
    _unified_reasoning = None

