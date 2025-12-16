"""
Focused Prompts Library

Small, task-specific prompts for handling messages that don't need
full reasoning. Used when Situation Assessor determines focused handling.
"""

from typing import Dict, Any, Optional
from patient_ai_service.models.situation_assessment import SituationType


class FocusedPrompts:
    """Library of focused prompts for specific situations."""
    
    @staticmethod
    def get_prompt(
        situation_type: SituationType,
        context: Dict[str, Any]
    ) -> Optional[str]:
        """
        Get focused prompt for a situation type.
        
        Args:
            situation_type: Type of situation
            context: Relevant context for the prompt
            
        Returns:
            Focused prompt string, or None if comprehensive reasoning needed
        """
        prompt_map = {
            SituationType.DIRECT_CONTINUATION: FocusedPrompts._confirmation_prompt,
            SituationType.SELECTION: FocusedPrompts._selection_prompt,
            SituationType.CONFIRMATION: FocusedPrompts._confirmation_prompt,
            SituationType.REJECTION: FocusedPrompts._rejection_prompt,
            SituationType.MODIFICATION: FocusedPrompts._modification_prompt,
            SituationType.GREETING: FocusedPrompts._greeting_prompt,
            SituationType.FAREWELL: FocusedPrompts._farewell_prompt,
            SituationType.CLARIFICATION_RESPONSE: FocusedPrompts._clarification_prompt,
        }
        
        prompt_func = prompt_map.get(situation_type)
        if prompt_func:
            return prompt_func(context)
        
        return None
    
    @staticmethod
    def _confirmation_prompt(context: Dict[str, Any]) -> str:
        """Prompt for handling confirmations."""
        awaiting = context.get("awaiting", "confirmation")
        last_proposal = context.get("last_proposal", "the proposed action")
        
        return f"""User confirmed: "{context.get('message', 'yes')}"

CONTEXT:
- We were waiting for: {awaiting}
- Last proposal: {last_proposal}

TASK:
The user has confirmed. Proceed with the action that was proposed.

Output JSON:
{{
    "action": "proceed",
    "confirmed_item": "what was confirmed",
    "next_step": "what to do now"
}}"""
    
    @staticmethod
    def _selection_prompt(context: Dict[str, Any]) -> str:
        """Prompt for handling selections from options."""
        options = context.get("presented_options", [])
        selected = context.get("extracted_entities", {}).get("selected_option")
        
        return f"""User selected from options: "{context.get('message', '')}"

OPTIONS PRESENTED:
{options}

EXTRACTED SELECTION: {selected}

TASK:
Match user's selection to one of the presented options.

Output JSON:
{{
    "matched_option": "the matched option or null",
    "match_confidence": 0.0-1.0,
    "selected_value": "the value to use",
    "needs_clarification": true/false
}}"""
    
    @staticmethod
    def _rejection_prompt(context: Dict[str, Any]) -> str:
        """Prompt for handling rejections."""
        last_proposal = context.get("last_proposal", "the proposal")
        
        return f"""User rejected: "{context.get('message', 'no')}"

CONTEXT:
- Last proposal: {last_proposal}

TASK:
Handle the rejection gracefully. Determine if user wants alternatives or to cancel.

Output JSON:
{{
    "rejection_type": "want_alternatives|cancel|other",
    "user_preference": "what they seem to want instead",
    "suggested_response": "how to respond"
}}"""
    
    @staticmethod
    def _modification_prompt(context: Dict[str, Any]) -> str:
        """Prompt for handling modifications."""
        current_prefs = context.get("current_preferences", {})
        changes = context.get("entity_changes", {})
        
        return f"""User wants to modify: "{context.get('message', '')}"

CURRENT PREFERENCES:
{current_prefs}

DETECTED CHANGES:
{changes}

TASK:
Extract the specific changes the user wants to make.

Output JSON:
{{
    "changes": {{
        "field_name": "new_value"
    }},
    "keep_unchanged": ["fields to keep as is"],
    "needs_revalidation": ["fields that need tool calls to validate"]
}}"""
    
    @staticmethod
    def _greeting_prompt(context: Dict[str, Any]) -> str:
        """Prompt for handling greetings."""
        is_registered = context.get("patient_info", {}).get("patient_id")
        patient_name = context.get("patient_info", {}).get("first_name", "")
        
        return f"""User greeting: "{context.get('message', 'hello')}"

PATIENT: {'Registered' if is_registered else 'Not registered'}
NAME: {patient_name or 'Unknown'}

Generate a warm, appropriate greeting response.

Output JSON:
{{
    "response": "greeting response",
    "suggest_registration": true/false,
    "suggest_help": true/false
}}"""
    
    @staticmethod
    def _farewell_prompt(context: Dict[str, Any]) -> str:
        """Prompt for handling farewells."""
        had_action = context.get("had_successful_action", False)
        
        return f"""User farewell: "{context.get('message', 'bye')}"

HAD SUCCESSFUL ACTION: {had_action}

Generate appropriate farewell.

Output JSON:
{{
    "response": "farewell response",
    "include_summary": true/false
}}"""
    
    @staticmethod
    def _clarification_prompt(context: Dict[str, Any]) -> str:
        """Prompt for handling clarification responses."""
        awaiting = context.get("awaiting", "information")
        
        return f"""User clarification: "{context.get('message', '')}"

WE ASKED FOR: {awaiting}

TASK:
Extract the information user provided.

Output JSON:
{{
    "extracted_info": {{
        "field": "value"
    }},
    "is_complete": true/false,
    "still_missing": ["fields still needed"]
}}"""


class FocusedResponseGenerator:
    """
    Generates responses using focused prompts.
    
    Used when Situation Assessor determines focused handling is sufficient.
    """
    
    def __init__(self, llm_client=None):
        from patient_ai_service.core.llm import get_llm_client
        self.llm_client = llm_client or get_llm_client()
    
    async def generate(
        self,
        situation_type: SituationType,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate response using focused prompt.
        
        Args:
            situation_type: Type of situation
            context: Context for prompt
            
        Returns:
            Parsed response dict
        """
        prompt = FocusedPrompts.get_prompt(situation_type, context)
        
        if not prompt:
            return {"needs_comprehensive": True}
        
        try:
            # Check if create_message is async
            import asyncio
            if asyncio.iscoroutinefunction(self.llm_client.create_message):
                response = await self.llm_client.create_message(
                    system="You are a fast response generator. Output ONLY valid JSON.",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=300
                )
            else:
                response = self.llm_client.create_message(
                    system="You are a fast response generator. Output ONLY valid JSON.",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=300
                )
            
            # Parse response
            import json
            response = response.strip()
            if response.startswith("{"):
                return json.loads(response)
            
            # Try to extract JSON
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(response[start:end])
            
            return {"needs_comprehensive": True}
            
        except Exception as e:
            import logging
            logging.warning(f"Focused response generation failed: {e}")
            return {"needs_comprehensive": True}

