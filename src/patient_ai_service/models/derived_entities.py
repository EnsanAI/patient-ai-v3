"""
Derived Entities - System Resolutions from Tool Calls

This module defines what the SYSTEM has resolved. These entities:
- Come from: Tool results (lookups, validations, bookings)
- Linked to: Patient entities they resolve
- Validity: Auto-invalidate when linked patient entity changes
- Expiry: Some have time-based expiry (e.g., availability checks)
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from enum import Enum


class DerivedEntityStatus(str, Enum):
    """Status of a derived entity."""
    VALID = "valid"                    # Fresh and matches current patient preference
    STALE = "stale"                    # Time-expired but preference unchanged
    INVALIDATED = "invalidated"        # Patient preference changed
    PENDING = "pending"                # Needs resolution


class DerivedEntity(BaseModel):
    """
    A single entity derived from a tool call.
    
    Each derived entity is LINKED to the patient entity it resolves.
    When that patient entity changes, this becomes invalid.
    """
    
    # Identity
    key: str                           # "doctor_uuid", "availability_check", "patient_id"
    
    # The resolved value
    value: Any                         # The actual data (UUID, availability info, etc.)
    
    # Provenance - where did this come from?
    source_tool: str                   # "find_doctor_by_name", "check_availability"
    source_params: Dict[str, Any] = Field(default_factory=dict)  # Params used in tool call
    derived_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Linkage to patient entity
    resolves_patient_entity: Optional[str] = None   # "appointment.doctor_preference"
    resolves_patient_value: Optional[Any] = None    # "Dr. Sarah" - the value we resolved
    
    # Validity
    valid_for: Optional[int] = None    # Seconds until expiry (None = no time expiry)
    status: DerivedEntityStatus = DerivedEntityStatus.VALID
    invalidation_reason: Optional[str] = None
    invalidated_at: Optional[datetime] = None
    
    def is_valid(self) -> bool:
        """Check if entity is still valid (not invalidated or expired)."""
        if self.status == DerivedEntityStatus.INVALIDATED:
            return False
        
        if self.status == DerivedEntityStatus.STALE:
            return False
        
        # Check time-based expiry
        if self.valid_for is not None:
            age_seconds = (datetime.utcnow() - self.derived_at).total_seconds()
            if age_seconds > self.valid_for:
                return False
        
        return True
    
    def is_stale(self) -> bool:
        """Check if entity has time-expired (but not invalidated)."""
        if self.status == DerivedEntityStatus.INVALIDATED:
            return True
        
        if self.valid_for is not None:
            age_seconds = (datetime.utcnow() - self.derived_at).total_seconds()
            return age_seconds > self.valid_for
        
        return False
    
    def age_seconds(self) -> float:
        """Get age in seconds."""
        return (datetime.utcnow() - self.derived_at).total_seconds()
    
    def age_display(self) -> str:
        """Get human-readable age."""
        seconds = self.age_seconds()
        if seconds < 60:
            return f"{int(seconds)} seconds ago"
        elif seconds < 3600:
            return f"{int(seconds / 60)} minutes ago"
        else:
            return f"{int(seconds / 3600)} hours ago"
    
    def invalidate(self, reason: str):
        """Mark entity as invalidated."""
        self.status = DerivedEntityStatus.INVALIDATED
        self.invalidation_reason = reason
        self.invalidated_at = datetime.utcnow()
    
    def matches_patient_value(self, current_patient_value: Any) -> bool:
        """Check if this still resolves the current patient preference."""
        if self.resolves_patient_value is None:
            return True  # No linkage, always valid
        return self.resolves_patient_value == current_patient_value


# Validity rules for different entity types
ENTITY_VALIDITY_RULES: Dict[str, Dict[str, Any]] = {
    # Static entities - valid until patient preference changes
    "doctor_uuid": {
        "valid_for": None,  # No time expiry
        "invalidate_on": ["appointment.doctor_preference"],
    },
    "doctor_info": {
        "valid_for": None,
        "invalidate_on": ["appointment.doctor_preference"],
    },
    "patient_id": {
        "valid_for": None,
        "invalidate_on": [],  # Never invalidates (patient ID doesn't change)
    },
    "clinic_info": {
        "valid_for": None,
        "invalidate_on": [],
    },
    
    # Time-sensitive entities
    "availability_check": {
        "valid_for": 300,  # 5 minutes
        "invalidate_on": [
            "appointment.doctor_preference",
            "appointment.date_preference",
            "appointment.time_preference"
        ],
    },
    "available_slots": {
        "valid_for": 300,  # 5 minutes
        "invalidate_on": [
            "appointment.doctor_preference",
            "appointment.date_preference"
        ],
    },
    
    # Result entities - never cache, always from fresh tool call
    "booking_result": {
        "cacheable": False,
    },
    "cancellation_result": {
        "cacheable": False,
    },
    "registration_result": {
        "cacheable": False,
    },
}


class DerivedEntitiesManager(BaseModel):
    """
    Manages all derived entities for a session.
    
    Handles:
    - Storing entities from tool results
    - Checking validity against patient entities
    - Automatic invalidation when patient preferences change
    - Providing status display for agent prompts
    """
    
    session_id: str
    entities: Dict[str, DerivedEntity] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    def store_entity(
        self,
        key: str,
        value: Any,
        source_tool: str,
        source_params: Dict[str, Any] = None,
        resolves_patient_entity: str = None,
        resolves_patient_value: Any = None,
        valid_for: int = None
    ) -> DerivedEntity:
        """
        Store a derived entity from a tool result.
        
        Args:
            key: Entity key (e.g., "doctor_uuid")
            value: The resolved value
            source_tool: Tool that provided this
            source_params: Parameters used in tool call
            resolves_patient_entity: Patient entity path this resolves
            resolves_patient_value: The patient value we resolved
            valid_for: Seconds until expiry (None = use default from rules)
            
        Returns:
            The stored DerivedEntity
        """
        # Get validity from rules if not specified
        if valid_for is None:
            rules = ENTITY_VALIDITY_RULES.get(key, {})
            valid_for = rules.get("valid_for")
        
        entity = DerivedEntity(
            key=key,
            value=value,
            source_tool=source_tool,
            source_params=source_params or {},
            resolves_patient_entity=resolves_patient_entity,
            resolves_patient_value=resolves_patient_value,
            valid_for=valid_for
        )
        
        self.entities[key] = entity
        self.last_updated_at = datetime.utcnow()
        
        return entity
    
    def get_entity(self, key: str) -> Optional[DerivedEntity]:
        """Get an entity by key."""
        return self.entities.get(key)
    
    def get_valid_entity(self, key: str, patient_entities: 'PatientEntities' = None) -> Optional[DerivedEntity]:
        """
        Get entity only if valid and matches current patient preferences.
        
        Args:
            key: Entity key
            patient_entities: Current patient entities to validate against
            
        Returns:
            DerivedEntity if valid, None otherwise
        """
        entity = self.entities.get(key)
        if entity is None:
            return None
        
        # Check basic validity (not invalidated, not expired)
        if not entity.is_valid():
            return None
        
        # Check against current patient value if we have patient entities
        if patient_entities and entity.resolves_patient_entity:
            current_value = self._get_patient_value(patient_entities, entity.resolves_patient_entity)
            if not entity.matches_patient_value(current_value):
                # Patient preference changed - invalidate
                entity.invalidate(f"Patient changed {entity.resolves_patient_entity} from '{entity.resolves_patient_value}' to '{current_value}'")
                return None
        
        return entity
    
    def _get_patient_value(self, patient_entities: 'PatientEntities', field_path: str) -> Any:
        """Get value from patient entities using dot notation path."""
        parts = field_path.split(".")
        obj = patient_entities
        for part in parts:
            obj = getattr(obj, part, None)
            if obj is None:
                return None
        return obj
    
    def invalidate_for_patient_change(self, changed_field: str, old_value: Any, new_value: Any) -> List[str]:
        """
        Invalidate derived entities affected by a patient preference change.
        
        Args:
            changed_field: Patient entity field that changed (e.g., "appointment.doctor_preference")
            old_value: Previous value
            new_value: New value
            
        Returns:
            List of invalidated entity keys
        """
        invalidated = []
        
        for key, entity in self.entities.items():
            # Check if this entity resolves the changed field
            if entity.resolves_patient_entity == changed_field:
                entity.invalidate(f"Patient changed {changed_field} from '{old_value}' to '{new_value}'")
                invalidated.append(key)
                continue
            
            # Check rules for this entity type
            rules = ENTITY_VALIDITY_RULES.get(key, {})
            invalidate_on = rules.get("invalidate_on", [])
            
            if changed_field in invalidate_on:
                entity.invalidate(f"Dependent on {changed_field} which changed")
                invalidated.append(key)
        
        if invalidated:
            self.last_updated_at = datetime.utcnow()
        
        return invalidated
    
    def get_resolution_status(self, patient_entities: 'PatientEntities') -> Dict[str, Any]:
        """
        Get status of all derived entities vs current patient preferences.
        
        Returns dict with:
        - ready_to_proceed: bool - all needed entities are valid
        - valid_entities: dict of valid entity key -> value
        - stale_entities: dict of stale/invalidated entities with reasons
        - needs_resolution: list of entities that need to be resolved
        """
        status = {
            "ready_to_proceed": True,
            "valid_entities": {},
            "stale_entities": {},
            "needs_resolution": []
        }
        
        # Check doctor resolution
        if patient_entities.appointment.doctor_preference:
            doctor_entity = self.get_valid_entity("doctor_uuid", patient_entities)
            if doctor_entity:
                status["valid_entities"]["doctor_uuid"] = doctor_entity.value
            else:
                status["ready_to_proceed"] = False
                existing = self.entities.get("doctor_uuid")
                if existing:
                    status["stale_entities"]["doctor_uuid"] = {
                        "was_for": existing.resolves_patient_value,
                        "patient_now_wants": patient_entities.appointment.doctor_preference,
                        "reason": existing.invalidation_reason
                    }
                status["needs_resolution"].append({
                    "entity": "doctor_uuid",
                    "patient_wants": patient_entities.appointment.doctor_preference,
                    "action": f"Call find_doctor_by_name or list_doctors"
                })
        
        # Check availability resolution
        if patient_entities.appointment.time_preference and patient_entities.appointment.date_preference:
            avail_entity = self.get_valid_entity("availability_check", patient_entities)
            if avail_entity:
                status["valid_entities"]["availability_check"] = avail_entity.value
            else:
                status["ready_to_proceed"] = False
                existing = self.entities.get("availability_check")
                if existing:
                    status["stale_entities"]["availability_check"] = {
                        "was_for": existing.resolves_patient_value,
                        "reason": existing.invalidation_reason or "expired"
                    }
                status["needs_resolution"].append({
                    "entity": "availability_check",
                    "patient_wants": f"{patient_entities.appointment.date_preference} at {patient_entities.appointment.time_preference}",
                    "action": "Call check_availability"
                })
        
        # Check patient_id resolution
        patient_id_entity = self.get_valid_entity("patient_id", patient_entities)
        if patient_id_entity:
            status["valid_entities"]["patient_id"] = patient_id_entity.value
        
        return status
    
    def get_status_display(self, patient_entities: 'PatientEntities') -> str:
        """
        Generate prompt-friendly status display for agent.
        
        This is what the agent sees to understand what's cached vs needs fetching.
        """
        lines = []
        lines.append("═" * 60)
        lines.append("DERIVED ENTITIES STATUS")
        lines.append("═" * 60)
        
        if not self.entities:
            lines.append("No entities resolved yet.")
            return "\n".join(lines)
        
        for key, entity in self.entities.items():
            # Determine current status
            is_valid = entity.is_valid()
            
            # Check against current patient value
            patient_mismatch = False
            if entity.resolves_patient_entity and patient_entities:
                current_value = self._get_patient_value(patient_entities, entity.resolves_patient_entity)
                patient_mismatch = not entity.matches_patient_value(current_value)
            
            if entity.status == DerivedEntityStatus.INVALIDATED:
                status_icon = "✗"
                status_text = f"INVALIDATED - {entity.invalidation_reason}"
            elif patient_mismatch:
                status_icon = "✗"
                current_value = self._get_patient_value(patient_entities, entity.resolves_patient_entity)
                status_text = f"MISMATCH - was for '{entity.resolves_patient_value}', patient now wants '{current_value}'"
            elif entity.is_stale():
                status_icon = "⚠"
                status_text = f"STALE - checked {entity.age_display()}"
            else:
                status_icon = "✓"
                status_text = f"VALID ({entity.age_display()})"
            
            lines.append(f"\n{status_icon} {key}: {entity.value}")
            lines.append(f"  └─ Source: {entity.source_tool}")
            if entity.resolves_patient_entity:
                lines.append(f"  └─ Resolves: {entity.resolves_patient_entity} = '{entity.resolves_patient_value}'")
            lines.append(f"  └─ Status: {status_text}")
        
        lines.append("")
        lines.append("═" * 60)
        
        return "\n".join(lines)
    
    def clear(self):
        """Clear all derived entities."""
        self.entities.clear()
        self.last_updated_at = datetime.utcnow()
    
    def clear_stale(self) -> List[str]:
        """Clear all stale/invalidated entities. Returns list of cleared keys."""
        cleared = []
        for key, entity in list(self.entities.items()):
            if not entity.is_valid() or entity.is_stale():
                del self.entities[key]
                cleared.append(key)
        
        if cleared:
            self.last_updated_at = datetime.utcnow()
        
        return cleared

