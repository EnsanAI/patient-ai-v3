"""
Agent Plan Models - Persistent Execution Plans

Plans are owned by agents and persist across conversation turns.
Each agent generates its own execution plan based on its role and tools.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from enum import Enum


class PlanStatus(str, Enum):
    """Lifecycle status of a plan."""
    CREATED = "created"           # Just generated, not started
    EXECUTING = "executing"       # Actively running tasks
    BLOCKED = "blocked"           # Waiting for user input
    COMPLETE = "complete"         # All tasks done successfully
    FAILED = "failed"             # Unrecoverable failure
    ABANDONED = "abandoned"       # User changed intent


class TaskStatus(str, Enum):
    """Status of a task within a plan."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    BLOCKED = "blocked"
    FAILED = "failed"
    SKIPPED = "skipped"


class PlanTask(BaseModel):
    """A single task in the execution plan."""
    
    # Identity
    id: str                                    # "task_1", "task_2", etc.
    description: str                           # Human-readable description
    
    # Execution
    tool: Optional[str] = None                 # Tool to call (None = info gathering)
    params: Dict[str, Any] = Field(default_factory=dict)
    depends_on: List[str] = Field(default_factory=list)
    
    # Status
    status: TaskStatus = TaskStatus.PENDING
    
    # Results
    result: Optional[Dict[str, Any]] = None
    result_type: Optional[str] = None          # success, blocked, failed, etc.
    
    # Blocking info
    blocked_reason: Optional[str] = None
    blocked_awaiting: Optional[str] = None     # What info we need
    blocked_options: Optional[List[Any]] = None
    
    # Failure info
    failed_reason: Optional[str] = None
    
    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class AgentPlan(BaseModel):
    """
    Persistent execution plan owned by an agent.
    
    Plans survive across conversation turns and maintain
    complete state of what's been done and what remains.
    """
    
    # ═══════════════════════════════════════════════════════════════
    # IDENTITY
    # ═══════════════════════════════════════════════════════════════
    plan_id: str = Field(default_factory=lambda: f"plan_{datetime.utcnow().timestamp()}")
    session_id: str
    agent_name: str                            # Which agent owns this plan
    
    # ═══════════════════════════════════════════════════════════════
    # ORIGIN (from Reasoning Engine)
    # ═══════════════════════════════════════════════════════════════
    objective: str                             # WHAT to achieve
    original_message: str                      # User's exact words
    initial_entities: Dict[str, Any] = Field(default_factory=dict)
    constraints: List[str] = Field(default_factory=list)
    
    # ═══════════════════════════════════════════════════════════════
    # TASKS (the actual plan)
    # ═══════════════════════════════════════════════════════════════
    tasks: List[PlanTask] = Field(default_factory=list)
    
    # ═══════════════════════════════════════════════════════════════
    # EXECUTION STATE
    # ═══════════════════════════════════════════════════════════════
    status: PlanStatus = PlanStatus.CREATED
    current_task_id: Optional[str] = None
    
    # Blocking info (when status=BLOCKED)
    blocked_reason: Optional[str] = None
    awaiting_info: Optional[str] = None        # What we need from user
    presented_options: List[Any] = Field(default_factory=list)
    
    # ═══════════════════════════════════════════════════════════════
    # ACCUMULATED KNOWLEDGE
    # ═══════════════════════════════════════════════════════════════
    resolved_entities: Dict[str, Any] = Field(default_factory=dict)
    
    # ═══════════════════════════════════════════════════════════════
    # LIFECYCLE
    # ═══════════════════════════════════════════════════════════════
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    abandoned_at: Optional[datetime] = None
    abandoned_reason: Optional[str] = None
    
    # ═══════════════════════════════════════════════════════════════
    # HELPER METHODS
    # ═══════════════════════════════════════════════════════════════
    
    def add_task(
        self,
        description: str,
        tool: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        depends_on: Optional[List[str]] = None
    ) -> PlanTask:
        """Add a task to the plan."""
        task_id = f"task_{len(self.tasks) + 1}"
        task = PlanTask(
            id=task_id,
            description=description,
            tool=tool,
            params=params or {},
            depends_on=depends_on or []
        )
        self.tasks.append(task)
        return task
    
    def get_task(self, task_id: str) -> Optional[PlanTask]:
        """Get a task by ID."""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None
    
    def get_completed_task_ids(self) -> List[str]:
        """Get IDs of all completed tasks."""
        return [t.id for t in self.tasks if t.status == TaskStatus.COMPLETE]
    
    def get_next_executable_task(self) -> Optional[PlanTask]:
        """Get the next task that can be executed."""
        completed_ids = self.get_completed_task_ids()
        
        for task in self.tasks:
            if task.status != TaskStatus.PENDING:
                continue
            
            # Check dependencies
            deps_met = all(dep_id in completed_ids for dep_id in task.depends_on)
            if deps_met:
                return task
        
        return None
    
    def all_tasks_complete(self) -> bool:
        """Check if all tasks are complete."""
        return all(t.status == TaskStatus.COMPLETE for t in self.tasks)
    
    def has_failed_tasks(self) -> bool:
        """Check if any task has failed."""
        return any(t.status == TaskStatus.FAILED for t in self.tasks)
    
    def is_blocked(self) -> bool:
        """Check if plan is blocked waiting for user input."""
        return self.status == PlanStatus.BLOCKED
    
    def is_terminal(self) -> bool:
        """Check if plan is in a terminal state."""
        return self.status in [
            PlanStatus.COMPLETE,
            PlanStatus.FAILED,
            PlanStatus.ABANDONED
        ]
    
    def mark_task_complete(self, task_id: str, result: Dict[str, Any]):
        """Mark a task as complete with its result."""
        task = self.get_task(task_id)
        if task:
            task.status = TaskStatus.COMPLETE
            task.result = result
            task.result_type = result.get("result_type", "success")
            task.completed_at = datetime.utcnow()
            
            # Update resolved entities from result
            if result.get("derived_entities"):
                self.resolved_entities.update(result["derived_entities"])
        
        self.updated_at = datetime.utcnow()
    
    def mark_task_blocked(
        self,
        task_id: str,
        reason: str,
        awaiting: str,
        options: Optional[List[Any]] = None
    ):
        """Mark a task as blocked, waiting for user input."""
        task = self.get_task(task_id)
        if task:
            task.status = TaskStatus.BLOCKED
            task.blocked_reason = reason
            task.blocked_awaiting = awaiting
            task.blocked_options = options
        
        self.status = PlanStatus.BLOCKED
        self.blocked_reason = reason
        self.awaiting_info = awaiting
        self.presented_options = options or []
        self.updated_at = datetime.utcnow()
    
    def mark_blocked(
        self,
        reason: str,
        awaiting: str,
        options: Optional[List[Any]] = None
    ):
        """
        Mark the entire plan as blocked, waiting for user input.

        This is used when the agent needs information that isn't tied to a single
        task (e.g., high-level clarification or registration details) but we still
        want the plan lifecycle to reflect a BLOCKED state.
        """
        self.status = PlanStatus.BLOCKED
        self.blocked_reason = reason
        self.awaiting_info = awaiting
        self.presented_options = options or []
        self.updated_at = datetime.utcnow()
    
    def unblock_with_info(self, new_entities: Dict[str, Any]):
        """Unblock the plan with new information from user."""
        self.resolved_entities.update(new_entities)
        self.status = PlanStatus.EXECUTING
        self.blocked_reason = None
        self.awaiting_info = None
        self.presented_options = []
        
        # Unblock the blocked task
        for task in self.tasks:
            if task.status == TaskStatus.BLOCKED:
                task.status = TaskStatus.PENDING
                task.blocked_reason = None
                task.blocked_awaiting = None
                task.blocked_options = None
        
        self.updated_at = datetime.utcnow()
    
    def mark_complete(self):
        """Mark the plan as complete."""
        self.status = PlanStatus.COMPLETE
        self.completed_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def mark_abandoned(self, reason: str):
        """Mark the plan as abandoned."""
        self.status = PlanStatus.ABANDONED
        self.abandoned_at = datetime.utcnow()
        self.abandoned_reason = reason
        self.updated_at = datetime.utcnow()
    
    def get_summary(self) -> str:
        """Get a human-readable summary of the plan."""
        complete = len([t for t in self.tasks if t.status == TaskStatus.COMPLETE])
        total = len(self.tasks)
        return f"Plan '{self.objective[:50]}' - {complete}/{total} tasks complete, status: {self.status.value}"


class PlanAction(str, Enum):
    """What the orchestrator tells the agent to do with plans."""
    NO_PLAN = "no_plan"                 # Skip planning entirely (fast-path, general assistant)
    CREATE_NEW = "create_new"           # Generate a new plan
    RESUME = "resume"                   # Continue existing plan
    UPDATE_AND_RESUME = "update_resume" # Add new info, then resume
    ABANDON_AND_CREATE = "abandon_create"  # Abandon old, create new


