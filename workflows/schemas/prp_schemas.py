"""
Pydantic schemas for PRP validation.

Enforces structural and mathematical correctness of PRPs before execution.
This is Layer 2 in the validation stack - catches math errors, invalid
dependencies, and structural issues.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, field_validator, model_validator
from enum import Enum


class EffortSize(str, Enum):
    """Task effort estimation"""
    SMALL = "S"
    MEDIUM = "M"
    LARGE = "L"


class Subtask(BaseModel):
    """Individual subtask within a task"""
    id: str
    description: str
    hours: float
    dependencies: List[str] = []

    @field_validator("hours")
    @classmethod
    def hours_positive(cls, v: float) -> float:
        """Subtask hours must be positive"""
        if v <= 0:
            raise ValueError("Subtask hours must be > 0")
        return v

    @field_validator("id")
    @classmethod
    def id_not_empty(cls, v: str) -> str:
        """Subtask ID cannot be empty"""
        if not v.strip():
            raise ValueError("Subtask ID cannot be empty")
        return v.strip()


class Task(BaseModel):
    """PRP task with subtasks and effort estimation"""
    id: str
    objective: str
    affected_components: List[str]
    subtasks: List[Subtask]
    total_hours: float
    effort: EffortSize
    dependencies: List[str] = []
    acceptance_criteria: List[str]
    risks: List[str]

    @field_validator("id")
    @classmethod
    def id_not_empty(cls, v: str) -> str:
        """Task ID cannot be empty"""
        if not v.strip():
            raise ValueError("Task ID cannot be empty")
        return v.strip()

    @field_validator("total_hours")
    @classmethod
    def total_hours_positive(cls, v: float) -> float:
        """Total hours must be positive"""
        if v <= 0:
            raise ValueError("Total hours must be > 0")
        return v

    @model_validator(mode="after")
    def validate_hours_math(self):
        """
        Math check: total_hours must equal sum of subtask hours.
        This catches AI hallucinations where totals don't add up.
        """
        calculated = sum(st.hours for st in self.subtasks)
        if abs(calculated - self.total_hours) > 0.01:
            raise ValueError(
                f"Math error in task {self.id}: total_hours={self.total_hours} "
                f"but subtasks sum to {calculated}"
            )
        return self

    @model_validator(mode="after")
    def validate_subtask_dependencies(self):
        """
        All subtask dependencies must reference existing subtasks.
        This catches AI hallucinations where dependencies don't exist.
        """
        subtask_ids = {st.id for st in self.subtasks}
        for st in self.subtasks:
            for dep in st.dependencies:
                if dep not in subtask_ids:
                    raise ValueError(
                        f"Subtask {st.id} depends on unknown subtask '{dep}'"
                    )
        return self

    @model_validator(mode="after")
    def validate_has_subtasks(self):
        """Tasks must have at least one subtask"""
        if not self.subtasks:
            raise ValueError(f"Task {self.id} has no subtasks")
        return self


class PRPPlan(BaseModel):
    """Complete PRP plan with tasks and metadata"""
    prp_id: str
    prp_name: str
    branch_name: str
    tasks: List[Task]
    total_effort_hours: float
    security_required: bool = False
    architecture_review_required: bool = True

    @field_validator("prp_id")
    @classmethod
    def prp_id_format(cls, v: str) -> str:
        """PRP ID must match pattern: ###-[a-z]-###"""
        import re
        if not re.match(r"^\d{3}-[a-z]-\d{3}$", v):
            raise ValueError(
                f"PRP ID '{v}' must match pattern: ###-[a-z]-### "
                f"(e.g., '007-a-002')"
            )
        return v

    @model_validator(mode="after")
    def validate_unique_task_ids(self):
        """
        All task IDs must be unique.
        This catches AI hallucinations where tasks are duplicated.
        """
        task_ids = [t.id for t in self.tasks]
        duplicates = [tid for tid in task_ids if task_ids.count(tid) > 1]
        if duplicates:
            raise ValueError(
                f"Duplicate task IDs found: {set(duplicates)}"
            )
        return self

    @model_validator(mode="after")
    def validate_task_dependencies(self):
        """
        All task dependencies must reference existing tasks.
        This catches AI hallucinations where dependencies don't exist.
        """
        task_ids = {t.id for t in self.tasks}
        for task in self.tasks:
            for dep in task.dependencies:
                if dep not in task_ids:
                    raise ValueError(
                        f"Task {task.id} depends on unknown task '{dep}'"
                    )
        return self

    @model_validator(mode="after")
    def validate_total_hours_math(self):
        """
        Math check: total must equal sum of all task hours.
        This catches AI hallucinations where totals don't add up.
        """
        calculated = sum(t.total_hours for t in self.tasks)
        if abs(calculated - self.total_effort_hours) > 0.01:
            raise ValueError(
                f"Math error: total_effort_hours={self.total_effort_hours} "
                f"but tasks sum to {calculated}"
            )
        return self

    @model_validator(mode="after")
    def validate_has_tasks(self):
        """PRP must have at least one task"""
        if not self.tasks:
            raise ValueError(f"PRP {self.prp_id} has no tasks")
        return self


class ValidationResult(BaseModel):
    """Result from a validation layer"""
    layer_name: str  # "reading_check", "pydantic", "embedding", etc.
    passed: bool
    errors: List[str] = []
    warnings: List[str] = []
    confidence: float  # 0.0 to 1.0
    details: Optional[Dict[str, Any]] = None

    @field_validator("confidence")
    @classmethod
    def confidence_in_range(cls, v: float) -> float:
        """Confidence must be between 0.0 and 1.0"""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v

    @field_validator("layer_name")
    @classmethod
    def layer_name_not_empty(cls, v: str) -> str:
        """Layer name cannot be empty"""
        if not v.strip():
            raise ValueError("Layer name cannot be empty")
        return v.strip()


class PRPMetadata(BaseModel):
    """Metadata extracted from PRP file"""
    prp_id: str
    prp_name: str
    file_path: str
    prp_type: str  # "draft" or "active"
    branch_name: str
    pr_number: Optional[int] = None

    @field_validator("prp_type")
    @classmethod
    def valid_prp_type(cls, v: str) -> str:
        """PRP type must be 'draft' or 'active'"""
        if v not in ["draft", "active"]:
            raise ValueError("PRP type must be 'draft' or 'active'")
        return v
