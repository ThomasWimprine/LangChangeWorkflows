from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class Atomicity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_atomic: bool = Field(..., description="Whether the task is atomic")
    reasons: List[str] = Field(..., min_length=1)


class ProposedTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    objective: str
    affected_components: List[str] = Field(..., min_length=1)
    dependencies: List[str] = Field(default_factory=list)
    acceptance: List[str] = Field(..., min_length=1)
    risk: List[str] = Field(default_factory=list)
    effort: str = Field(..., pattern="^(S|M|L)$")
    agent: Optional[str] = None


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent: str
    question: str


class Draft001(BaseModel):
    """Schema matching templates/prp/prp-draft-001.json structure."""
    # Allow 'questions' (lowercase) as alias for 'Questions' since Claude may use either
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    agent: str
    description: str
    atomicity: Atomicity
    proposed_tasks: List[ProposedTask] = Field(..., min_length=1)
    # Template uses objects like {"t-001": "description"} not plain strings
    split_recommendation: List[Dict[str, str]] = Field(default_factory=list)
    # Template uses objects like {"agent-name": "reason"} not plain strings
    delegation_suggestions: List[Dict[str, str]] = Field(default_factory=list)
    # Accept both 'Questions' and 'questions' since Claude may use either case
    Questions: List[Question] = Field(default_factory=list, alias="questions")
