from typing import List, Optional
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
    model_config = ConfigDict(extra="forbid")

    agent: str
    description: str
    atomicity: Atomicity
    proposed_tasks: List[ProposedTask] = Field(..., min_length=1)
    split_recommendation: List[str] = Field(default_factory=list)
    delegation_suggestions: List[str] = Field(default_factory=list)
    Questions: List[Question] = Field(default_factory=list)
