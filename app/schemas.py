from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class WorkflowStep(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    type: Literal[
        "TRANSFORM_TEXT",
        "CALCULATE",
        "CLASSIFY_TEXT",
    ]
    config: dict[str, Any]


class WorkflowCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    steps: list[WorkflowStep] = Field(min_length=1)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Workflow name cannot be empty.")

        return value


class WorkflowDescription(BaseModel):
    description: str = Field(min_length=5, max_length=2000)


class WorkflowExecutionRequest(BaseModel):
    input: dict[str, Any]


class WorkflowResponse(BaseModel):
    id: int
    name: str
    steps: list[dict[str, Any]]


class WorkflowExecutionResponse(BaseModel):
    id: int
    workflow_id: int
    status: Literal["COMPLETED", "FAILED"]
    input_data: dict[str, Any]
    output_data: dict[str, Any]
    step_results: list[dict[str, Any]]
    retry_count: int
    error: str | None