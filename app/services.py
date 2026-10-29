from typing import Any

from sqlalchemy.orm import Session

from app.llm import classify_text
from app.models import Workflow, WorkflowExecution


def create_workflow(
    db: Session,
    data: dict[str, Any],
) -> Workflow:
    workflow = Workflow(
        name=data["name"],
        steps=data["steps"],
    )

    db.add(workflow)
    db.commit()
    db.refresh(workflow)

    return workflow


def get_all_workflows(db: Session) -> list[Workflow]:
    return db.query(Workflow).order_by(Workflow.id.desc()).all()


def get_workflow(db: Session, workflow_id: int) -> Workflow:
    workflow = db.get(Workflow, workflow_id)

    if workflow is None:
        raise ValueError(f"Workflow {workflow_id} not found.")

    return workflow


def get_execution(
    db: Session,
    execution_id: int,
) -> WorkflowExecution:
    execution = db.get(WorkflowExecution, execution_id)

    if execution is None:
        raise ValueError(f"Execution {execution_id} not found.")

    return execution


def transform_text(
    config: dict[str, Any],
    data: dict[str, Any],
) -> dict[str, Any]:
    operation = config.get("operation")
    input_key = config.get("input_key")
    output_key = config.get("output_key")

    if not isinstance(input_key, str) or not isinstance(output_key, str):
        raise ValueError("input_key and output_key must be strings.")

    if input_key not in data:
        raise ValueError(f"Missing input key: {input_key}")

    value = str(data[input_key])

    operations = {
        "uppercase": str.upper,
        "lowercase": str.lower,
        "strip": str.strip,
    }

    if operation not in operations:
        raise ValueError(f"Unsupported text operation: {operation}")

    return {output_key: operations[operation](value)}


def calculate(
    config: dict[str, Any],
    data: dict[str, Any],
) -> dict[str, Any]:
    operation = config.get("operation")
    left_key = config.get("left_key")
    right_key = config.get("right_key")
    output_key = config.get("output_key")

    if not all(
        isinstance(key, str)
        for key in (left_key, right_key, output_key)
    ):
        raise ValueError(
            "left_key, right_key, and output_key must be strings."
        )

    if left_key not in data:
        raise ValueError(f"Missing input key: {left_key}")

    if right_key not in data:
        raise ValueError(f"Missing input key: {right_key}")

    left = data[left_key]
    right = data[right_key]

    if (
        isinstance(left, bool)
        or isinstance(right, bool)
        or not isinstance(left, (int, float))
        or not isinstance(right, (int, float))
    ):
        raise ValueError("Calculation inputs must be numbers.")

    if operation == "add":
        result = left + right
    elif operation == "subtract":
        result = left - right
    elif operation == "multiply":
        result = left * right
    elif operation == "divide":
        if right == 0:
            raise ValueError("Cannot divide by zero.")
        result = left / right
    else:
        raise ValueError(f"Unsupported calculation: {operation}")

    return {output_key: result}


def classify_text_step(
    config: dict[str, Any],
    data: dict[str, Any],
) -> dict[str, Any]:
    input_key = config.get("input_key")
    output_key = config.get("output_key")
    categories = config.get("categories")

    if not isinstance(input_key, str) or not isinstance(output_key, str):
        raise ValueError("input_key and output_key must be strings.")

    if input_key not in data:
        raise ValueError(f"Missing input key: {input_key}")

    if (
        not isinstance(categories, list)
        or not categories
        or not all(
            isinstance(category, str) and category.strip()
            for category in categories
        )
    ):
        raise ValueError("categories must be a non-empty list of strings.")

    result = classify_text(str(data[input_key]), categories)

    return {output_key: result}


def execute_step(
    step: dict[str, Any],
    data: dict[str, Any],
) -> dict[str, Any]:
    step_type = step["type"]
    config = step["config"]

    if step_type == "TRANSFORM_TEXT":
        return transform_text(config, data)

    if step_type == "CALCULATE":
        return calculate(config, data)

    if step_type == "CLASSIFY_TEXT":
        return classify_text_step(config, data)

    raise ValueError(f"Unsupported step type: {step_type}")


def run_workflow(
    workflow: Workflow,
    input_data: dict[str, Any],
) -> dict[str, Any]:
    data = input_data.copy()
    step_results = []

    for step in workflow.steps:
        step_name = step["name"]

        try:
            output = execute_step(step, data)
            data.update(output)

            step_results.append(
                {
                    "name": step_name,
                    "status": "COMPLETED",
                    "output": output,
                    "error": None,
                }
            )

        except Exception as exc:
            step_results.append(
                {
                    "name": step_name,
                    "status": "FAILED",
                    "output": {},
                    "error": str(exc),
                }
            )

            return {
                "status": "FAILED",
                "output": data,
                "step_results": step_results,
                "error": str(exc),
            }

    return {
        "status": "COMPLETED",
        "output": data,
        "step_results": step_results,
        "error": None,
    }


def execute_workflow(
    db: Session,
    workflow_id: int,
    input_data: dict[str, Any],
) -> WorkflowExecution:
    workflow = get_workflow(db, workflow_id)
    result = run_workflow(workflow, input_data)

    execution = WorkflowExecution(
        workflow_id=workflow.id,
        status=result["status"],
        input_data=input_data,
        output_data=result["output"],
        step_results=result["step_results"],
        retry_count=0,
        error=result["error"],
    )

    db.add(execution)
    db.commit()
    db.refresh(execution)

    return execution


def retry_execution(
    db: Session,
    execution_id: int,
) -> WorkflowExecution:
    previous = get_execution(db, execution_id)

    if previous.status != "FAILED":
        raise ValueError("Only failed executions can be retried.")

    workflow = get_workflow(db, previous.workflow_id)
    result = run_workflow(workflow, previous.input_data)

    execution = WorkflowExecution(
        workflow_id=workflow.id,
        status=result["status"],
        input_data=previous.input_data,
        output_data=result["output"],
        step_results=result["step_results"],
        retry_count=previous.retry_count + 1,
        error=result["error"],
    )

    db.add(execution)
    db.commit()
    db.refresh(execution)

    return execution