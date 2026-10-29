from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

from app.database import Base, engine, get_db
from app.llm import generate_workflow
from app.schemas import (
    WorkflowCreate,
    WorkflowDescription,
    WorkflowExecutionRequest,
)
from app.services import (
    create_workflow,
    execute_workflow,
    get_all_workflows,
    get_execution,
    get_workflow,
    retry_execution,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="AI Workflow API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def home():
    return {"message": "AI Workflow API is running"}


@app.post("/workflows", status_code=201)
def create_workflow_endpoint(
    data: WorkflowCreate,
    db: Session = Depends(get_db),
):
    return create_workflow(db, data.model_dump())


@app.get("/workflows")
def list_workflows_endpoint(
    db: Session = Depends(get_db),
):
    return get_all_workflows(db)


@app.get("/workflows/{workflow_id}")
def get_workflow_endpoint(
    workflow_id: int,
    db: Session = Depends(get_db),
):
    try:
        return get_workflow(db, workflow_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc


@app.post("/workflows/generate", status_code=201)
def generate_workflow_endpoint(
    data: WorkflowDescription,
    db: Session = Depends(get_db),
):
    try:
        generated = generate_workflow(data.description)
        validated = WorkflowCreate.model_validate(generated)
        return create_workflow(db, validated.model_dump())
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Workflow generation failed: {exc}",
        ) from exc


@app.post("/workflows/{workflow_id}/execution", status_code=201)
def execute_workflow_endpoint(
    workflow_id: int,
    data: WorkflowExecutionRequest,
    db: Session = Depends(get_db),
):
    try:
        return execute_workflow(db, workflow_id, data.input)
    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc


@app.get("/executions/{execution_id}")
def get_execution_endpoint(
    execution_id: int,
    db: Session = Depends(get_db),
):
    try:
        return get_execution(db, execution_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc


@app.post("/executions/{execution_id}/retry", status_code=201)
def retry_execution_endpoint(
    execution_id: int,
    db: Session = Depends(get_db),
):
    try:
        return retry_execution(db, execution_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=400 if "Only failed" in str(exc) else 404,
            detail=str(exc),
        ) from exc