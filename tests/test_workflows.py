import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.main import app
from app.llm import classify_text, generate_workflow


@pytest.fixture
def client():
    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    TestingSessionLocal = sessionmaker(
        bind=test_engine,
        autoflush=False,
        autocommit=False,
    )

    Base.metadata.create_all(bind=test_engine)

    def override_get_db():
        db = TestingSessionLocal()

        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=test_engine)
    test_engine.dispose()


def test_home(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "AI Workflow API is running"


def test_create_workflow(client):
    response = client.post(
        "/workflows",
        json={
            "name": "Text Workflow",
            "steps": [
                {
                    "name": "Uppercase Name",
                    "type": "TRANSFORM_TEXT",
                    "config": {
                        "operation": "uppercase",
                        "input_key": "name",
                        "output_key": "normalized_name",
                    },
                }
            ],
        },
    )

    assert response.status_code == 201
    assert response.json()["name"] == "Text Workflow"
    assert response.json()["id"] == 1


def test_list_workflows(client):
    client.post(
        "/workflows",
        json={
            "name": "Text Workflow",
            "steps": [
                {
                    "name": "Uppercase",
                    "type": "TRANSFORM_TEXT",
                    "config": {
                        "operation": "uppercase",
                        "input_key": "name",
                        "output_key": "result",
                    },
                }
            ],
        },
    )

    response = client.get("/workflows")

    assert response.status_code == 200
    assert len(response.json()) == 1


def test_execute_workflow(client):
    created = client.post(
        "/workflows",
        json={
            "name": "Order Processing",
            "steps": [
                {
                    "name": "Normalize Name",
                    "type": "TRANSFORM_TEXT",
                    "config": {
                        "operation": "uppercase",
                        "input_key": "customer_name",
                        "output_key": "normalized_name",
                    },
                },
                {
                    "name": "Calculate Total",
                    "type": "CALCULATE",
                    "config": {
                        "operation": "multiply",
                        "left_key": "unit_price",
                        "right_key": "quantity",
                        "output_key": "total_price",
                    },
                },
                {
                    "name": "Apply Discount",
                    "type": "CALCULATE",
                    "config": {
                        "operation": "subtract",
                        "left_key": "total_price",
                        "right_key": "discount",
                        "output_key": "final_price",
                    },
                },
            ],
        },
    )

    workflow_id = created.json()["id"]

    response = client.post(
        f"/workflows/{workflow_id}/execution",
        json={
            "input": {
                "customer_name": "Anirban",
                "unit_price": 100,
                "quantity": 10,
                "discount": 80,
            }
        },
    )

    assert response.status_code == 201

    result = response.json()

    assert result["status"] == "COMPLETED"
    assert result["output_data"]["normalized_name"] == "ANIRBAN"
    assert result["output_data"]["total_price"] == 1000
    assert result["output_data"]["final_price"] == 920
    assert len(result["step_results"]) == 3


def test_failed_execution_and_retry(client):
    created = client.post(
        "/workflows",
        json={
            "name": "Division Workflow",
            "steps": [
                {
                    "name": "Divide Numbers",
                    "type": "CALCULATE",
                    "config": {
                        "operation": "divide",
                        "left_key": "a",
                        "right_key": "b",
                        "output_key": "result",
                    },
                }
            ],
        },
    )

    workflow_id = created.json()["id"]

    failed = client.post(
        f"/workflows/{workflow_id}/execution",
        json={"input": {"a": 10, "b": 0}},
    )

    assert failed.status_code == 201
    assert failed.json()["status"] == "FAILED"

    execution_id = failed.json()["id"]

    retry = client.post(f"/executions/{execution_id}/retry")

    assert retry.status_code == 201
    assert retry.json()["status"] == "FAILED"
    assert retry.json()["retry_count"] == 1

    history = client.get(f"/executions/{execution_id}")

    assert history.status_code == 200
    assert history.json()["status"] == "FAILED"


def test_missing_workflow(client):
    response = client.get("/workflows/999")

    assert response.status_code == 404


def test_invalid_workflow_request(client):
    response = client.post(
        "/workflows",
        json={"name": "Invalid Workflow", "steps": []},
    )

    assert response.status_code == 422


def test_generated_workflow_uses_llm(client, monkeypatch):
    def fake_generate_workflow(description):
        return {
            "name": "Generated Workflow",
            "steps": [
                {
                    "name": "Normalize",
                    "type": "TRANSFORM_TEXT",
                    "config": {
                        "operation": "uppercase",
                        "input_key": "name",
                        "output_key": "result",
                    },
                }
            ],
        }

    monkeypatch.setattr(
        "app.main.generate_workflow",
        fake_generate_workflow,
    )

    response = client.post(
        "/workflows/generate",
        json={
            "description": "Create a workflow that converts a name to uppercase."
        },
    )

    assert response.status_code == 201
    assert response.json()["name"] == "Generated Workflow"


def test_classification_function(monkeypatch):
    def fake_chat(**kwargs):
        return {"message": {"content": "positive"}}

    monkeypatch.setattr(
        "app.llm.client.chat",
        fake_chat,
    )

    assert classify_text(
        "Excellent product",
        ["positive", "negative", "neutral"],
    ) == "positive"


def test_generate_workflow_function(monkeypatch):
    def fake_chat(**kwargs):
        return {
            "message": {
                "content": (
                    '{"name":"Generated",'
                    '"steps":[{"name":"Uppercase",'
                    '"type":"TRANSFORM_TEXT",'
                    '"config":{"operation":"uppercase",'
                    '"input_key":"name","output_key":"result"}}]}'
                )
            }
        }

    monkeypatch.setattr(
        "app.llm.client.chat",
        fake_chat,
    )

    result = generate_workflow("Convert a name to uppercase.")

    assert result["name"] == "Generated"
    assert result["steps"][0]["type"] == "TRANSFORM_TEXT"