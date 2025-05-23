import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from app.main import app
from app.models.database.database import SessionLocal
from app.models.database.workflow import Workflow, WorkflowStep
from app.models.database.geist_user import GeistUser
from app.schemas.workflow import WorkflowCreate, WorkflowStepCreate

@pytest.fixture
def client():
    """Test client fixture."""
    return TestClient(app)

@pytest.fixture
def db_session():
    """Database session fixture."""
    session = SessionLocal()
    try:
        yield session
    finally:
        # Clean up created test users and workflows
        session.query(WorkflowStep).delete()
        session.query(Workflow).delete()
        session.query(GeistUser).delete()
        session.commit()
        session.close()

@pytest.fixture
def test_user(db_session: Session):
    """Test user fixture. Creates a user in the database."""
    user_data = {
        "user_id": 1,
        "email": "test@example.com",
        "username": "testuser",
        "name": "Test User",
        "password": "testpassword" # Add other required fields as per your GeistUser model
    }
    user = GeistUser(**user_data)
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user) # Refresh to get the auto-generated ID if it's not manually set
    return {
        "user_id": user.user_id, # Use the actual user_id from the DB
        "email": user.email
    }

@pytest.fixture
def auth_headers(test_user):
    """Authentication headers fixture."""
    return {"Authorization": f"Bearer test_token_{test_user['user_id']}"}

def test_create_workflow(client, auth_headers, test_user):
    """Test creating a new workflow."""
    workflow_data = {
        "name": "Test Workflow",
        "steps": [
            {
                "step_name": "Test Step",
                "step_description": "Test Description",
                "step_status": "pending",
                "display_x": 100,
                "display_y": 100,
                "commmmand_str": "test_command",
                "step_type": "custom"
            }
        ]
    }
    
    response = client.post("/api/v1/workflows/", json=workflow_data, headers=auth_headers)
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == workflow_data["name"]
    assert len(data["steps"]) == 1
    assert data["steps"][0]["step_name"] == workflow_data["steps"][0]["step_name"]

def test_list_workflows(client, auth_headers, test_user, db_session):
    """Test listing workflows for a user."""
    # Create test workflows
    workflow1 = Workflow(name="Workflow 1", user_id=test_user["user_id"])
    workflow2 = Workflow(name="Workflow 2", user_id=test_user["user_id"])
    db_session.add_all([workflow1, workflow2])
    db_session.commit()
    
    response = client.get("/api/v1/workflows/", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert {w["name"] for w in data} == {"Workflow 1", "Workflow 2"}

def test_get_workflow(client, auth_headers, test_user, db_session):
    """Test retrieving a specific workflow."""
    # Create test workflow
    workflow = Workflow(name="Test Workflow", user_id=test_user["user_id"])
    db_session.add(workflow)
    db_session.commit()
    
    response = client.get(f"/api/v1/workflows/{workflow.workflow_id}", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Test Workflow"
    assert data["workflow_id"] == workflow.workflow_id

def test_update_workflow(client, auth_headers, test_user, db_session):
    """Test updating a workflow."""
    # Create test workflow
    workflow = Workflow(name="Original Name", user_id=test_user["user_id"])
    db_session.add(workflow)
    db_session.commit()
    
    update_data = {
        "name": "Updated Name",
        "steps": [
            {
                "step_name": "New Step",
                "step_description": "New Description",
                "step_status": "pending",
                "display_x": 200,
                "display_y": 200,
                "commmmand_str": "new_command",
                "step_type": "custom"
            }
        ]
    }
    
    response = client.put(
        f"/api/v1/workflows/{workflow.workflow_id}",
        json=update_data,
        headers=auth_headers
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Updated Name"
    assert len(data["steps"]) == 1
    assert data["steps"][0]["step_name"] == "New Step"

def test_delete_workflow(client, auth_headers, test_user, db_session):
    """Test deleting a workflow."""
    # Create test workflow
    workflow = Workflow(name="To Delete", user_id=test_user["user_id"])
    db_session.add(workflow)
    db_session.commit()
    
    response = client.delete(f"/api/v1/workflows/{workflow.workflow_id}", headers=auth_headers)
    assert response.status_code == 204
    
    # Verify workflow is deleted
    deleted_workflow = db_session.query(Workflow).filter_by(workflow_id=workflow.workflow_id).first()
    assert deleted_workflow is None

def test_get_nonexistent_workflow(client, auth_headers):
    """Test retrieving a non-existent workflow."""
    response = client.get("/api/v1/workflows/99999", headers=auth_headers)
    assert response.status_code == 404
    
def test_access_other_user_workflow(client, auth_headers, db_session):
    """Test accessing another user's workflow."""
    # Create a different user for this test
    other_user_data = {
        "user_id": 999,
        "email": "other@example.com",
        "username": "otheruser",
        "name": "Other User",
        "password": "otherpassword"
    }
    other_user = GeistUser(**other_user_data)
    db_session.add(other_user)
    db_session.commit()
    db_session.refresh(other_user)

    # Create workflow for the different user
    other_user_workflow = Workflow(name="Other User Workflow", user_id=other_user.user_id)
    db_session.add(other_user_workflow)
    db_session.commit()
    
    response = client.get(
        f"/api/v1/workflows/{other_user_workflow.workflow_id}",
        headers=auth_headers
    )
    assert response.status_code == 403
