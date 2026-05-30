import datetime
import importlib

import pytest

from app.models.database.database import (
    Base,
    DATABASE_BACKEND,
    SQLALCHEMY_DATABASE_URL,
    Session,
    SessionLocal,
    configure_database,
)


@pytest.fixture()
def sqlite_database(tmp_path):
    original_url = SQLALCHEMY_DATABASE_URL
    original_backend = DATABASE_BACKEND
    sqlite_url = f"sqlite:///{tmp_path / 'geist.sqlite3'}"

    engine = configure_database(database_url=sqlite_url, backend="sqlite")

    importlib.import_module("app.models.database")

    Base.metadata.create_all(bind=engine)
    try:
        yield
    finally:
        Session.remove()
        Base.metadata.drop_all(bind=engine)
        configure_database(database_url=original_url, backend=original_backend)


def create_test_user() -> int:
    from app.models.database.geist_user import GeistUser

    with SessionLocal() as session:
        user = GeistUser(
            username="sqlite-user",
            name="SQLite User",
            email="sqlite@example.com",
            password="",
        )
        session.add(user)
        session.commit()
        session.refresh(user)
        return user.user_id


def test_sqlite_persists_core_database_models(sqlite_database):
    from app.models.database.agent import Agent
    from app.models.database.agent_preset import AgentPreset, Restriction
    from app.models.database.chat_session import get_chat_history, update_chat_history
    from app.models.database.file_upload import (
        FileUploadModel,
        create_file_upload,
        delete_file_upload,
        get_file_upload_by_id,
        get_files_by_user,
    )
    from app.models.database.user_settings import (
        create_default_user_settings,
        get_or_create_user_settings,
        update_user_settings,
    )
    from app.models.database.workflow import (
        Workflow,
        WorkflowRunStatus,
        WorkflowStep,
        create_workflow,
        create_workflow_run,
        create_workflow_step_result,
        get_workflow_by_id,
        update_workflow_run_status,
        update_workflow_step_result,
    )

    user_id = create_test_user()

    with SessionLocal() as session:
        agent = Agent(process_id="sqlite-process", world_context="world")
        session.add(agent)
        session.commit()
        session.refresh(agent)
        agent_id = agent.agent_id

    assert Agent.get_agent_by_id(agent_id).process_id == "sqlite-process"

    AgentPreset.upsert_agent_preset(
        name="SQLite Preset",
        version="1.0",
        description="SQLite-compatible preset",
        max_tokens=128,
        n=1,
        temperature=1,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        tags="sqlite",
        working_context_length=128,
        long_term_context_length=128,
        agent_type="local",
        prompt="test",
        interactive_only=False,
        process_world=False,
    )

    with SessionLocal() as session:
        preset = session.query(AgentPreset).filter_by(name="SQLite Preset").one()
        restriction = Restriction(
            agent_preset_id=preset.agent_preset_id,
            name="SQLite Restriction",
            rate=10,
            period_hours=1,
            spending_limit=100,
            restriction_type="allowlist",
            allowed_plugins=["files"],
            allowed_methods=["read"],
            create_date=datetime.datetime.utcnow(),
            update_date=datetime.datetime.utcnow(),
        )
        session.add(restriction)
        session.commit()
        restriction_id = restriction.restriction_id

    with SessionLocal() as session:
        saved_restriction = session.query(Restriction).filter_by(restriction_id=restriction_id).one()
        assert saved_restriction.allowed_plugins == ["files"]
        assert saved_restriction.allowed_methods == ["read"]

    chat_session = update_chat_history("hello", "hi", session_id=77)
    assert chat_session.chat_session_id == 77
    assert get_chat_history(77)[0] == {"user": "hello", "ai": "hi"}

    created_file = create_file_upload(
        FileUploadModel(
            filename="stored.txt",
            original_filename="original.txt",
            file_data=b"sqlite-data",
            file_size=11,
            mime_type="text/plain",
            file_hash="sqlite-hash",
            user_id=user_id,
        )
    )
    assert get_file_upload_by_id(created_file.file_id).file_data == b"sqlite-data"
    assert get_files_by_user(user_id)[0].original_filename == "original.txt"

    settings = create_default_user_settings(user_id)
    assert settings.default_agent_type == "local"
    updated_settings = update_user_settings(user_id, {"default_agent_type": "online"})
    assert updated_settings.default_agent_type == "online"
    assert get_or_create_user_settings(user_id).default_agent_type == "online"

    workflow = create_workflow(
        Workflow(
            name="SQLite Workflow",
            user_id=user_id,
            steps=[
                WorkflowStep(
                    step_name="SQLite Step",
                    step_description="Runs on SQLite",
                    step_status="pending",
                    display_x=1,
                    display_y=2,
                    command_str="echo sqlite",
                    step_type="custom",
                )
            ],
        )
    )
    saved_workflow = get_workflow_by_id(workflow.workflow_id)
    assert saved_workflow.name == "SQLite Workflow"
    assert saved_workflow.steps[0].step_name == "SQLite Step"

    run = create_workflow_run(workflow.workflow_id, user_id, input_data={"source": "sqlite"})
    finished_run = update_workflow_run_status(
        run.run_id,
        WorkflowRunStatus.COMPLETED,
        output_data={"ok": True},
    )
    assert finished_run.output_data == {"ok": True}

    result = create_workflow_step_result(run.run_id, saved_workflow.steps[0].step_id)
    finished_result = update_workflow_step_result(
        result.result_id,
        WorkflowRunStatus.COMPLETED,
        output_data={"step": "done"},
    )
    assert finished_result.output_data == {"step": "done"}

    assert delete_file_upload(created_file.file_id, user_id) is True
    assert get_file_upload_by_id(created_file.file_id) is None
