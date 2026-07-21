import datetime
import importlib
import json

import pytest

from app.models.database.chat_session import ChatSession, update_chat_history
from app.models.database.database import (
    DATABASE_CONFIG,
    Base,
    Session,
    SessionLocal,
    configure_database,
)
from app.models.database.database_config import DatabaseConfig
from app.models.database.geist_user import GeistUser
from app.models.database.job import Job, JobStatus
from app.models.database.memory import MemoryRecord
from app.services.memory_context import build_memory_context
from app.services.memory_processor import process_chat_memory
from app.services.memory_service import (
    create_folder,
    search_memories,
    update_chat_memory_settings,
)


@pytest.fixture()
def memory_database(tmp_path, monkeypatch):
    original_config = DATABASE_CONFIG
    sqlite_config = DatabaseConfig(
        provider="sqlite",
        database_url=f"sqlite:///{tmp_path / 'memory.sqlite3'}",
    )
    engine = configure_database(sqlite_config)
    importlib.import_module("app.models.database")
    Base.metadata.create_all(bind=engine)
    monkeypatch.setenv("GEIST_MEMORY_IDLE_SECONDS", "0")
    with SessionLocal() as session:
        session.add_all(
            [
                GeistUser(
                    user_id=1,
                    username="one",
                    name="One",
                    email="one@example.com",
                    password="",
                ),
                GeistUser(
                    user_id=2,
                    username="two",
                    name="Two",
                    email="two@example.com",
                    password="",
                ),
            ]
        )
        session.commit()
    try:
        yield
    finally:
        Session.remove()
        Base.metadata.drop_all(bind=engine)
        configure_database(original_config)


def _complete_chat(
    *,
    user_id: int,
    chat_id: int,
    user_message: str,
    memory_mode: str = "public",
    folder_id: int | None = None,
    memory_enabled: bool = True,
) -> ChatSession:
    return update_chat_history(
        user_message,
        "Acknowledged.",
        session_id=chat_id,
        user_id=user_id,
        run_id=f"run-{chat_id}",
        status="completed",
        memory_enabled=memory_enabled,
        memory_mode=memory_mode,
        folder_id=folder_id,
    )


def test_public_chat_is_debounced_vectorized_and_searchable(memory_database):
    chat = _complete_chat(user_id=1, chat_id=11, user_message="Remember cobalt.")

    with SessionLocal() as session:
        jobs = session.query(Job).all()
        assert len(jobs) == 1
        assert jobs[0].status == JobStatus.QUEUED.value
        assert jobs[0].dedupe_key == "chat-memory:1:11"
        assert json.loads(jobs[0].payload) == {
            "user_id": 1,
            "chat_session_id": 11,
            "expected_revision": 1,
            "pipeline_version": 1,
        }

    result = process_chat_memory(
        user_id=1,
        chat_session_id=11,
        expected_revision=chat.memory_revision,
    )

    assert result["status"] == "processed"
    assert result["accepted_facts"] == 1
    duplicate = process_chat_memory(
        user_id=1,
        chat_session_id=11,
        expected_revision=chat.memory_revision,
    )
    assert duplicate["status"] == "noop"
    matches = search_memories(1, "What color should Geist remember?", scope="user")
    assert any("cobalt" in match["content"].lower() for match in matches)


def test_secret_like_turn_is_redacted_before_summary_and_embedding(memory_database):
    chat = _complete_chat(
        user_id=1,
        chat_id=13,
        user_message="Remember my API key is do-not-store-this.",
    )

    process_chat_memory(user_id=1, chat_session_id=13, expected_revision=chat.memory_revision)

    with SessionLocal() as session:
        summary = (
            session.query(MemoryRecord)
            .filter(
                MemoryRecord.chat_session_id == 13,
                MemoryRecord.record_type == "thread_summary",
                MemoryRecord.active.is_(True),
            )
            .one()
        )
        assert "do-not-store-this" not in summary.content
        assert "sensitive content omitted" in summary.content
    assert search_memories(1, "do-not-store-this", scope="user") == []


def test_new_turn_coalesces_the_pending_digest(memory_database):
    _complete_chat(user_id=1, chat_id=12, user_message="First completed turn.")
    update_chat_history(
        "Remember indigo.",
        "Acknowledged.",
        session_id=12,
        user_id=1,
        run_id="run-12-2",
        status="completed",
    )

    with SessionLocal() as session:
        jobs = session.query(Job).filter(Job.dedupe_key == "chat-memory:1:12").all()
        assert len(jobs) == 1
        assert json.loads(jobs[0].payload)["expected_revision"] == 2

    stale = process_chat_memory(user_id=1, chat_session_id=12, expected_revision=1)
    assert stale["status"] == "stale"


def test_failed_or_cancelled_turns_do_not_schedule_memory(memory_database):
    update_chat_history(
        "Failed prompt",
        None,
        session_id=14,
        user_id=1,
        run_id="failed-run",
        status="failed",
    )
    update_chat_history(
        "Cancelled prompt",
        "Partial",
        session_id=15,
        user_id=1,
        run_id="cancelled-run",
        status="cancelled",
    )

    with SessionLocal() as session:
        assert session.query(Job).count() == 0
        assert session.query(ChatSession).filter_by(chat_session_id=14).one().memory_revision == 0
        assert session.query(ChatSession).filter_by(chat_session_id=15).one().memory_revision == 0


def test_private_chat_never_enters_global_memory(memory_database):
    chat = _complete_chat(
        user_id=1,
        chat_id=21,
        user_message="Remember obsidian.",
        memory_mode="private",
    )
    process_chat_memory(user_id=1, chat_session_id=21, expected_revision=chat.memory_revision)

    assert search_memories(1, "obsidian", scope="user") == []
    private_matches = search_memories(
        1,
        "obsidian",
        scope="thread",
        chat_session_id=21,
    )
    assert any("obsidian" in match["content"].lower() for match in private_matches)


def test_private_context_never_reads_global_profile(memory_database):
    public_chat = _complete_chat(
        user_id=1,
        chat_id=22,
        user_message="Remember global-cobalt.",
    )
    process_chat_memory(
        user_id=1,
        chat_session_id=22,
        expected_revision=public_chat.memory_revision,
    )
    private_chat = _complete_chat(
        user_id=1,
        chat_id=23,
        user_message="A private conversation.",
        memory_mode="private",
    )
    process_chat_memory(
        user_id=1,
        chat_session_id=23,
        expected_revision=private_chat.memory_revision,
    )

    context = build_memory_context(
        1,
        "global-cobalt",
        chat_session_id=23,
        memory_enabled=True,
        memory_mode="private",
    )

    assert "global-cobalt" not in context


def test_public_context_combines_global_and_thread_memory(memory_database):
    global_chat = _complete_chat(
        user_id=1,
        chat_id=24,
        user_message="Remember global-cerulean.",
    )
    process_chat_memory(
        user_id=1,
        chat_session_id=24,
        expected_revision=global_chat.memory_revision,
    )
    current_chat = _complete_chat(
        user_id=1,
        chat_id=25,
        user_message="The thread topic is local-saffron.",
    )
    process_chat_memory(
        user_id=1,
        chat_session_id=25,
        expected_revision=current_chat.memory_revision,
    )

    context = build_memory_context(
        1,
        "global-cerulean local-saffron",
        chat_session_id=25,
        memory_enabled=True,
        memory_mode="public",
    )

    assert "global-cerulean" in context
    assert "local-saffron" in context


def test_folder_memory_is_isolated_from_global_other_folders_and_users(memory_database):
    folder = create_folder(1, "Project Zephyr")
    other_folder = create_folder(1, "Project Quartz")
    private_chat = _complete_chat(
        user_id=1,
        chat_id=31,
        user_message="Remember zephyr launch is Tuesday.",
        memory_mode="private",
        folder_id=folder["folder_id"],
    )
    process_chat_memory(
        user_id=1,
        chat_session_id=31,
        expected_revision=private_chat.memory_revision,
    )

    folder_matches = search_memories(
        1,
        "When is the zephyr launch?",
        scope="folder",
        folder_id=folder["folder_id"],
    )
    assert any("Tuesday" in match["content"] for match in folder_matches)
    assert search_memories(1, "zephyr", scope="user") == []
    assert (
        search_memories(
            1,
            "zephyr",
            scope="folder",
            folder_id=other_folder["folder_id"],
        )
        == []
    )
    assert search_memories(2, "zephyr", scope="user") == []


def test_disabling_memory_deactivates_all_chat_derived_records(memory_database):
    chat = _complete_chat(user_id=1, chat_id=41, user_message="Remember marigold.")
    process_chat_memory(user_id=1, chat_session_id=41, expected_revision=chat.memory_revision)
    assert search_memories(1, "marigold", scope="user")

    settings = update_chat_memory_settings(1, 41, memory_enabled=False)

    assert settings["effective_scope"] == "disabled"
    assert search_memories(1, "marigold", scope="user") == []
    assert (
        search_memories(1, "marigold", scope="thread", chat_session_id=41)
        == []
    )


def test_reenabling_memory_does_not_backfill_disabled_turns(memory_database):
    chat = _complete_chat(
        user_id=1,
        chat_id=42,
        user_message="Remember disabled-lilac.",
        memory_enabled=False,
    )
    assert chat.memory_processed_revision == chat.memory_revision

    settings = update_chat_memory_settings(1, 42, memory_enabled=True)

    assert settings["status"] == "ready"
    assert search_memories(1, "disabled-lilac", scope="user") == []


def test_moving_public_chat_into_folder_withdraws_global_fact(memory_database):
    chat = _complete_chat(user_id=1, chat_id=51, user_message="Remember periwinkle.")
    process_chat_memory(user_id=1, chat_session_id=51, expected_revision=chat.memory_revision)
    assert search_memories(1, "periwinkle", scope="user")
    folder = create_folder(1, "Private notes")

    settings = update_chat_memory_settings(1, 51, folder_id=folder["folder_id"])

    assert settings["memory_mode"] == "private"
    assert settings["effective_scope"] == "folder"
    assert search_memories(1, "periwinkle", scope="user") == []


def test_privatizing_one_source_keeps_fact_supported_by_another_public_chat(
    memory_database,
):
    first = _complete_chat(user_id=1, chat_id=53, user_message="Remember shared-ochre.")
    second = _complete_chat(user_id=1, chat_id=54, user_message="Remember shared-ochre.")
    process_chat_memory(user_id=1, chat_session_id=53, expected_revision=first.memory_revision)
    process_chat_memory(user_id=1, chat_session_id=54, expected_revision=second.memory_revision)
    folder = create_folder(1, "Source isolation")

    update_chat_memory_settings(1, 53, folder_id=folder["folder_id"])

    matches = search_memories(1, "shared-ochre", scope="user")
    assert any("shared-ochre" in match["content"] for match in matches)


def test_switching_private_chat_public_does_not_promote_private_history(memory_database):
    chat = _complete_chat(
        user_id=1,
        chat_id=52,
        user_message="Remember private-amethyst.",
        memory_mode="private",
    )
    process_chat_memory(user_id=1, chat_session_id=52, expected_revision=chat.memory_revision)

    settings = update_chat_memory_settings(1, 52, memory_mode="public")

    assert settings["effective_scope"] == "public"
    assert search_memories(1, "private-amethyst", scope="user") == []


def test_processor_waits_until_idle_window(memory_database, monkeypatch):
    monkeypatch.setenv("GEIST_MEMORY_IDLE_SECONDS", "20")
    chat = _complete_chat(user_id=1, chat_id=61, user_message="Remember sienna.")
    with SessionLocal() as session:
        row = session.query(ChatSession).filter_by(chat_session_id=61).first()
        row.memory_last_activity_at = datetime.datetime.utcnow()
        session.commit()

    result = process_chat_memory(
        user_id=1,
        chat_session_id=61,
        expected_revision=chat.memory_revision,
    )

    assert result["status"] == "waiting"
    with SessionLocal() as session:
        assert (
            session.query(MemoryRecord)
            .filter(MemoryRecord.source_chat_session_id == 61)
            .count()
            == 0
        )


def test_turn_arriving_during_summarization_discards_stale_output(
    memory_database,
    monkeypatch,
):
    chat = _complete_chat(user_id=1, chat_id=62, user_message="Remember stale-cyan.")

    def concurrent_summary(_previous, _entries):
        with SessionLocal() as concurrent_session:
            row = (
                concurrent_session.query(ChatSession)
                .filter_by(chat_session_id=62)
                .first()
            )
            row.memory_revision = 2
            concurrent_session.commit()
        return "This summary must not commit."

    monkeypatch.setattr(
        "app.services.memory_processor.summarize_entries",
        concurrent_summary,
    )

    result = process_chat_memory(
        user_id=1,
        chat_session_id=62,
        expected_revision=chat.memory_revision,
    )

    assert result["status"] == "stale"
    with SessionLocal() as session:
        assert (
            session.query(MemoryRecord)
            .filter(MemoryRecord.source_chat_session_id == 62)
            .count()
            == 0
        )
