import datetime
import hashlib
import json
from typing import Any

from app.models.database.chat_session import ChatSession
from app.models.database.database import SessionLocal
from app.models.database.memory import MemoryEmbedding, MemoryFolder, MemoryRecord
from app.services.embedding_service import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL_ID,
    embed_text,
    pack_vector,
)
from app.services.memory_extraction import extract_durable_facts, summarize_entries
from app.services.memory_scheduler import memory_idle_seconds, schedule_chat_memory


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _completed_entries(chat: ChatSession) -> list[dict[str, Any]]:
    history = json.loads(chat.chat_history or "[]")
    return [
        entry
        for entry in history
        if isinstance(entry, dict) and entry.get("status", "completed") == "completed"
    ]


def _active_summary(session, *, record_type: str, chat_id=None, folder_id=None, user_id=None):
    query = session.query(MemoryRecord).filter(
        MemoryRecord.record_type == record_type,
        MemoryRecord.active.is_(True),
    )
    if chat_id is not None:
        query = query.filter(MemoryRecord.chat_session_id == chat_id)
    if folder_id is not None:
        query = query.filter(MemoryRecord.folder_id == folder_id)
    if user_id is not None:
        query = query.filter(MemoryRecord.user_id == user_id)
    return query.order_by(MemoryRecord.memory_id.desc()).first()


def _add_record(session, **values) -> MemoryRecord:
    content = str(values["content"]).strip()
    record = MemoryRecord(content_hash=_content_hash(content), **values)
    session.add(record)
    session.flush()
    vector = embed_text(content)
    session.add(
        MemoryEmbedding(
            memory_id=record.memory_id,
            model_id=EMBEDDING_MODEL_ID,
            dimensions=EMBEDDING_DIMENSIONS,
            vector=pack_vector(vector),
            content_hash=record.content_hash,
        )
    )
    return record


def _replace_summary(session, *, prior: MemoryRecord | None, **values) -> MemoryRecord:
    if prior is not None:
        prior.active = False
    return _add_record(session, **values)


def rebuild_profile_summary(session, user_id: int) -> MemoryRecord | None:
    prior = _active_summary(session, record_type="user_profile_summary", user_id=user_id)
    facts = (
        session.query(MemoryRecord)
        .filter(
            MemoryRecord.user_id == user_id,
            MemoryRecord.scope == "user",
            MemoryRecord.record_type == "user_fact",
            MemoryRecord.active.is_(True),
        )
        .order_by(MemoryRecord.importance.desc(), MemoryRecord.memory_id.desc())
        .limit(40)
        .all()
    )
    if prior is not None:
        prior.active = False
    if not facts:
        return None
    unique_contents = list(dict.fromkeys(fact.content for fact in facts))
    content = "Durable user profile:\n" + "\n".join(
        f"- {fact_content}" for fact_content in unique_contents
    )
    return _add_record(
        session,
        user_id=user_id,
        scope="user",
        record_type="user_profile_summary",
        content=content[:3000],
        importance=1.0,
        confidence=1.0,
        active=True,
    )


def rebuild_folder_summary(session, user_id: int, folder_id: int) -> MemoryRecord | None:
    prior = _active_summary(
        session,
        record_type="folder_summary",
        folder_id=folder_id,
        user_id=user_id,
    )
    summaries = (
        session.query(MemoryRecord)
        .join(ChatSession, ChatSession.chat_session_id == MemoryRecord.chat_session_id)
        .filter(
            MemoryRecord.user_id == user_id,
            MemoryRecord.record_type == "thread_summary",
            MemoryRecord.active.is_(True),
            ChatSession.folder_id == folder_id,
            ChatSession.memory_enabled.is_(True),
        )
        .order_by(MemoryRecord.memory_id.desc())
        .limit(20)
        .all()
    )
    if prior is not None:
        prior.active = False
    if not summaries:
        return None
    content = "Folder memory:\n" + "\n\n".join(summary.content for summary in summaries)
    return _add_record(
        session,
        user_id=user_id,
        folder_id=folder_id,
        scope="folder",
        record_type="folder_summary",
        content=content[-4000:],
        importance=0.9,
        confidence=1.0,
        active=True,
    )


def deactivate_chat_memories(session, user_id: int, chat_session_id: int) -> set[int]:
    records = (
        session.query(MemoryRecord)
        .filter(
            MemoryRecord.user_id == user_id,
            MemoryRecord.active.is_(True),
            (
                (MemoryRecord.chat_session_id == chat_session_id)
                | (MemoryRecord.source_chat_session_id == chat_session_id)
            ),
        )
        .all()
    )
    folder_ids = {int(record.folder_id) for record in records if record.folder_id is not None}
    for record in records:
        record.active = False
    session.flush()
    return folder_ids


def process_chat_memory(
    *,
    user_id: int,
    chat_session_id: int,
    expected_revision: int,
) -> dict[str, Any]:
    with SessionLocal() as session:
        chat = (
            session.query(ChatSession)
            .filter(
                ChatSession.chat_session_id == chat_session_id,
                ChatSession.user_id == user_id,
            )
            .first()
        )
        if chat is None:
            return {"status": "missing", "chat_session_id": chat_session_id}
        if not chat.memory_enabled:
            return {"status": "disabled", "chat_session_id": chat_session_id}
        if int(chat.memory_revision or 0) != expected_revision:
            return {"status": "stale", "chat_session_id": chat_session_id}
        idle_seconds = memory_idle_seconds()
        if chat.memory_last_activity_at is not None:
            ready_at = chat.memory_last_activity_at + datetime.timedelta(seconds=idle_seconds)
            remaining = (ready_at - datetime.datetime.utcnow()).total_seconds()
            if remaining > 0:
                schedule_chat_memory(
                    user_id,
                    chat_session_id,
                    expected_revision,
                    delay_seconds=max(1, int(remaining + 0.999)),
                )
                return {"status": "waiting", "chat_session_id": chat_session_id}

        completed = _completed_entries(chat)
        processed_revision = int(chat.memory_processed_revision or 0)
        delta = completed[processed_revision:expected_revision]
        if not delta:
            chat.memory_processed_revision = expected_revision
            session.commit()
            return {"status": "noop", "chat_session_id": chat_session_id}

        snapshot_prior = _active_summary(
            session,
            record_type="thread_summary",
            chat_id=chat_session_id,
        )
        previous_text = snapshot_prior.content if snapshot_prior is not None else ""
        scope_mode = "private" if chat.folder_id is not None else str(chat.memory_mode)
        folder_id = int(chat.folder_id) if chat.folder_id is not None else None
        snapshot_mode = str(chat.memory_mode)
        snapshot_folder_id = chat.folder_id

    summary = summarize_entries(previous_text, delta)
    facts = extract_durable_facts(delta)

    with SessionLocal() as session:
        current_query = session.query(ChatSession).filter(
            ChatSession.chat_session_id == chat_session_id,
            ChatSession.user_id == user_id,
        )
        if session.get_bind().dialect.name == "postgresql":
            current_query = current_query.with_for_update()
        current = current_query.first()
        if (
            current is None
            or int(current.memory_revision or 0) != expected_revision
            or not current.memory_enabled
            or current.folder_id != snapshot_folder_id
            or current.memory_mode != snapshot_mode
        ):
            return {"status": "stale", "chat_session_id": chat_session_id}
        if int(current.memory_processed_revision or 0) >= expected_revision:
            return {"status": "noop", "chat_session_id": chat_session_id}

        prior_thread = _active_summary(
            session,
            record_type="thread_summary",
            chat_id=chat_session_id,
        )
        thread_record = _replace_summary(
            session,
            prior=prior_thread,
            user_id=user_id,
            chat_session_id=chat_session_id,
            source_chat_session_id=chat_session_id,
            scope="thread",
            record_type="thread_summary",
            content=summary,
            source_from_revision=processed_revision + 1,
            source_through_revision=expected_revision,
            importance=0.7,
            confidence=1.0,
            active=True,
        )

        accepted = 0
        fact_scope = "user" if scope_mode == "public" else ("folder" if folder_id else None)
        fact_type = "user_fact" if fact_scope == "user" else "folder_fact"
        if fact_scope is not None:
            for fact in facts:
                fact_hash = _content_hash(fact)
                duplicate = (
                    session.query(MemoryRecord)
                    .filter(
                        MemoryRecord.user_id == user_id,
                        MemoryRecord.scope == fact_scope,
                        MemoryRecord.folder_id == folder_id,
                        MemoryRecord.content_hash == fact_hash,
                        MemoryRecord.source_chat_session_id == chat_session_id,
                        MemoryRecord.active.is_(True),
                    )
                    .first()
                )
                if duplicate is not None:
                    continue
                _add_record(
                    session,
                    user_id=user_id,
                    folder_id=folder_id,
                    chat_session_id=chat_session_id if fact_scope == "folder" else None,
                    source_chat_session_id=chat_session_id,
                    scope=fact_scope,
                    record_type=fact_type,
                    content=fact,
                    source_from_revision=processed_revision + 1,
                    source_through_revision=expected_revision,
                    importance=0.9,
                    confidence=0.95,
                    active=True,
                )
                accepted += 1

        if fact_scope == "user":
            rebuild_profile_summary(session, user_id)
        elif folder_id is not None:
            rebuild_folder_summary(session, user_id, folder_id)
            folder = (
                session.query(MemoryFolder)
                .filter(
                    MemoryFolder.folder_id == folder_id,
                    MemoryFolder.user_id == user_id,
                )
                .first()
            )
            if folder is not None:
                folder.revision = int(folder.revision or 0) + 1

        current.memory_processed_revision = expected_revision
        session.commit()
        return {
            "status": "processed",
            "chat_session_id": chat_session_id,
            "revision": expected_revision,
            "thread_memory_id": thread_record.memory_id,
            "accepted_facts": accepted,
            "scope": "folder" if folder_id is not None else scope_mode,
        }
