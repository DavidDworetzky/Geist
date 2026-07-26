import datetime
import hashlib
from typing import Any, cast

from sqlalchemy import or_

from app.models.database.chat_session import ChatSession
from app.models.database.database import SessionLocal
from app.models.database.memory import MemoryEmbedding, MemoryFolder, MemoryRecord
from app.services.embedding_service import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL_ID,
    cosine_similarity,
    embed_text,
    pack_vector,
    unpack_vector,
)
from app.services.memory_extraction import is_secret_like
from app.services.memory_processor import (
    deactivate_chat_memories,
    rebuild_folder_summary,
    rebuild_profile_summary,
)
from app.services.memory_scheduler import schedule_chat_memory


FOLDER_COLORS = {"violet", "blue", "mint", "amber", "rose", "slate"}


def _folder_dict(
    folder: MemoryFolder,
    chat_count: int = 0,
    summary: str | None = None,
) -> dict[str, Any]:
    return {
        "folder_id": folder.folder_id,
        "name": folder.name,
        "color": folder.color,
        "revision": folder.revision,
        "chat_count": chat_count,
        "summary": summary,
        "created_at": folder.created_at.isoformat() if folder.created_at else None,
        "updated_at": folder.updated_at.isoformat() if folder.updated_at else None,
    }


def _chat_settings(chat: ChatSession) -> dict[str, Any]:
    if not chat.memory_enabled:
        status = "disabled"
    elif int(chat.memory_processed_revision or 0) >= int(chat.memory_revision or 0):
        status = "ready"
    elif chat.memory_last_activity_at:
        status = "waiting_for_idle"
    else:
        status = "ready"
    return {
        "chat_session_id": chat.chat_session_id,
        "memory_enabled": bool(chat.memory_enabled),
        "memory_mode": str(chat.memory_mode or "public"),
        "folder_id": chat.folder_id,
        "effective_scope": (
            "disabled"
            if not chat.memory_enabled
            else ("folder" if chat.folder_id is not None else str(chat.memory_mode or "public"))
        ),
        "status": status,
        "memory_revision": int(chat.memory_revision or 0),
        "memory_processed_revision": int(chat.memory_processed_revision or 0),
    }


def list_folders(user_id: int) -> list[dict[str, Any]]:
    with SessionLocal() as session:
        folders = (
            session.query(MemoryFolder)
            .filter(MemoryFolder.user_id == user_id)
            .order_by(MemoryFolder.name)
            .all()
        )
        result = []
        for folder in folders:
            chat_count = (
                session.query(ChatSession)
                .filter(
                    ChatSession.user_id == user_id,
                    ChatSession.folder_id == folder.folder_id,
                )
                .count()
            )
            summary = (
                session.query(MemoryRecord)
                .filter(
                    MemoryRecord.user_id == user_id,
                    MemoryRecord.folder_id == folder.folder_id,
                    MemoryRecord.record_type == "folder_summary",
                    MemoryRecord.active.is_(True),
                )
                .order_by(MemoryRecord.memory_id.desc())
                .first()
            )
            result.append(_folder_dict(folder, chat_count, summary.content if summary else None))
        return result


def create_folder(user_id: int, name: str, color: str = "violet") -> dict[str, Any]:
    clean_name = " ".join(name.split())
    if not clean_name:
        raise ValueError("Folder name is required")
    if color not in FOLDER_COLORS:
        raise ValueError("Unknown folder color")
    with SessionLocal() as session:
        existing = (
            session.query(MemoryFolder)
            .filter(MemoryFolder.user_id == user_id, MemoryFolder.name == clean_name)
            .first()
        )
        if existing is not None:
            raise ValueError("A folder with that name already exists")
        folder = MemoryFolder(user_id=user_id, name=clean_name, color=color)
        session.add(folder)
        session.commit()
        session.refresh(folder)
        return _folder_dict(folder)


def update_folder(user_id: int, folder_id: int, **changes) -> dict[str, Any] | None:
    with SessionLocal() as session:
        folder = (
            session.query(MemoryFolder)
            .filter(MemoryFolder.folder_id == folder_id, MemoryFolder.user_id == user_id)
            .first()
        )
        if folder is None:
            return None
        if changes.get("name") is not None:
            clean_name = " ".join(str(changes["name"]).split())
            if not clean_name:
                raise ValueError("Folder name is required")
            duplicate = (
                session.query(MemoryFolder)
                .filter(
                    MemoryFolder.user_id == user_id,
                    MemoryFolder.name == clean_name,
                    MemoryFolder.folder_id != folder_id,
                )
                .first()
            )
            if duplicate is not None:
                raise ValueError("A folder with that name already exists")
            folder.name = clean_name
        if changes.get("color") is not None:
            if changes["color"] not in FOLDER_COLORS:
                raise ValueError("Unknown folder color")
            folder.color = changes["color"]
        session.commit()
        session.refresh(folder)
        return _folder_dict(folder)


def delete_folder(user_id: int, folder_id: int) -> bool:
    with SessionLocal() as session:
        folder = (
            session.query(MemoryFolder)
            .filter(MemoryFolder.folder_id == folder_id, MemoryFolder.user_id == user_id)
            .first()
        )
        if folder is None:
            return False
        chats = (
            session.query(ChatSession)
            .filter(ChatSession.user_id == user_id, ChatSession.folder_id == folder_id)
            .all()
        )
        for chat in chats:
            chat.folder_id = None
            chat.memory_mode = "private"
        session.query(MemoryRecord).filter(
            MemoryRecord.user_id == user_id,
            MemoryRecord.folder_id == folder_id,
        ).delete(synchronize_session=False)
        session.delete(folder)
        session.commit()
        return True


def get_chat_memory_settings(user_id: int, chat_session_id: int) -> dict[str, Any] | None:
    with SessionLocal() as session:
        chat = (
            session.query(ChatSession)
            .filter(
                ChatSession.chat_session_id == chat_session_id,
                ChatSession.user_id == user_id,
            )
            .first()
        )
        return _chat_settings(chat) if chat is not None else None


def update_chat_memory_settings(
    user_id: int,
    chat_session_id: int,
    *,
    memory_enabled: bool | None = None,
    memory_mode: str | None = None,
    folder_id: int | None = None,
    clear_folder: bool = False,
) -> dict[str, Any] | None:
    if memory_mode is not None and memory_mode not in {"public", "private"}:
        raise ValueError("memory_mode must be public or private")
    schedule: tuple[int, int, int] | None = None
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
            return None
        old_folder_id = int(chat.folder_id) if chat.folder_id is not None else None
        old_mode = str(chat.memory_mode or "public")
        old_enabled = bool(chat.memory_enabled)

        target_folder_id = None if clear_folder else (folder_id or chat.folder_id)
        if target_folder_id is not None:
            folder = (
                session.query(MemoryFolder)
                .filter(
                    MemoryFolder.folder_id == target_folder_id,
                    MemoryFolder.user_id == user_id,
                )
                .first()
            )
            if folder is None:
                raise ValueError("Folder not found")

        if memory_enabled is not None:
            chat.memory_enabled = memory_enabled
        if clear_folder:
            chat.folder_id = None
        elif folder_id is not None:
            chat.folder_id = folder_id
        if chat.folder_id is not None:
            chat.memory_mode = "private"
        elif memory_mode is not None:
            chat.memory_mode = memory_mode

        new_folder_id = int(chat.folder_id) if chat.folder_id is not None else None
        new_mode = str(chat.memory_mode or "public")
        scope_changed = old_folder_id != new_folder_id or old_mode != new_mode

        if not chat.memory_enabled:
            affected_folders = deactivate_chat_memories(session, user_id, chat_session_id)
            for affected_folder_id in affected_folders:
                rebuild_folder_summary(session, user_id, affected_folder_id)
            rebuild_profile_summary(session, user_id)
            chat.memory_processed_revision = int(chat.memory_revision or 0)
        elif scope_changed:
            if old_folder_id is not None:
                session.query(MemoryRecord).filter(
                    MemoryRecord.user_id == user_id,
                    MemoryRecord.folder_id == old_folder_id,
                    MemoryRecord.source_chat_session_id == chat_session_id,
                ).update({"active": False}, synchronize_session=False)
                rebuild_folder_summary(session, user_id, old_folder_id)
            session.query(MemoryRecord).filter(
                MemoryRecord.user_id == user_id,
                MemoryRecord.scope == "user",
                MemoryRecord.source_chat_session_id == chat_session_id,
            ).update({"active": False}, synchronize_session=False)
            rebuild_profile_summary(session, user_id)
            if new_folder_id is not None:
                rebuild_folder_summary(session, user_id, new_folder_id)
        if (scope_changed or old_enabled != bool(chat.memory_enabled)) and chat.memory_revision:
            chat.memory_last_activity_at = datetime.datetime.utcnow()
        session.commit()
        session.refresh(chat)
        result = _chat_settings(chat)
        if chat.memory_enabled and int(chat.memory_revision or 0) > int(
            chat.memory_processed_revision or 0
        ):
            schedule = (user_id, chat_session_id, int(chat.memory_revision or 0))
    if schedule is not None:
        schedule_chat_memory(*schedule)
    return result


def search_memories(
    user_id: int,
    query_text: str,
    *,
    scope: str,
    folder_id: int | None = None,
    chat_session_id: int | None = None,
    limit: int = 8,
) -> list[dict[str, Any]]:
    query_vector = embed_text(query_text)
    query_tokens = set(query_text.lower().split())
    with SessionLocal() as session:
        query = (
            session.query(MemoryRecord, MemoryEmbedding)
            .join(MemoryEmbedding, MemoryEmbedding.memory_id == MemoryRecord.memory_id)
            .filter(
                MemoryRecord.user_id == user_id,
                MemoryRecord.active.is_(True),
            )
        )
        if scope == "user":
            query = query.filter(MemoryRecord.scope == "user")
        elif scope == "folder":
            if folder_id is None:
                raise ValueError("folder_id is required for folder search")
            allowed_chat_ids = session.query(ChatSession.chat_session_id).filter(
                ChatSession.user_id == user_id,
                ChatSession.folder_id == folder_id,
                ChatSession.memory_enabled.is_(True),
            )
            query = query.filter(
                or_(
                    (
                        (MemoryRecord.scope == "folder")
                        & (MemoryRecord.folder_id == folder_id)
                    ),
                    (
                        (MemoryRecord.scope == "thread")
                        & MemoryRecord.chat_session_id.in_(allowed_chat_ids)
                    ),
                )
            )
        elif scope == "thread":
            if chat_session_id is None:
                raise ValueError("chat_session_id is required for thread search")
            owned_chat = (
                session.query(ChatSession)
                .filter(
                    ChatSession.chat_session_id == chat_session_id,
                    ChatSession.user_id == user_id,
                    ChatSession.memory_enabled.is_(True),
                )
                .first()
            )
            if owned_chat is None:
                return []
            query = query.filter(
                MemoryRecord.scope == "thread",
                MemoryRecord.chat_session_id == chat_session_id,
            )
        else:
            raise ValueError("scope must be user, folder, or thread")

        scored = []
        for record, embedding in query.all():
            vector = unpack_vector(embedding.vector, embedding.dimensions)
            semantic = cosine_similarity(query_vector, vector)
            record_tokens = set(record.content.lower().split())
            lexical = len(query_tokens & record_tokens) / max(1, len(query_tokens))
            score = semantic * 0.75 + lexical * 0.2 + float(record.importance) * 0.05
            scored.append(
                {
                    "memory_id": record.memory_id,
                    "scope": record.scope,
                    "record_type": record.record_type,
                    "content": record.content,
                    "folder_id": record.folder_id,
                    "chat_session_id": record.chat_session_id,
                    "content_hash": record.content_hash,
                    "score": round(score, 6),
                }
            )
        scored.sort(key=lambda item: (item["score"], item["memory_id"]), reverse=True)
        unique_results = []
        seen_hashes: set[str] = set()
        for item in scored:
            content_hash = str(item.pop("content_hash"))
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)
            unique_results.append(item)
            if len(unique_results) >= limit:
                break
        return unique_results


def get_profile(user_id: int) -> dict[str, Any]:
    with SessionLocal() as session:
        summary = (
            session.query(MemoryRecord)
            .filter(
                MemoryRecord.user_id == user_id,
                MemoryRecord.record_type == "user_profile_summary",
                MemoryRecord.active.is_(True),
            )
            .order_by(MemoryRecord.memory_id.desc())
            .first()
        )
        facts = (
            session.query(MemoryRecord)
            .filter(
                MemoryRecord.user_id == user_id,
                MemoryRecord.record_type == "user_fact",
                MemoryRecord.active.is_(True),
            )
            .order_by(MemoryRecord.memory_id.desc())
            .all()
        )
        return {
            "summary": summary.content if summary else None,
            "facts": [
                {
                    "memory_id": fact.memory_id,
                    "content": fact.content,
                    "source_chat_session_id": fact.source_chat_session_id,
                }
                for fact in facts
            ],
        }


def update_memory_record(
    user_id: int,
    memory_id: int,
    content: str,
) -> dict[str, Any] | None:
    clean_content = " ".join(content.split())
    if not clean_content:
        raise ValueError("Memory content is required")
    if is_secret_like(clean_content):
        raise ValueError("Secret-like values cannot be stored in searchable memory")
    with SessionLocal() as session:
        record = (
            session.query(MemoryRecord)
            .filter(
                MemoryRecord.memory_id == memory_id,
                MemoryRecord.user_id == user_id,
                MemoryRecord.record_type == "user_fact",
                MemoryRecord.active.is_(True),
            )
            .first()
        )
        if record is None:
            return None
        record.content = clean_content
        record.content_hash = hashlib.sha256(clean_content.encode("utf-8")).hexdigest()
        record.importance = cast(Any, 1.0)
        record.confidence = cast(Any, 1.0)
        embedding = (
            session.query(MemoryEmbedding)
            .filter(MemoryEmbedding.memory_id == memory_id)
            .first()
        )
        if embedding is None:
            embedding = MemoryEmbedding(memory_id=memory_id)
            session.add(embedding)
        embedding.model_id = EMBEDDING_MODEL_ID
        embedding.dimensions = EMBEDDING_DIMENSIONS
        embedding.vector = pack_vector(embed_text(clean_content))
        embedding.content_hash = record.content_hash
        rebuild_profile_summary(session, user_id)
        session.commit()
        return {
            "memory_id": record.memory_id,
            "content": record.content,
            "scope": record.scope,
            "record_type": record.record_type,
        }


def delete_memory_record(user_id: int, memory_id: int) -> bool:
    with SessionLocal() as session:
        record = (
            session.query(MemoryRecord)
            .filter(
                MemoryRecord.memory_id == memory_id,
                MemoryRecord.user_id == user_id,
                MemoryRecord.record_type == "user_fact",
                MemoryRecord.active.is_(True),
            )
            .first()
        )
        if record is None:
            return False
        record.active = False
        session.flush()
        rebuild_profile_summary(session, user_id)
        session.commit()
        return True
