from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.database import chat_session
from app.models.database.database import Base


def test_chat_identity_is_allocated_before_history_and_reused(monkeypatch, tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'chat.sqlite3'}")
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine)
    monkeypatch.setattr(chat_session, "SessionLocal", sessions)
    try:
        first = chat_session.create_chat_session(1, memory_enabled=False, memory_mode="private")
        second = chat_session.create_chat_session(1)
        assert first != second
        with sessions() as session:
            saved = session.get(chat_session.ChatSession, first)
            assert saved.chat_history == "[]"
            assert saved.user_id == 1
            assert saved.memory_enabled is False
            assert saved.memory_mode == "private"
        updated = chat_session.update_chat_history("hello", "hi", session_id=first, user_id=1)
        assert updated.chat_session_id == first
    finally:
        engine.dispose()
