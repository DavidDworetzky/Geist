import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)

from app.models.database.database import Base


class MemoryFolder(Base):
    __tablename__ = "memory_folder"

    folder_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("geist_user.user_id"), nullable=False, index=True)
    name = Column(String(120), nullable=False)
    color = Column(String(20), nullable=False, default="violet")
    revision = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    updated_at = Column(
        DateTime,
        nullable=False,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
    )

    __table_args__ = (UniqueConstraint("user_id", "name", name="uq_memory_folder_user_name"),)


class MemoryRecord(Base):
    __tablename__ = "memory_record"

    memory_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("geist_user.user_id"), nullable=False, index=True)
    scope = Column(String(20), nullable=False, index=True)
    record_type = Column(String(40), nullable=False, index=True)
    content = Column(Text, nullable=False)
    folder_id = Column(
        Integer,
        ForeignKey("memory_folder.folder_id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    chat_session_id = Column(
        Integer,
        ForeignKey("chat_session.chat_session_id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    source_chat_session_id = Column(
        Integer,
        ForeignKey("chat_session.chat_session_id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    source_from_revision = Column(Integer, nullable=True)
    source_through_revision = Column(Integer, nullable=True)
    importance = Column(Float, nullable=False, default=0.5)
    confidence = Column(Float, nullable=False, default=1.0)
    active = Column(Boolean, nullable=False, default=True, index=True)
    content_hash = Column(String(64), nullable=False, index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    updated_at = Column(
        DateTime,
        nullable=False,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
    )

    __table_args__ = (
        UniqueConstraint(
            "chat_session_id",
            "source_through_revision",
            "record_type",
            "content_hash",
            name="uq_memory_record_chat_revision_type",
        ),
    )


class MemoryEmbedding(Base):
    __tablename__ = "memory_embedding"

    memory_id = Column(
        Integer,
        ForeignKey("memory_record.memory_id", ondelete="CASCADE"),
        primary_key=True,
    )
    model_id = Column(String(100), nullable=False)
    dimensions = Column(Integer, nullable=False)
    vector = Column(LargeBinary, nullable=False)
    content_hash = Column(String(64), nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
