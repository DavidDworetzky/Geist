import datetime
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Text
from sqlalchemy.orm import relationship
from app.models.database.database import Base, SessionLocal
from dataclasses import dataclass
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class Note(Base):
    __tablename__ = "notes"

    note_id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False, default="")
    user_id = Column(Integer, ForeignKey('geist_user.user_id'), nullable=False)
    create_date = Column(DateTime, default=datetime.datetime.utcnow)
    update_date = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    user = relationship("GeistUser", backref="notes")


@dataclass
class NoteModel:
    note_id: Optional[int] = None
    title: str = ""
    content: str = ""
    user_id: int = 0
    create_date: Optional[datetime.datetime] = None
    update_date: Optional[datetime.datetime] = None


@dataclass
class NoteResponse:
    note_id: int
    title: str
    content: str
    user_id: int
    create_date: Optional[datetime.datetime] = None
    update_date: Optional[datetime.datetime] = None


def _note_to_response(db_note: Note) -> NoteResponse:
    return NoteResponse(
        note_id=db_note.note_id,
        title=db_note.title,
        content=db_note.content,
        user_id=db_note.user_id,
        create_date=db_note.create_date,
        update_date=db_note.update_date
    )


def create_note(title: str, content: str, user_id: int) -> NoteResponse:
    with SessionLocal() as session:
        now = datetime.datetime.utcnow()
        db_note = Note(
            title=title,
            content=content,
            user_id=user_id,
            create_date=now,
            update_date=now
        )
        session.add(db_note)
        session.commit()
        session.refresh(db_note)
        return _note_to_response(db_note)


def get_note_by_id(note_id: int) -> Optional[NoteResponse]:
    with SessionLocal() as session:
        db_note = session.query(Note).filter_by(note_id=note_id).first()
        if not db_note:
            return None
        return _note_to_response(db_note)


def get_notes_by_user(user_id: int, skip: int = 0, limit: int = 100) -> List[NoteResponse]:
    with SessionLocal() as session:
        notes = (
            session.query(Note)
            .filter_by(user_id=user_id)
            .order_by(Note.update_date.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )
        return [_note_to_response(n) for n in notes]


def update_note(note_id: int, user_id: int, title: Optional[str] = None, content: Optional[str] = None) -> Optional[NoteResponse]:
    with SessionLocal() as session:
        db_note = session.query(Note).filter_by(note_id=note_id, user_id=user_id).first()
        if not db_note:
            return None
        if title is not None:
            db_note.title = title
        if content is not None:
            db_note.content = content
        db_note.update_date = datetime.datetime.utcnow()
        session.commit()
        session.refresh(db_note)
        return _note_to_response(db_note)


def delete_note(note_id: int, user_id: int) -> bool:
    with SessionLocal() as session:
        db_note = session.query(Note).filter_by(note_id=note_id, user_id=user_id).first()
        if not db_note:
            return False
        session.delete(db_note)
        session.commit()
        return True


def search_notes(user_id: int, query: str, skip: int = 0, limit: int = 100) -> List[NoteResponse]:
    with SessionLocal() as session:
        search_term = f"%{query}%"
        notes = (
            session.query(Note)
            .filter(
                Note.user_id == user_id,
                (Note.title.ilike(search_term) | Note.content.ilike(search_term))
            )
            .order_by(Note.update_date.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )
        return [_note_to_response(n) for n in notes]


def get_note_by_title(user_id: int, title: str) -> Optional[NoteResponse]:
    with SessionLocal() as session:
        db_note = session.query(Note).filter_by(user_id=user_id, title=title).first()
        if not db_note:
            return None
        return _note_to_response(db_note)
