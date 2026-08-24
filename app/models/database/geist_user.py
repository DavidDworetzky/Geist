from dataclasses import dataclass

from sqlalchemy import Column, Integer, String
from sqlalchemy.exc import IntegrityError

from app.models.database.database import Base, SessionLocal


DEFAULT_WORKSPACE_KEY = "default"
DEFAULT_WORKSPACE_USERNAME = "local"
DEFAULT_WORKSPACE_NAME = "Local User"

# This value is retained only to adopt databases created by older Geist builds.
# New rows and runtime identity resolution never use an email address.
LEGACY_DEFAULT_EMAIL = "david@phantasmal.ai"
LEGACY_DEFAULT_USERNAME = "ddworetzky"
LEGACY_DEFAULT_NAME = "David Dworetzky"


class GeistUser(Base):
    __tablename__ = "geist_user"
    user_id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_key = Column(String, nullable=True, unique=True, index=True)
    username = Column(String)
    name = Column(String)
    email = Column(String)
    # Retained for database compatibility; Geist does not authenticate with it.
    password = Column(String)


@dataclass
class UserModel:
    user_id: int
    workspace_key: str | None
    username: str | None
    name: str | None
    email: str | None


def _to_model(user: GeistUser) -> UserModel:
    return UserModel(
        user_id=user.user_id,
        workspace_key=user.workspace_key,
        username=user.username,
        name=user.name,
        email=user.email,
    )


def get_user_by_id(user_id: int) -> UserModel:
    with SessionLocal() as session:
        user = session.query(GeistUser).filter_by(user_id=user_id).first()
        if user is None:
            raise LookupError(f"Geist user {user_id} does not exist")
        return _to_model(user)


def create_user(user: UserModel) -> UserModel:
    with SessionLocal() as session:
        database_user = GeistUser(
            user_id=user.user_id,
            workspace_key=user.workspace_key,
            username=user.username,
            name=user.name,
            email=user.email,
        )
        session.add(database_user)
        session.commit()
        session.refresh(database_user)
        return _to_model(database_user)


def ensure_default_user() -> UserModel:
    """Return the neutral local workspace, creating or adopting it as needed."""
    with SessionLocal() as session:
        user = (
            session.query(GeistUser)
            .filter_by(workspace_key=DEFAULT_WORKSPACE_KEY)
            .first()
        )
        legacy_users = (
            session.query(GeistUser)
            .filter_by(email=LEGACY_DEFAULT_EMAIL)
            .order_by(GeistUser.user_id)
            .all()
        )
        if user is None:
            if legacy_users:
                user = legacy_users[0]
                user.workspace_key = DEFAULT_WORKSPACE_KEY
            else:
                user = GeistUser(
                    workspace_key=DEFAULT_WORKSPACE_KEY,
                    username=DEFAULT_WORKSPACE_USERNAME,
                    name=DEFAULT_WORKSPACE_NAME,
                    email=None,
                    password=None,
                )
                session.add(user)

        for legacy_user in legacy_users:
            legacy_user.email = None
            legacy_user.password = None
            if legacy_user.username == LEGACY_DEFAULT_USERNAME:
                legacy_user.username = DEFAULT_WORKSPACE_USERNAME
            if legacy_user.name == LEGACY_DEFAULT_NAME:
                legacy_user.name = DEFAULT_WORKSPACE_NAME

        if session.new or session.dirty:
            try:
                session.commit()
            except IntegrityError:
                # Concurrent startup may have created the unique workspace row.
                session.rollback()
                user = (
                    session.query(GeistUser)
                    .filter_by(workspace_key=DEFAULT_WORKSPACE_KEY)
                    .one()
                )
            else:
                session.refresh(user)
        return _to_model(user)


def get_default_user() -> UserModel:
    """Return the local workspace identity used until multi-user auth exists."""
    return ensure_default_user()
