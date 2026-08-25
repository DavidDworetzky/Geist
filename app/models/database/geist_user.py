from dataclasses import dataclass

from sqlalchemy import Column, Integer, String
from sqlalchemy.exc import IntegrityError

from app.models.database.database import Base, SessionLocal


DEFAULT_WORKSPACE_KEY = "default"
DEFAULT_WORKSPACE_NAME = "Local Workspace"

# This value is retained only to adopt databases created by older Geist builds.
# New rows and runtime identity resolution never use an email address.
LEGACY_DEFAULT_EMAIL = "david@phantasmal.ai"
LEGACY_DEFAULT_USERNAME = "ddworetzky"
LEGACY_DEFAULT_NAME = "David Dworetzky"
LEGACY_NEUTRAL_USERNAME = "local"
LEGACY_NEUTRAL_NAME = "Local User"


class GeistUser(Base):
    __tablename__ = "geist_user"
    user_id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_key = Column(String, nullable=True, unique=True, index=True)
    username = Column(String)
    name = Column(String)
    email = Column(String)
    # Retained for database compatibility; Geist does not authenticate with it.
    password = Column(String)


@dataclass(frozen=True)
class WorkspaceModel:
    """Application identity for the singleton local data workspace."""

    workspace_id: int
    workspace_key: str
    display_name: str | None


def _to_workspace(user: GeistUser) -> WorkspaceModel:
    if user.workspace_key is None:
        raise ValueError("Workspace row is missing its stable key")
    return WorkspaceModel(
        workspace_id=int(user.user_id),
        workspace_key=user.workspace_key,
        display_name=user.name,
    )


def ensure_default_workspace() -> WorkspaceModel:
    """Return the neutral local workspace, creating or adopting it as needed."""
    with SessionLocal() as session:
        user = session.query(GeistUser).filter_by(workspace_key=DEFAULT_WORKSPACE_KEY).first()
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
                    username=None,
                    name=DEFAULT_WORKSPACE_NAME,
                    email=None,
                    password=None,
                )
                session.add(user)

        # Normalize a workspace created by an earlier version of this migration,
        # including local databases used to test the unmerged branch.
        user.email = None
        user.password = None
        if user.username in {LEGACY_DEFAULT_USERNAME, LEGACY_NEUTRAL_USERNAME}:
            user.username = None
        if user.name in {LEGACY_DEFAULT_NAME, LEGACY_NEUTRAL_NAME}:
            user.name = DEFAULT_WORKSPACE_NAME

        for legacy_user in legacy_users:
            legacy_user.email = None
            legacy_user.password = None
            if legacy_user.username == LEGACY_DEFAULT_USERNAME:
                legacy_user.username = None
            if legacy_user.name == LEGACY_DEFAULT_NAME:
                legacy_user.name = DEFAULT_WORKSPACE_NAME

        if session.new or session.dirty:
            try:
                session.commit()
            except IntegrityError:
                # Concurrent startup may have created the unique workspace row.
                session.rollback()
                user = session.query(GeistUser).filter_by(workspace_key=DEFAULT_WORKSPACE_KEY).one()
            else:
                session.refresh(user)
        return _to_workspace(user)


def get_default_workspace() -> WorkspaceModel:
    """Return the singleton local workspace used to partition owned data."""
    return ensure_default_workspace()
