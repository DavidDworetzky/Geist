"""
ProviderCredential database model for storing per-provider API keys.

Keys entered through the UI land here so online providers and gated
Hugging Face downloads work without editing the server's .env file.
Raw keys never leave the backend: API responses only carry a masked hint.
"""
import datetime
from dataclasses import dataclass

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, UniqueConstraint

from app.models.database.database import Base, SessionLocal


class ProviderCredential(Base):
    """
    One stored API key (and optional base URL) per user and provider.
    """
    __tablename__ = 'provider_credential'
    __table_args__ = (
        UniqueConstraint('user_id', 'provider_id', name='uq_provider_credential_user_provider'),
    )

    provider_credential_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('geist_user.user_id'), nullable=False, index=True)
    provider_id = Column(String, nullable=False)
    api_key = Column(String, nullable=False)
    base_url = Column(String, nullable=True)

    create_date = Column(DateTime, default=datetime.datetime.utcnow)
    update_date = Column(
        DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow
    )


@dataclass
class ProviderCredentialModel:
    """
    Data model for a stored provider credential.
    """
    provider_credential_id: int
    user_id: int
    provider_id: str
    api_key: str
    base_url: str | None
    create_date: datetime.datetime
    update_date: datetime.datetime


def _to_model(credential: ProviderCredential) -> ProviderCredentialModel:
    return ProviderCredentialModel(
        provider_credential_id=credential.provider_credential_id,
        user_id=credential.user_id,
        provider_id=credential.provider_id,
        api_key=credential.api_key,
        base_url=credential.base_url,
        create_date=credential.create_date,
        update_date=credential.update_date,
    )


def upsert_provider_credential(
    user_id: int,
    provider_id: str,
    api_key: str,
    base_url: str | None = None,
) -> ProviderCredentialModel:
    """Create or replace the stored key for one user/provider pair."""
    normalized_provider = provider_id.strip().lower()
    with SessionLocal() as session:
        credential = (
            session.query(ProviderCredential)
            .filter_by(user_id=user_id, provider_id=normalized_provider)
            .first()
        )
        if credential is None:
            credential = ProviderCredential(
                user_id=user_id,
                provider_id=normalized_provider,
                api_key=api_key,
                base_url=base_url,
            )
            session.add(credential)
        else:
            credential.api_key = api_key
            credential.base_url = base_url
            credential.update_date = datetime.datetime.utcnow()
        session.commit()
        session.refresh(credential)
        return _to_model(credential)


def get_provider_credential(user_id: int, provider_id: str) -> ProviderCredentialModel | None:
    """Return the stored credential for one user/provider pair, if any."""
    with SessionLocal() as session:
        credential = (
            session.query(ProviderCredential)
            .filter_by(user_id=user_id, provider_id=provider_id.strip().lower())
            .first()
        )
        return _to_model(credential) if credential else None


def list_provider_credentials(user_id: int) -> list[ProviderCredentialModel]:
    """Return all stored credentials for a user."""
    with SessionLocal() as session:
        credentials = (
            session.query(ProviderCredential)
            .filter_by(user_id=user_id)
            .order_by(ProviderCredential.provider_id)
            .all()
        )
        return [_to_model(credential) for credential in credentials]


def delete_provider_credential(user_id: int, provider_id: str) -> bool:
    """Delete the stored credential for one user/provider pair."""
    with SessionLocal() as session:
        deleted = (
            session.query(ProviderCredential)
            .filter_by(user_id=user_id, provider_id=provider_id.strip().lower())
            .delete()
        )
        session.commit()
        return deleted > 0
