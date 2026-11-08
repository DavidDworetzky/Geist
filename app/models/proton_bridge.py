from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr


class ProtonBridgeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="proton-mail", pattern=r"^[A-Za-z0-9_-]{1,64}$")
    host: Literal["127.0.0.1", "::1"] = "127.0.0.1"
    imap_port: int = Field(default=1143, ge=1, le=65535)
    smtp_port: int = Field(default=1025, ge=1, le=65535)
    imap_security: Literal["starttls", "tls"] = "starttls"
    smtp_security: Literal["starttls", "tls"] = "starttls"
    username: str = Field(min_length=1, max_length=254)
    sender: str = Field(min_length=1, max_length=254)
    password: SecretStr | None = None
    certificate: str = Field(min_length=1, max_length=16000)
    recipient_allowlist: list[str] = Field(default_factory=list, max_length=200)
    max_writes_per_hour: int = Field(default=20, ge=1, le=1000)
