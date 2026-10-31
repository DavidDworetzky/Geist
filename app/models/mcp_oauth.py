from pydantic import BaseModel, ConfigDict, Field


class OAuthStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")


class OAuthProviderInfo(BaseModel):
    id: str
    label: str
    scopes: list[str]
    ready: bool
    setup_message: str | None = None


class OAuthStatus(BaseModel):
    provider: str | None = None
    status: str = "disconnected"
    providers: list[OAuthProviderInfo] = Field(default_factory=list)
    redirect_uri: str | None = None


class OAuthStartResponse(BaseModel):
    authorization_url: str


class OAuthDisconnectResponse(BaseModel):
    disconnected: bool = True
    revocation_complete: bool = False
