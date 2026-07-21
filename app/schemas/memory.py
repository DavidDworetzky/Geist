from pydantic import BaseModel, Field


class FolderCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    color: str = Field(default="violet", max_length=20)


class FolderUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    color: str | None = Field(default=None, max_length=20)


class ChatMemoryUpdate(BaseModel):
    memory_enabled: bool | None = None
    memory_mode: str | None = None
    folder_id: int | None = None
    clear_folder: bool = False


class MemorySearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    scope: str = "user"
    folder_id: int | None = None
    chat_session_id: int | None = None
    limit: int = Field(default=8, ge=1, le=30)


class MemoryRecordUpdate(BaseModel):
    content: str = Field(min_length=1, max_length=4000)
