from fastapi import APIRouter, HTTPException, status

from app.models.database.geist_user import get_default_user
from app.schemas.memory import (
    ChatMemoryUpdate,
    FolderCreate,
    FolderUpdate,
    MemoryRecordUpdate,
    MemorySearchRequest,
)
from app.services.memory_service import (
    create_folder,
    delete_folder,
    delete_memory_record,
    get_chat_memory_settings,
    get_profile,
    list_folders,
    search_memories,
    update_chat_memory_settings,
    update_folder,
    update_memory_record,
)


router = APIRouter()


def _user_id() -> int:
    return int(get_default_user().user_id)


@router.get("/folders")
def folders():
    return list_folders(_user_id())


@router.post("/folders", status_code=status.HTTP_201_CREATED)
def add_folder(payload: FolderCreate):
    try:
        return create_folder(_user_id(), payload.name, payload.color)
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.patch("/folders/{folder_id}")
def edit_folder(folder_id: int, payload: FolderUpdate):
    try:
        folder = update_folder(
            _user_id(),
            folder_id,
            **payload.model_dump(exclude_none=True),
        )
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    if folder is None:
        raise HTTPException(status_code=404, detail="Folder not found")
    return folder


@router.delete("/folders/{folder_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_folder(folder_id: int):
    if not delete_folder(_user_id(), folder_id):
        raise HTTPException(status_code=404, detail="Folder not found")


@router.get("/chats/{chat_session_id}")
def chat_memory(chat_session_id: int):
    settings = get_chat_memory_settings(_user_id(), chat_session_id)
    if settings is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    return settings


@router.patch("/chats/{chat_session_id}")
def edit_chat_memory(chat_session_id: int, payload: ChatMemoryUpdate):
    try:
        settings = update_chat_memory_settings(
            _user_id(),
            chat_session_id,
            **payload.model_dump(),
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    if settings is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    return settings


@router.get("/profile")
def profile():
    return get_profile(_user_id())


@router.patch("/records/{memory_id}")
def edit_memory_record(memory_id: int, payload: MemoryRecordUpdate):
    try:
        record = update_memory_record(_user_id(), memory_id, payload.content)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    if record is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return record


@router.delete("/records/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_memory_record(memory_id: int):
    if not delete_memory_record(_user_id(), memory_id):
        raise HTTPException(status_code=404, detail="Memory not found")


@router.post("/search")
def search(payload: MemorySearchRequest):
    try:
        return {
            "results": search_memories(
                _user_id(),
                payload.query,
                scope=payload.scope,
                folder_id=payload.folder_id,
                chat_session_id=payload.chat_session_id,
                limit=payload.limit,
            )
        }
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
