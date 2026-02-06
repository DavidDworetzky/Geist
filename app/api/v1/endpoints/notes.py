from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional, Dict, Any
from pydantic import BaseModel
from app.models.database.note import (
    create_note, get_note_by_id, get_notes_by_user,
    update_note, delete_note, search_notes, get_note_by_title
)
from app.models.database.geist_user import get_default_user

router = APIRouter()


class NoteCreateRequest(BaseModel):
    title: str
    content: str = ""


class NoteUpdateRequest(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None


def get_current_user():
    return get_default_user()


@router.post("/")
async def create_note_endpoint(
    request: NoteCreateRequest,
    current_user=Depends(get_current_user)
) -> Dict[str, Any]:
    try:
        note = create_note(
            title=request.title,
            content=request.content,
            user_id=current_user.user_id
        )
        return {
            'success': True,
            'note': {
                'note_id': note.note_id,
                'title': note.title,
                'content': note.content,
                'user_id': note.user_id,
                'create_date': note.create_date.isoformat() if note.create_date else None,
                'update_date': note.update_date.isoformat() if note.update_date else None
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create note: {str(e)}")


@router.get("/")
async def list_notes(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    search: Optional[str] = Query(None),
    current_user=Depends(get_current_user)
) -> Dict[str, Any]:
    try:
        if search:
            notes = search_notes(current_user.user_id, search, skip=skip, limit=limit)
        else:
            notes = get_notes_by_user(current_user.user_id, skip=skip, limit=limit)

        return {
            'success': True,
            'notes': [
                {
                    'note_id': n.note_id,
                    'title': n.title,
                    'content': n.content,
                    'user_id': n.user_id,
                    'create_date': n.create_date.isoformat() if n.create_date else None,
                    'update_date': n.update_date.isoformat() if n.update_date else None
                }
                for n in notes
            ],
            'total': len(notes),
            'skip': skip,
            'limit': limit
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list notes: {str(e)}")


@router.get("/{note_id}")
async def get_note_endpoint(
    note_id: int,
    current_user=Depends(get_current_user)
) -> Dict[str, Any]:
    try:
        note = get_note_by_id(note_id)
        if not note:
            raise HTTPException(status_code=404, detail="Note not found")
        if note.user_id != current_user.user_id:
            raise HTTPException(status_code=403, detail="Access denied")

        return {
            'success': True,
            'note': {
                'note_id': note.note_id,
                'title': note.title,
                'content': note.content,
                'user_id': note.user_id,
                'create_date': note.create_date.isoformat() if note.create_date else None,
                'update_date': note.update_date.isoformat() if note.update_date else None
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get note: {str(e)}")


@router.get("/by-title/{title}")
async def get_note_by_title_endpoint(
    title: str,
    current_user=Depends(get_current_user)
) -> Dict[str, Any]:
    try:
        note = get_note_by_title(current_user.user_id, title)
        if not note:
            raise HTTPException(status_code=404, detail="Note not found")

        return {
            'success': True,
            'note': {
                'note_id': note.note_id,
                'title': note.title,
                'content': note.content,
                'user_id': note.user_id,
                'create_date': note.create_date.isoformat() if note.create_date else None,
                'update_date': note.update_date.isoformat() if note.update_date else None
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get note: {str(e)}")


@router.put("/{note_id}")
async def update_note_endpoint(
    note_id: int,
    request: NoteUpdateRequest,
    current_user=Depends(get_current_user)
) -> Dict[str, Any]:
    try:
        note = update_note(
            note_id=note_id,
            user_id=current_user.user_id,
            title=request.title,
            content=request.content
        )
        if not note:
            raise HTTPException(status_code=404, detail="Note not found or access denied")

        return {
            'success': True,
            'note': {
                'note_id': note.note_id,
                'title': note.title,
                'content': note.content,
                'user_id': note.user_id,
                'create_date': note.create_date.isoformat() if note.create_date else None,
                'update_date': note.update_date.isoformat() if note.update_date else None
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update note: {str(e)}")


@router.delete("/{note_id}")
async def delete_note_endpoint(
    note_id: int,
    current_user=Depends(get_current_user)
) -> Dict[str, Any]:
    try:
        deleted = delete_note(note_id, current_user.user_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Note not found or access denied")

        return {'success': True, 'message': 'Note deleted'}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete note: {str(e)}")
