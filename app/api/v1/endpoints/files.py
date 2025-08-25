from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from fastapi.responses import StreamingResponse
from typing import List, Optional, Dict, Any
import io
from app.models.database.file_upload import (
    FileUploadResponse, get_files_by_user, get_file_upload_by_id, delete_file_upload
)
from app.models.database.geist_user import get_default_user
from app.services.file_storage import FileStorageService
from app.services.file_processor import FileProcessingService

router = APIRouter()

# Dependency to get current user (simplified for now)
def get_current_user():
    """Get the current user - simplified implementation using default user"""
    return get_default_user()

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    current_user = Depends(get_current_user)
) -> Dict[str, Any]:
    """Upload a file and store it in database"""
    try:
        # Read file data
        file_data = await file.read()
        
        # Store file using the file storage service
        storage_result = FileStorageService.store_file(
            file_data=file_data,
            original_filename=file.filename or "unknown_file",
            user_id=current_user.user_id
        )
        
        if not storage_result['success']:
            raise HTTPException(status_code=400, detail=storage_result['error'])
        
        # If file is newly uploaded (not duplicate), process it asynchronously
        file_id = storage_result['file_id']
        if not storage_result.get('duplicate', False):
            # Start processing in background
            try:
                processing_result = await FileProcessingService.process_file(file_id)
                storage_result['processing_status'] = processing_result
            except Exception as e:
                # Processing failed, but file is still uploaded
                storage_result['processing_status'] = {
                    'success': False,
                    'error': f'Processing failed: {str(e)}'
                }
        
        return {
            'success': True,
            'file_id': file_id,
            'filename': storage_result['filename'],
            'file_size': storage_result.get('file_size', 0),
            'mime_type': storage_result.get('mime_type', ''),
            'duplicate': storage_result.get('duplicate', False),
            'processing_status': storage_result.get('processing_status', {})
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

@router.get("/")
async def list_files(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    search: Optional[str] = Query(None),
    file_type: Optional[str] = Query(None),
    current_user = Depends(get_current_user)
) -> Dict[str, Any]:
    """List uploaded files with filtering and pagination"""
    try:
        files = get_files_by_user(current_user.user_id, skip=skip, limit=limit)
        
        # Apply search filter if provided
        if search:
            search_lower = search.lower()
            files = [f for f in files if search_lower in f.filename.lower() or 
                    search_lower in f.original_filename.lower()]
        
        # Apply file type filter if provided
        if file_type:
            files = [f for f in files if f.mime_type.startswith(file_type)]
        
        return {
            'success': True,
            'files': [
                {
                    'file_id': f.file_id,
                    'filename': f.filename,
                    'original_filename': f.original_filename,
                    'file_size': f.file_size,
                    'mime_type': f.mime_type,
                    'upload_date': f.upload_date.isoformat() if f.upload_date else None,
                    'is_processed': f.is_processed,
                    'processing_error': f.processing_error
                }
                for f in files
            ],
            'total': len(files),
            'skip': skip,
            'limit': limit
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")

@router.get("/{file_id}")
async def get_file_metadata(
    file_id: int,
    current_user = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get file metadata by ID"""
    try:
        file_info = FileStorageService.get_file_info(file_id)
        if not file_info:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Check if user owns the file
        if file_info['user_id'] != current_user.user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        return {
            'success': True,
            **file_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get file metadata: {str(e)}")

@router.get("/{file_id}/download")
async def download_file(
    file_id: int,
    current_user = Depends(get_current_user)
):
    """Download the original file"""
    try:
        file_upload = get_file_upload_by_id(file_id)
        if not file_upload:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Check if user owns the file
        if file_upload.user_id != current_user.user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Create file stream
        file_stream = io.BytesIO(file_upload.file_data)
        
        return StreamingResponse(
            io.BytesIO(file_upload.file_data),
            media_type=file_upload.mime_type,
            headers={
                'Content-Disposition': f'attachment; filename="{file_upload.original_filename}"',
                'Content-Length': str(file_upload.file_size)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File download failed: {str(e)}")

@router.get("/{file_id}/content")
async def get_file_content(
    file_id: int,
    current_user = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get extracted text content"""
    try:
        file_upload = get_file_upload_by_id(file_id)
        if not file_upload:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Check if user owns the file
        if file_upload.user_id != current_user.user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        return {
            'success': True,
            'file_id': file_upload.file_id,
            'filename': file_upload.filename,
            'is_processed': file_upload.is_processed,
            'processing_error': file_upload.processing_error,
            'extracted_text': file_upload.extracted_text,
            'text_length': len(file_upload.extracted_text) if file_upload.extracted_text else 0
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get file content: {str(e)}")

@router.delete("/{file_id}")
async def delete_file(
    file_id: int,
    current_user = Depends(get_current_user)
) -> Dict[str, Any]:
    """Delete a file and its metadata"""
    try:
        delete_result = FileStorageService.delete_file(file_id, current_user.user_id)
        
        if not delete_result['success']:
            if 'not found' in delete_result['error'].lower():
                raise HTTPException(status_code=404, detail=delete_result['error'])
            else:
                raise HTTPException(status_code=403, detail=delete_result['error'])
        
        return delete_result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File deletion failed: {str(e)}")

@router.post("/{file_id}/reprocess")
async def reprocess_file(
    file_id: int,
    force: bool = Query(False),
    current_user = Depends(get_current_user)
) -> Dict[str, Any]:
    """Reprocess a file (useful if processing failed or needs updating)"""
    try:
        # Check if file exists and user owns it
        file_upload = get_file_upload_by_id(file_id)
        if not file_upload:
            raise HTTPException(status_code=404, detail="File not found")
        
        if file_upload.user_id != current_user.user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Reprocess the file
        processing_result = FileProcessingService.reprocess_file(file_id, force=force)
        
        return {
            'success': True,
            'file_id': file_id,
            'processing_result': processing_result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File reprocessing failed: {str(e)}")

@router.get("/{file_id}/status")
async def get_processing_status(
    file_id: int,
    current_user = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get the current processing status of a file"""
    try:
        # Check if file exists and user owns it
        file_upload = get_file_upload_by_id(file_id)
        if not file_upload:
            raise HTTPException(status_code=404, detail="File not found")
        
        if file_upload.user_id != current_user.user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get processing status
        status_result = FileProcessingService.get_processing_status(file_id)
        
        return {
            'success': True,
            **status_result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get processing status: {str(e)}")

@router.post("/search")
async def search_files(
    query: str = Query(..., min_length=1),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    current_user = Depends(get_current_user)
) -> Dict[str, Any]:
    """Search files by content and metadata"""
    try:
        # Get all user files
        files = get_files_by_user(current_user.user_id, skip=0, limit=1000)  # Get more files for search
        
        # Search in filename, original filename, and extracted text
        query_lower = query.lower()
        matching_files = []
        
        for file in files:
            match_score = 0
            match_type = []
            
            # Search in filename
            if query_lower in file.filename.lower():
                match_score += 2
                match_type.append('filename')
            
            # Search in original filename  
            if query_lower in file.original_filename.lower():
                match_score += 2
                match_type.append('original_filename')
            
            # Search in extracted text (if available)
            if file.is_processed and hasattr(file, 'extracted_text'):
                file_detail = get_file_upload_by_id(file.file_id)
                if file_detail and file_detail.extracted_text and query_lower in file_detail.extracted_text.lower():
                    match_score += 1
                    match_type.append('content')
            
            if match_score > 0:
                matching_files.append({
                    'file_id': file.file_id,
                    'filename': file.filename,
                    'original_filename': file.original_filename,
                    'file_size': file.file_size,
                    'mime_type': file.mime_type,
                    'upload_date': file.upload_date.isoformat() if file.upload_date else None,
                    'is_processed': file.is_processed,
                    'match_score': match_score,
                    'match_type': match_type
                })
        
        # Sort by match score (descending)
        matching_files.sort(key=lambda x: x['match_score'], reverse=True)
        
        # Apply pagination
        paginated_files = matching_files[skip:skip + limit]
        
        return {
            'success': True,
            'files': paginated_files,
            'total_matches': len(matching_files),
            'skip': skip,
            'limit': limit,
            'query': query
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")