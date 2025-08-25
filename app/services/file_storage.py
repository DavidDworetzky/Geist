import hashlib
import magic
import mimetypes
from typing import Optional, List, Dict, Any
from app.models.database.file_upload import FileUploadModel, create_file_upload, get_file_by_hash

# Configuration constants
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_FILE_TYPES = [
    '.pdf', '.txt', '.docx', '.xlsx', '.csv', '.md', '.json', '.xml',
    '.doc', '.xls', '.ppt', '.pptx', '.rtf', '.odt', '.ods', '.odp'
]

class FileStorageService:
    """Service for handling file uploads, validation, and storage"""
    
    @staticmethod
    def validate_file(file_data: bytes, filename: str) -> Dict[str, Any]:
        """
        Validate file size, type, and content
        Returns dict with 'valid' bool and 'error' message if invalid
        """
        # Check file size
        if len(file_data) > MAX_FILE_SIZE:
            return {
                'valid': False,
                'error': f'File size {len(file_data)} bytes exceeds maximum allowed size of {MAX_FILE_SIZE} bytes'
            }
        
        # Check file extension
        file_ext = None
        if '.' in filename:
            file_ext = '.' + filename.rsplit('.', 1)[1].lower()
            
        if file_ext not in ALLOWED_FILE_TYPES:
            return {
                'valid': False,
                'error': f'File type {file_ext} not allowed. Allowed types: {", ".join(ALLOWED_FILE_TYPES)}'
            }
        
        # Detect MIME type
        try:
            mime_type = magic.from_buffer(file_data, mime=True)
        except:
            # Fallback to filename-based detection
            mime_type, _ = mimetypes.guess_type(filename)
            if not mime_type:
                mime_type = 'application/octet-stream'
        
        return {
            'valid': True,
            'mime_type': mime_type,
            'file_ext': file_ext
        }
    
    @staticmethod
    def calculate_file_hash(file_data: bytes) -> str:
        """Calculate SHA-256 hash of file data for deduplication"""
        return hashlib.sha256(file_data).hexdigest()
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename by removing or replacing unsafe characters"""
        # Remove path separators and null bytes
        filename = filename.replace('/', '_').replace('\\', '_').replace('\0', '')
        
        # Replace other potentially problematic characters
        unsafe_chars = '<>:"|?*'
        for char in unsafe_chars:
            filename = filename.replace(char, '_')
        
        # Limit length
        if len(filename) > 255:
            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            max_name_len = 255 - len(ext) - 1 if ext else 255
            filename = name[:max_name_len] + ('.' + ext if ext else '')
        
        return filename
    
    @staticmethod
    def store_file(file_data: bytes, original_filename: str, user_id: int) -> Dict[str, Any]:
        """
        Store file in database after validation and deduplication
        Returns dict with success status and file info or error
        """
        # Validate file
        validation_result = FileStorageService.validate_file(file_data, original_filename)
        if not validation_result['valid']:
            return {'success': False, 'error': validation_result['error']}
        
        # Calculate file hash for deduplication
        file_hash = FileStorageService.calculate_file_hash(file_data)
        
        # Check if file already exists
        existing_file = get_file_by_hash(file_hash)
        if existing_file:
            # File already exists, return reference to existing file
            return {
                'success': True,
                'file_id': existing_file.file_id,
                'filename': existing_file.filename,
                'duplicate': True,
                'message': 'File already exists'
            }
        
        # Sanitize filename
        sanitized_filename = FileStorageService.sanitize_filename(original_filename)
        
        # Create file upload record
        file_upload = FileUploadModel(
            filename=sanitized_filename,
            original_filename=original_filename,
            file_data=file_data,
            file_size=len(file_data),
            mime_type=validation_result['mime_type'],
            file_hash=file_hash,
            user_id=user_id,
            is_processed=False
        )
        
        try:
            created_file = create_file_upload(file_upload)
            return {
                'success': True,
                'file_id': created_file.file_id,
                'filename': created_file.filename,
                'file_size': created_file.file_size,
                'mime_type': created_file.mime_type,
                'duplicate': False
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to store file: {str(e)}'
            }
    
    @staticmethod
    def get_file_info(file_id: int) -> Optional[Dict[str, Any]]:
        """Get file metadata by ID"""
        from app.models.database.file_upload import get_file_upload_by_id
        
        file_upload = get_file_upload_by_id(file_id)
        if not file_upload:
            return None
            
        return {
            'file_id': file_upload.file_id,
            'filename': file_upload.filename,
            'original_filename': file_upload.original_filename,
            'file_size': file_upload.file_size,
            'mime_type': file_upload.mime_type,
            'upload_date': file_upload.upload_date.isoformat() if file_upload.upload_date else None,
            'user_id': file_upload.user_id,
            'is_processed': file_upload.is_processed,
            'processing_error': file_upload.processing_error,
            'extracted_text': file_upload.extracted_text
        }
    
    @staticmethod
    def delete_file(file_id: int, user_id: int) -> Dict[str, Any]:
        """Delete file (only if owned by user)"""
        from app.models.database.file_upload import delete_file_upload
        
        success = delete_file_upload(file_id, user_id)
        if success:
            return {'success': True, 'message': 'File deleted successfully'}
        else:
            return {'success': False, 'error': 'File not found or access denied'}