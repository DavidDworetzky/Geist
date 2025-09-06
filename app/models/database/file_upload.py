import datetime
from sqlalchemy import Column, Integer, String, ForeignKey, LargeBinary, DateTime, Boolean, Text
from sqlalchemy.orm import relationship
from app.models.database.database import Base
from app.models.database.database import SessionLocal
from dataclasses import dataclass
from typing import Optional, List

class FileUpload(Base):
    __tablename__ = "file_uploads"
    
    file_id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    file_data = Column(LargeBinary, nullable=False)  # Store file as BLOB
    file_size = Column(Integer, nullable=False)  # Size in bytes
    mime_type = Column(String, nullable=False)
    file_hash = Column(String, nullable=False)  # SHA-256 hash for deduplication
    upload_date = Column(DateTime, default=datetime.datetime.utcnow)
    user_id = Column(Integer, ForeignKey('geist_user.user_id'), nullable=False)
    
    # Extracted content for searchability
    extracted_text = Column(Text)  # For PDFs and text files
    is_processed = Column(Boolean, default=False)
    processing_error = Column(String)  # Store any processing errors
    
    # Relationships
    user = relationship("GeistUser", backref="uploaded_files")

@dataclass
class FileUploadModel:
    file_id: Optional[int] = None
    filename: str = ""
    original_filename: str = ""
    file_data: bytes = b""
    file_size: int = 0
    mime_type: str = ""
    file_hash: str = ""
    upload_date: Optional[datetime.datetime] = None
    user_id: int = 0
    extracted_text: Optional[str] = None
    is_processed: bool = False
    processing_error: Optional[str] = None

@dataclass
class FileUploadResponse:
    file_id: int
    filename: str
    original_filename: str
    file_size: int
    mime_type: str
    upload_date: datetime.datetime
    user_id: int
    is_processed: bool
    processing_error: Optional[str] = None

def create_file_upload(file_upload: FileUploadModel) -> FileUploadModel:
    """Create a new file upload record"""
    with SessionLocal() as session:
        db_file = FileUpload(
            filename=file_upload.filename,
            original_filename=file_upload.original_filename,
            file_data=file_upload.file_data,
            file_size=file_upload.file_size,
            mime_type=file_upload.mime_type,
            file_hash=file_upload.file_hash,
            user_id=file_upload.user_id,
            extracted_text=file_upload.extracted_text,
            is_processed=file_upload.is_processed,
            processing_error=file_upload.processing_error
        )
        session.add(db_file)
        session.commit()
        session.refresh(db_file)
        
        return FileUploadModel(
            file_id=db_file.file_id,
            filename=db_file.filename,
            original_filename=db_file.original_filename,
            file_data=db_file.file_data,
            file_size=db_file.file_size,
            mime_type=db_file.mime_type,
            file_hash=db_file.file_hash,
            upload_date=db_file.upload_date,
            user_id=db_file.user_id,
            extracted_text=db_file.extracted_text,
            is_processed=db_file.is_processed,
            processing_error=db_file.processing_error
        )

def get_file_upload_by_id(file_id: int) -> Optional[FileUploadModel]:
    """Get file upload by ID"""
    with SessionLocal() as session:
        db_file = session.query(FileUpload).filter_by(file_id=file_id).first()
        if not db_file:
            return None
            
        return FileUploadModel(
            file_id=db_file.file_id,
            filename=db_file.filename,
            original_filename=db_file.original_filename,
            file_data=db_file.file_data,
            file_size=db_file.file_size,
            mime_type=db_file.mime_type,
            file_hash=db_file.file_hash,
            upload_date=db_file.upload_date,
            user_id=db_file.user_id,
            extracted_text=db_file.extracted_text,
            is_processed=db_file.is_processed,
            processing_error=db_file.processing_error
        )

def get_files_by_user(user_id: int, skip: int = 0, limit: int = 100) -> List[FileUploadResponse]:
    """Get files uploaded by a specific user"""
    with SessionLocal() as session:
        files = session.query(FileUpload).filter_by(user_id=user_id).offset(skip).limit(limit).all()
        
        return [FileUploadResponse(
            file_id=f.file_id,
            filename=f.filename,
            original_filename=f.original_filename,
            file_size=f.file_size,
            mime_type=f.mime_type,
            upload_date=f.upload_date,
            user_id=f.user_id,
            is_processed=f.is_processed,
            processing_error=f.processing_error
        ) for f in files]

def get_file_by_hash(file_hash: str) -> Optional[FileUploadModel]:
    """Check if file with given hash already exists"""
    with SessionLocal() as session:
        db_file = session.query(FileUpload).filter_by(file_hash=file_hash).first()
        if not db_file:
            return None
            
        return FileUploadModel(
            file_id=db_file.file_id,
            filename=db_file.filename,
            original_filename=db_file.original_filename,
            file_data=db_file.file_data,
            file_size=db_file.file_size,
            mime_type=db_file.mime_type,
            file_hash=db_file.file_hash,
            upload_date=db_file.upload_date,
            user_id=db_file.user_id,
            extracted_text=db_file.extracted_text,
            is_processed=db_file.is_processed,
            processing_error=db_file.processing_error
        )

def delete_file_upload(file_id: int, user_id: int) -> bool:
    """Delete file upload (only if owned by user)"""
    with SessionLocal() as session:
        db_file = session.query(FileUpload).filter_by(file_id=file_id, user_id=user_id).first()
        if not db_file:
            return False
            
        session.delete(db_file)
        session.commit()
        return True

def update_file_processing_status(file_id: int, extracted_text: str = None, processing_error: str = None):
    """Update file processing status and extracted text"""
    with SessionLocal() as session:
        db_file = session.query(FileUpload).filter_by(file_id=file_id).first()
        if db_file:
            db_file.extracted_text = extracted_text
            db_file.is_processed = True
            db_file.processing_error = processing_error
            session.commit()