# File Upload Feature Implementation Plan

## Overview
This plan details the implementation of a file upload system for the Geist application, enabling users to upload, store, retrieve, and reference files within chat contexts. The feature will support PDF, text files, and other arbitrary file types.

## Architecture Analysis
Based on the current codebase structure:
- **Backend**: FastAPI with SQLAlchemy ORM and Alembic migrations
- **Database**: PostgreSQL with existing models in `app/models/database/`
- **Frontend**: React with TypeScript, using existing navigation patterns
- **API Structure**: RESTful endpoints under `/api/v1/` with user authentication
- **File Storage**: Files will be stored as BLOBs directly in the database

## Implementation Plan

### Phase 1: Database Layer

#### 1.1 Create File Upload Model
**Location**: `app/models/database/file_upload.py`

```python
class FileUpload(Base):
    __tablename__ = "file_uploads"
    
    file_id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    file_data = Column(LargeBinary, nullable=False)  # Store file as BLOB
    file_size = Column(Integer, nullable=False)  # Size in bytes
    mime_type = Column(String, nullable=False)
    file_hash = Column(String, nullable=False)  # SHA-256 hash for deduplication
    upload_date = Column(DateTime, default=datetime.utcnow)
    user_id = Column(Integer, ForeignKey('geist_user.user_id'), nullable=False)
    
    # Extracted content for searchability
    extracted_text = Column(String)  # For PDFs and text files
    is_processed = Column(Boolean, default=False)
    processing_error = Column(String)  # Store any processing errors
    
    # Relationships
    user = relationship("GeistUser", backref="uploaded_files")
```

#### 1.2 Create Database Migration
**Location**: `migrations/versions/{timestamp}_add_file_upload_table.py`

Create migration to add the file_uploads table with proper indexes on user_id, file_hash, and upload_date.

#### 1.3 File Storage Service
**Location**: `app/services/file_storage.py`

Implement file storage service:
- File deduplication using SHA-256 hashes
- File type validation 
- Basic file sanitization

### Phase 2: File Processing Services

#### 2.1 Text Extraction Service
**Location**: `app/services/text_extraction.py`

Implement text extraction for various file types:
- **PDF files**: Use PyPDF2 or pdfplumber for text extraction
- **Text files**: Direct reading with encoding detection
- **Office documents**: Support for .docx, .xlsx using python-docx, openpyxl

#### 2.2 File Processing Pipeline
**Location**: `app/services/file_processor.py`

Processing service to:
- Extract text content from uploaded files
- Generate searchable metadata
- Update database with processing results
- Handle processing errors gracefully

### Phase 3: API Endpoints

#### 3.1 File Upload Endpoint
**Location**: `app/api/v1/endpoints/files.py`

```python
@router.post("/upload", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
) -> FileUploadResponse:
    """Upload a file and store it in database"""
```

Features:
- File validation (type, size limits)
- Duplicate detection using file hashes
- Store file as BLOB in database
- Metadata extraction and storage
- Text extraction processing

#### 3.2 File Retrieval Endpoints
**Location**: `app/api/v1/endpoints/files.py`

```python
@router.get("/", response_model=List[FileUploadResponse])
async def list_files(
    skip: int = 0,
    limit: int = 100,
    search: Optional[str] = None,
    file_type: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
) -> List[FileUploadResponse]:
    """List uploaded files with filtering and pagination"""

@router.get("/{file_id}", response_model=FileUploadResponse)
async def get_file_metadata(file_id: int, ...):
    """Get file metadata by ID"""

@router.get("/{file_id}/download")
async def download_file(file_id: int, ...):
    """Download the original file"""

@router.get("/{file_id}/content")
async def get_file_content(file_id: int, ...):
    """Get extracted text content"""

@router.delete("/{file_id}")
async def delete_file(file_id: int, ...):
    """Delete a file and its metadata"""
```

#### 3.3 Integration with Main API
**Location**: `app/main.py`

Register the files router:
```python
app.include_router(files_router, prefix="/api/v1/files", tags=["files"])
```

### Phase 4: Frontend Implementation

#### 4.1 Files Tab Navigation
**Location**: `client/geist/src/App.tsx`

Add new navigation element:
```javascript
{
  name: 'Files',
  link: '/files',
  svg: 'M... [file icon SVG path]'
}
```

#### 4.2 File Upload Component
**Location**: `client/geist/src/Components/FileUpload.tsx`

Features:
- Drag-and-drop file upload interface
- File type and size validation
- Upload progress indicators
- Error handling and user feedback
- Support for multiple file selection

#### 4.3 File Management Page
**Location**: `client/geist/src/Files.tsx`

Complete file management interface:
- File listing with search and filtering
- File preview capabilities (text content)
- File metadata display (size, upload date, type)
- Download and delete functionality
- Pagination for large file lists

#### 4.4 File Selection Modal
**Location**: `client/geist/src/Components/FileSelectionModal.tsx`

Modal component for selecting files in chat contexts:
- Searchable file list
- File preview
- Multiple selection support
- Integration with @ referencing system

### Phase 5: Chat Integration

#### 5.1 @ File Referencing Parser
**Location**: `client/geist/src/Utils/fileReferenceParser.ts`

Implement parser for @ file references:
- Pattern matching for @filename or @file:id syntax
- Autocomplete suggestions based on uploaded files
- File content injection into chat context

#### 5.2 Chat Context Enhancement
**Location**: Modify existing chat components

Enhance chat functionality:
- File reference detection in messages
- Automatic file content inclusion in API calls
- Visual indicators for messages with file attachments
- File context management (character limits, relevance scoring)

#### 5.3 File Context API Integration
**Location**: `client/geist/src/Hooks/useFileContext.tsx`

Custom hook for managing file context in chats:
- File content retrieval
- Context size management

### Phase 6: Enhanced Features

#### 6.1 File Search and Indexing
- Full-text search across uploaded files
- Advanced filtering and sorting options

## Dependencies Installation

### Backend Dependencies
```bash
# Install via conda
conda install -c conda-forge pypdf2
conda install -c conda-forge pdfplumber
conda install -c conda-forge python-docx
conda install -c conda-forge openpyxl
conda install -c conda-forge python-magic
conda install -c conda-forge aiofiles
```

### Frontend Dependencies  
```bash
# Install via npm
npm install react-dropzone
npm install react-query
npm install @types/file-saver
npm install file-saver
```

## Configuration Variables

```python
# File upload settings
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_FILE_TYPES = ['.pdf', '.txt', '.docx', '.xlsx', '.csv', '.md']
MAX_FILES_PER_USER = 1000
```

## Database Schema Changes

```sql
-- New table
CREATE TABLE file_uploads (
    file_id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    original_filename VARCHAR(255) NOT NULL,
    file_data BYTEA NOT NULL,
    file_size INTEGER NOT NULL,
    mime_type VARCHAR(100) NOT NULL,
    file_hash VARCHAR(64) NOT NULL UNIQUE,
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_id INTEGER REFERENCES geist_user(user_id) ON DELETE CASCADE,
    extracted_text TEXT,
    is_processed BOOLEAN DEFAULT FALSE,
    processing_error TEXT
);

-- Indexes for performance
CREATE INDEX idx_file_uploads_user_id ON file_uploads(user_id);
CREATE INDEX idx_file_uploads_upload_date ON file_uploads(upload_date);
CREATE INDEX idx_file_uploads_filename ON file_uploads(filename);
CREATE INDEX idx_file_uploads_hash ON file_uploads(file_hash);
```