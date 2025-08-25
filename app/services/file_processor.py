import asyncio
from typing import Dict, Any, Optional
from app.services.text_extraction import TextExtractionService
from app.models.database.file_upload import update_file_processing_status, get_file_upload_by_id

class FileProcessingService:
    """Service for processing uploaded files asynchronously"""
    
    @staticmethod
    async def process_file(file_id: int) -> Dict[str, Any]:
        """
        Process a file by extracting text content and updating the database
        Returns dict with processing results
        """
        try:
            # Get file data from database
            file_upload = get_file_upload_by_id(file_id)
            if not file_upload:
                return {
                    'success': False,
                    'error': f'File with ID {file_id} not found'
                }
            
            # Skip processing if already processed
            if file_upload.is_processed:
                return {
                    'success': True,
                    'message': 'File already processed',
                    'extracted_text': file_upload.extracted_text
                }
            
            # Extract text from file
            extraction_result = TextExtractionService.extract_text_from_file(
                file_upload.file_data,
                file_upload.mime_type,
                file_upload.filename
            )
            
            if extraction_result['success']:
                # Clean the extracted text
                cleaned_text = TextExtractionService.clean_extracted_text(
                    extraction_result['text']
                )
                
                # Update database with extracted text
                update_file_processing_status(
                    file_id=file_id,
                    extracted_text=cleaned_text,
                    processing_error=None
                )
                
                return {
                    'success': True,
                    'message': 'File processed successfully',
                    'extracted_text': cleaned_text,
                    'text_length': len(cleaned_text)
                }
            else:
                # Update database with processing error
                error_message = extraction_result.get('error', 'Unknown error during text extraction')
                update_file_processing_status(
                    file_id=file_id,
                    extracted_text=None,
                    processing_error=error_message
                )
                
                return {
                    'success': False,
                    'error': error_message
                }
                
        except Exception as e:
            error_message = f'File processing failed: {str(e)}'
            
            # Try to update database with error (but don't fail if this fails)
            try:
                update_file_processing_status(
                    file_id=file_id,
                    extracted_text=None,
                    processing_error=error_message
                )
            except:
                pass
            
            return {
                'success': False,
                'error': error_message
            }
    
    @staticmethod
    def process_file_sync(file_id: int) -> Dict[str, Any]:
        """
        Synchronous wrapper for process_file
        Use this when you need to process files without async context
        """
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(FileProcessingService.process_file(file_id))
        except RuntimeError:
            # Create new event loop if none exists
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(FileProcessingService.process_file(file_id))
            finally:
                loop.close()
    
    @staticmethod
    async def process_multiple_files(file_ids: list) -> Dict[str, Any]:
        """
        Process multiple files concurrently
        Returns dict with results for each file
        """
        if not file_ids:
            return {'success': True, 'results': {}}
        
        # Process files concurrently
        tasks = [FileProcessingService.process_file(file_id) for file_id in file_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Compile results
        file_results = {}
        success_count = 0
        error_count = 0
        
        for i, result in enumerate(results):
            file_id = file_ids[i]
            
            if isinstance(result, Exception):
                file_results[file_id] = {
                    'success': False,
                    'error': f'Processing exception: {str(result)}'
                }
                error_count += 1
            else:
                file_results[file_id] = result
                if result.get('success', False):
                    success_count += 1
                else:
                    error_count += 1
        
        return {
            'success': True,
            'results': file_results,
            'summary': {
                'total_files': len(file_ids),
                'successful': success_count,
                'failed': error_count
            }
        }
    
    @staticmethod
    def reprocess_file(file_id: int, force: bool = False) -> Dict[str, Any]:
        """
        Reprocess a file (useful if processing failed or needs updating)
        """
        if not force:
            # Check if file exists and get current status
            file_upload = get_file_upload_by_id(file_id)
            if not file_upload:
                return {
                    'success': False,
                    'error': f'File with ID {file_id} not found'
                }
            
            if file_upload.is_processed and not file_upload.processing_error:
                return {
                    'success': True,
                    'message': 'File already processed successfully. Use force=True to reprocess.',
                    'extracted_text': file_upload.extracted_text
                }
        
        # Reset processing status before reprocessing
        try:
            update_file_processing_status(
                file_id=file_id,
                extracted_text=None,
                processing_error=None
            )
            # Set is_processed back to False
            from app.models.database.database import SessionLocal
            from app.models.database.file_upload import FileUpload
            with SessionLocal() as session:
                db_file = session.query(FileUpload).filter_by(file_id=file_id).first()
                if db_file:
                    db_file.is_processed = False
                    session.commit()
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to reset processing status: {str(e)}'
            }
        
        # Process the file
        return FileProcessingService.process_file_sync(file_id)
    
    @staticmethod
    def get_processing_status(file_id: int) -> Dict[str, Any]:
        """Get the current processing status of a file"""
        file_upload = get_file_upload_by_id(file_id)
        if not file_upload:
            return {
                'success': False,
                'error': f'File with ID {file_id} not found'
            }
        
        return {
            'success': True,
            'file_id': file_upload.file_id,
            'filename': file_upload.filename,
            'is_processed': file_upload.is_processed,
            'processing_error': file_upload.processing_error,
            'has_extracted_text': bool(file_upload.extracted_text),
            'text_length': len(file_upload.extracted_text) if file_upload.extracted_text else 0
        }