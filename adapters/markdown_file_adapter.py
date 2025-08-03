import os
from typing import List, Optional
from adapters.base_adapter import BaseAdapter


class MarkdownFileAdapter(BaseAdapter):
    """
    Markdown File Adapter for reading, writing, and managing markdown files
    """
    
    def __init__(self, file_root: str = ".", **kwargs):
        """
        Initialize the markdown file adapter
        
        Args:
            file_root: Root directory for file operations (default: current directory)
        """
        self.file_root = os.path.abspath(file_root)
        
        # Ensure the root directory exists
        os.makedirs(self.file_root, exist_ok=True)
    
    def enumerate_actions(self) -> List[str]:
        """Return list of available actions for this adapter"""
        return ["read_file", "write_file", "get_files"]
    
    def read_file(self, filename: str) -> str:
        """
        Read content from a markdown file
        
        Args:
            filename: Name of the file to read (relative to file_root)
            
        Returns:
            Content of the file as string
            
        Raises:
            FileNotFoundError: If file doesn't exist
            IOError: If file cannot be read
        """
        filepath = self._get_full_path(filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filename}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            raise IOError(f"Error reading file {filename}: {str(e)}")
    
    def write_file(self, filename: str, content: str) -> bool:
        """
        Write content to a markdown file
        
        Args:
            filename: Name of the file to write (relative to file_root)
            content: Content to write to the file
            
        Returns:
            True if successful, False otherwise
        """
        filepath = self._get_full_path(filename)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as file:
                file.write(content)
            return True
        except Exception as e:
            print(f"Error writing file {filename}: {str(e)}")
            return False
    
    def get_files(self, file_root: Optional[str] = None) -> List[str]:
        """
        Get list of markdown files in the specified directory
        
        Args:
            file_root: Directory to search (optional, defaults to adapter's file_root)
            
        Returns:
            List of markdown file paths relative to the search directory
        """
        search_root = file_root if file_root else self.file_root
        search_path = os.path.abspath(search_root) if file_root else self.file_root
        
        if not os.path.exists(search_path):
            return []
        
        markdown_files = []
        
        try:
            for root, dirs, files in os.walk(search_path):
                for file in files:
                    if file.lower().endswith(('.md', '.markdown')):
                        # Get relative path from search root
                        full_path = os.path.join(root, file)
                        relative_path = os.path.relpath(full_path, search_path)
                        markdown_files.append(relative_path)
            
            return sorted(markdown_files)
        except Exception as e:
            print(f"Error listing files in {search_path}: {str(e)}")
            return []
    
    def _get_full_path(self, filename: str) -> str:
        """
        Get full file path relative to file_root
        
        Args:
            filename: Relative filename
            
        Returns:
            Absolute file path
        """
        return os.path.join(self.file_root, filename)