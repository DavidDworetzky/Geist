import pytest
import os
import tempfile
import shutil
from unittest.mock import patch, mock_open
from adapters.markdown_file_adapter import MarkdownFileAdapter


class TestMarkdownFileAdapter:
    
    def setup_method(self):
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.adapter = MarkdownFileAdapter(file_root=self.temp_dir)
    
    def teardown_method(self):
        # Clean up temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_init_creates_root_directory(self):
        """Test that initialization creates the root directory if it doesn't exist"""
        new_temp_dir = os.path.join(tempfile.gettempdir(), "test_markdown_adapter")
        if os.path.exists(new_temp_dir):
            shutil.rmtree(new_temp_dir)
        
        adapter = MarkdownFileAdapter(file_root=new_temp_dir)
        assert os.path.exists(new_temp_dir)
        assert adapter.file_root == os.path.abspath(new_temp_dir)
        
        # Cleanup
        shutil.rmtree(new_temp_dir)
    
    def test_init_default_root(self):
        """Test initialization with default root directory"""
        adapter = MarkdownFileAdapter()
        assert adapter.file_root == os.path.abspath(".")
    
    def test_enumerate_actions(self):
        """Test that enumerate_actions returns all expected actions"""
        actions = self.adapter.enumerate_actions()
        expected_actions = ["read_file", "write_file", "get_files"]
        
        assert isinstance(actions, list)
        assert len(actions) == 3
        for action in expected_actions:
            assert action in actions
    
    def test_read_file_success(self):
        """Test successful file reading"""
        content = "# Test Markdown\n\nThis is a test file."
        filename = "test.md"
        filepath = os.path.join(self.temp_dir, filename)
        
        # Create test file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        result = self.adapter.read_file(filename)
        assert result == content
    
    def test_read_file_not_found(self):
        """Test reading non-existent file raises FileNotFoundError"""
        with pytest.raises(FileNotFoundError) as exc_info:
            self.adapter.read_file("nonexistent.md")
        
        assert "File not found: nonexistent.md" in str(exc_info.value)
    
    def test_read_file_subdirectory(self):
        """Test reading file from subdirectory"""
        content = "# Subdirectory Test"
        subdir = "docs"
        filename = "sub.md"
        
        # Create subdirectory and file
        os.makedirs(os.path.join(self.temp_dir, subdir))
        filepath = os.path.join(self.temp_dir, subdir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        result = self.adapter.read_file(f"{subdir}/{filename}")
        assert result == content
    
    def test_read_file_io_error(self):
        """Test IOError handling during file read"""
        # Create a file first
        filename = "test.md"
        filepath = os.path.join(self.temp_dir, filename)
        with open(filepath, 'w') as f:
            f.write("test")
        
        # Now mock open to raise IOError only for the read operation
        with patch("builtins.open", side_effect=IOError("Permission denied")):
            with pytest.raises(IOError) as exc_info:
                self.adapter.read_file(filename)
            
            assert "Error reading file test.md" in str(exc_info.value)
    
    def test_write_file_success(self):
        """Test successful file writing"""
        content = "# New Markdown File\n\nThis is new content."
        filename = "new_file.md"
        
        result = self.adapter.write_file(filename, content)
        
        assert result is True
        
        # Verify file was created with correct content
        filepath = os.path.join(self.temp_dir, filename)
        assert os.path.exists(filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            assert f.read() == content
    
    def test_write_file_creates_subdirectory(self):
        """Test that write_file creates necessary subdirectories"""
        content = "# Subdirectory File"
        filename = "docs/subdocs/deep.md"
        
        result = self.adapter.write_file(filename, content)
        
        assert result is True
        
        # Verify file and directories were created
        filepath = os.path.join(self.temp_dir, filename)
        assert os.path.exists(filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            assert f.read() == content
    
    def test_write_file_overwrite_existing(self):
        """Test overwriting existing file"""
        filename = "existing.md"
        original_content = "Original content"
        new_content = "New content"
        
        # Create initial file
        self.adapter.write_file(filename, original_content)
        
        # Overwrite
        result = self.adapter.write_file(filename, new_content)
        
        assert result is True
        
        # Verify content was updated
        filepath = os.path.join(self.temp_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            assert f.read() == new_content
    
    @patch("builtins.open", side_effect=IOError("Permission denied"))
    def test_write_file_io_error(self, mock_file):
        """Test IOError handling during file write"""
        with patch("os.makedirs"):  # Mock makedirs to avoid directory creation issues
            result = self.adapter.write_file("test.md", "content")
            assert result is False
    
    def test_get_files_empty_directory(self):
        """Test get_files with empty directory"""
        result = self.adapter.get_files()
        assert result == []
    
    def test_get_files_markdown_extensions(self):
        """Test get_files finds files with both .md and .markdown extensions"""
        # Create test files
        files_to_create = ["test1.md", "test2.markdown", "test3.MD", "test4.txt", "test5.py"]
        for filename in files_to_create:
            filepath = os.path.join(self.temp_dir, filename)
            with open(filepath, 'w') as f:
                f.write("test content")
        
        result = self.adapter.get_files()
        
        # Should only return markdown files, sorted
        expected = ["test1.md", "test2.markdown", "test3.MD"]
        assert sorted(result) == sorted(expected)
    
    def test_get_files_with_subdirectories(self):
        """Test get_files recursively finds markdown files in subdirectories"""
        # Create files in various subdirectories
        files_to_create = [
            "root.md",
            "docs/doc1.md", 
            "docs/subdocs/doc2.markdown",
            "other/other.md"
        ]
        
        for filename in files_to_create:
            filepath = os.path.join(self.temp_dir, filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                f.write("test content")
        
        result = self.adapter.get_files()
        
        # All files should be found with relative paths
        expected = ["root.md", "docs/doc1.md", "docs/subdocs/doc2.markdown", "other/other.md"]
        assert sorted(result) == sorted(expected)
    
    def test_get_files_custom_root(self):
        """Test get_files with custom file_root parameter"""
        # Create a separate directory
        custom_dir = os.path.join(self.temp_dir, "custom")
        os.makedirs(custom_dir)
        
        # Create file in custom directory
        filepath = os.path.join(custom_dir, "custom.md")
        with open(filepath, 'w') as f:
            f.write("custom content")
        
        result = self.adapter.get_files(file_root=custom_dir)
        
        assert result == ["custom.md"]
    
    def test_get_files_nonexistent_directory(self):
        """Test get_files with non-existent directory"""
        nonexistent_path = "/path/that/does/not/exist"
        result = self.adapter.get_files(file_root=nonexistent_path)
        assert result == []
    
    @patch("os.walk", side_effect=OSError("Permission denied"))
    def test_get_files_os_error(self, mock_walk):
        """Test get_files handles OS errors gracefully"""
        result = self.adapter.get_files()
        assert result == []
    
    def test_get_full_path(self):
        """Test _get_full_path method"""
        filename = "test.md"
        expected_path = os.path.join(self.temp_dir, filename)
        
        result = self.adapter._get_full_path(filename)
        assert result == expected_path
    
    def test_get_full_path_subdirectory(self):
        """Test _get_full_path with subdirectory"""
        filename = "docs/subdir/test.md"
        expected_path = os.path.join(self.temp_dir, filename)
        
        result = self.adapter._get_full_path(filename)
        assert result == expected_path
    
    def test_integration_workflow(self):
        """Test complete workflow: write, read, list files"""
        # Write multiple files
        files_data = {
            "readme.md": "# Project README\n\nThis is the main readme.",
            "docs/api.md": "# API Documentation\n\nAPI endpoints...",
            "docs/guide.markdown": "# User Guide\n\nHow to use..."
        }
        
        # Write all files
        for filename, content in files_data.items():
            result = self.adapter.write_file(filename, content)
            assert result is True
        
        # Verify all files can be read
        for filename, expected_content in files_data.items():
            content = self.adapter.read_file(filename)
            assert content == expected_content
        
        # List all files
        all_files = self.adapter.get_files()
        expected_files = list(files_data.keys())
        assert sorted(all_files) == sorted(expected_files)
        
        # Verify enumerate_actions still works
        actions = self.adapter.enumerate_actions()
        assert len(actions) == 3