#!/usr/bin/env python3
"""
Script to manage model weights for llama_3_1.

This script provides functionality to:
1. Delete all weights in the app/model_weights/llama_3_1 folder
2. Copy weights from a desktop location to the app/model_weights/llama_3_1 folder
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Optional, List, Union, Tuple
from dotenv import load_dotenv

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

models = {
    "llama_3_1" : "llama_3_1"
}

def delete_weights(weights_dir: Union[str, Path] = None) -> Tuple[bool, str]:
    """
    Delete all files in the specified weights directory.
    
    Args:
        weights_dir: Path to the weights directory. If None, defaults to app/model_weights/llama_3_1
        
    Returns:
        Tuple containing:
            - Success status (bool)
            - Message describing the result
    """
    if weights_dir is None:
        # Get the project root directory (assuming this script is in scripts/)
        root_dir = Path(__file__).parent.parent
        weights_dir = root_dir / "app" / "model_weights" / "llama_3_1"
    
    weights_dir = Path(weights_dir)
    
    if not weights_dir.exists():
        logger.warning(f"Directory {weights_dir} does not exist.")
        return False, f"Directory {weights_dir} does not exist."
    
    try:
        # Count files before deletion
        files = list(weights_dir.glob("*"))
        file_count = len(files)
        
        if file_count == 0:
            logger.info(f"No files found in {weights_dir}.")
            return True, f"No files found in {weights_dir}."
        
        # Delete all files in the directory
        for file_path in files:
            if file_path.is_file():
                file_path.unlink()
            elif file_path.is_dir():
                shutil.rmtree(file_path)
        
        logger.info(f"Successfully deleted {file_count} files/directories from {weights_dir}.")
        return True, f"Successfully deleted {file_count} files/directories from {weights_dir}."
    
    except Exception as e:
        logger.error(f"Error deleting files: {str(e)}")
        return False, f"Error deleting files: {str(e)}"

def copy_weights(source_dir: Union[str, Path], 
                 dest_dir: Optional[Union[str, Path]] = None) -> Tuple[bool, str]:
    """
    Copy model weights from source directory to destination directory.
    
    Args:
        source_dir: Path to the source directory containing model weights
        dest_dir: Path to the destination directory. If None, defaults to app/model_weights/llama_3_1
        
    Returns:
        Tuple containing:
            - Success status (bool)
            - Message describing the result
    """
    source_dir = Path(source_dir)
    
    if not source_dir.exists():
        logger.error(f"Source directory {source_dir} does not exist.")
        return False, f"Source directory {source_dir} does not exist."
    
    if dest_dir is None:
        # Get the project root directory (assuming this script is in scripts/)
        root_dir = Path(__file__).parent.parent
        dest_dir = root_dir / "app" / "model_weights" / "llama_3_1"
    
    dest_dir = Path(dest_dir)
    
    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    
    try:
        # Count files to copy
        files = list(source_dir.glob("*"))
        file_count = len(files)
        
        if file_count == 0:
            logger.warning(f"No files found in {source_dir}.")
            return False, f"No files found in {source_dir}."
        
        # Copy all files from source to destination
        for file_path in files:
            if file_path.is_file():
                shutil.copy2(file_path, dest_dir)
            elif file_path.is_dir():
                dest_subdir = dest_dir / file_path.name
                shutil.copytree(file_path, dest_subdir, dirs_exist_ok=True)
        
        logger.info(f"Successfully copied {file_count} files/directories from {source_dir} to {dest_dir}.")
        return True, f"Successfully copied {file_count} files/directories from {source_dir} to {dest_dir}."
    
    except Exception as e:
        logger.error(f"Error copying files: {str(e)}")
        return False, f"Error copying files: {str(e)}"

def copy_from_desktop() -> Tuple[bool, str]:
    """
    Copy model weights from the directory specified in LOCAL_WEIGHTS_DIR environment variable
    to app/model_weights/llama_3_1.
    
    If LOCAL_WEIGHTS_DIR is not set, falls back to desktop llama_3_1 folder.
    
    Returns:
        Tuple containing:
            - Success status (bool)
            - Message describing the result
    """
    # Try to get the path from environment variable
    local_weights_dir = os.getenv("LOCAL_WEIGHTS_DIR")
    
    if local_weights_dir:
        local_weights_path = Path(local_weights_dir)
        for model in models.keys(): 
            model_path = local_weights_path / models[model]
            if model_path.exists():
                logger.info(f"Using weights from LOCAL_WEIGHTS_DIR: {model_path}")
                return copy_weights(model_path)
            else:
                logger.warning(f"LOCAL_WEIGHTS_DIR path does not exist: {model_path}")
    
    # Fall back to desktop path if LOCAL_WEIGHTS_DIR is not set or doesn't exist
    home_dir = Path.home()
    desktop_dir = home_dir / "Desktop" / "llama_3_1"
    
    if not desktop_dir.exists():
        # Try alternative desktop path on some systems
        desktop_dir = home_dir / "OneDrive" / "Desktop" / "llama_3_1"
        
        if not desktop_dir.exists():
            logger.error(f"Could not find llama_3_1 folder on desktop.")
            return False, f"Could not find llama_3_1 folder on desktop."
    
    logger.info(f"Using weights from desktop: {desktop_dir}")
    return copy_weights(desktop_dir)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage llama_3_1 model weights")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete all weights")
    
    # Copy command
    copy_parser = subparsers.add_parser("copy", help="Copy weights from source to destination")
    copy_parser.add_argument("--source", type=str, help="Source directory (defaults to desktop/llama_3_1)")
    
    args = parser.parse_args()
    
    if args.command == "delete":
        success, message = delete_weights()
        print(message)
        exit(0 if success else 1)
    
    elif args.command == "copy":
        if args.source:
            success, message = copy_weights(args.source)
        else:
            success, message = copy_from_desktop()
        print(message)
        exit(0 if success else 1)
    
    else:
        parser.print_help()
