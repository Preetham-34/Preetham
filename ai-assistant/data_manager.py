"""Data manager for loading sample datasets with robust file path handling."""

import os
import json
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class DataManager:
    """Handles loading and managing sample data for the LMS AI system."""
    
    def __init__(self):
        """Initialize data manager with proper path handling."""
        # Get the directory where this script is located
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Define data file paths - try both with and without data folder
        self.data_folder = os.path.join(self.base_dir, "data")
        
        # Primary paths (with data folder)
        self.courses_path = os.path.join(self.data_folder, "courses.json")
        self.users_path = os.path.join(self.data_folder, "users.json")
        self.faq_path = os.path.join(self.data_folder, "faq.json")
        
        # Fallback paths (without data folder)
        self.courses_path_fallback = os.path.join(self.base_dir, "courses.json")
        self.users_path_fallback = os.path.join(self.base_dir, "users.json")
        self.faq_path_fallback = os.path.join(self.base_dir, "faq.json")
        
        logger.info(f"DataManager initialized with base directory: {self.base_dir}")
        self._log_debug_info()
    
    def _log_debug_info(self):
        """Log debug information about file paths and existence."""
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Script base directory: {self.base_dir}")
        logger.info(f"Files in base directory: {os.listdir(self.base_dir)}")
        
        # Check data folder
        if os.path.exists(self.data_folder):
            logger.info(f"Data folder exists: {self.data_folder}")
            logger.info(f"Files in data folder: {os.listdir(self.data_folder)}")
        else:
            logger.info(f"Data folder does not exist: {self.data_folder}")
        
        # Check file existence
        for file_type, (primary, fallback) in [
            ("courses", (self.courses_path, self.courses_path_fallback)),
            ("users", (self.users_path, self.users_path_fallback)),
            ("faq", (self.faq_path, self.faq_path_fallback))
        ]:
            if os.path.exists(primary):
                logger.info(f"{file_type}.json found at primary path: {primary}")
            elif os.path.exists(fallback):
                logger.info(f"{file_type}.json found at fallback path: {fallback}")
            else:
                logger.warning(f"{file_type}.json not found at either path")
    
    def _get_file_path(self, primary_path: str, fallback_path: str) -> Optional[str]:
        """Get the correct file path, trying primary first then fallback."""
        if os.path.exists(primary_path):
            return primary_path
        elif os.path.exists(fallback_path):
            return fallback_path
        else:
            return None
    
    def load_json_file(self, primary_path: str, fallback_path: str, file_type: str = "data") -> List[Dict]:
        """Load JSON file with error handling and path fallback."""
        file_path = self._get_file_path(primary_path, fallback_path)
        
        if file_path is None:
            logger.error(f"{file_type} file not found at either path:")
            logger.error(f"  Primary: {primary_path}")
            logger.error(f"  Fallback: {fallback_path}")
            return []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            logger.info(f"Successfully loaded {len(data)} {file_type} records from {file_path}")
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {file_type} file {file_path}: {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to load {file_type} file {file_path}: {e}")
            return []
    
    def load_all_data(self) -> Dict[str, List[Dict]]:
        """Load all sample data files."""
        return {
            'courses': self.load_json_file(self.courses_path, self.courses_path_fallback, "courses"),
            'users': self.load_json_file(self.users_path, self.users_path_fallback, "users"),
            'faq': self.load_json_file(self.faq_path, self.faq_path_fallback, "FAQ")
        }
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get information about data files."""
        return {
            'base_directory': self.base_dir,
            'data_folder': self.data_folder,
            'data_folder_exists': os.path.exists(self.data_folder),
            'courses_path_primary': self.courses_path,
            'courses_path_fallback': self.courses_path_fallback,
            'users_path_primary': self.users_path,
            'users_path_fallback': self.users_path_fallback,
            'faq_path_primary': self.faq_path,
            'faq_path_fallback': self.faq_path_fallback,
            'courses_exists': self._get_file_path(self.courses_path, self.courses_path_fallback) is not None,
            'users_exists': self._get_file_path(self.users_path, self.users_path_fallback) is not None,
            'faq_exists': self._get_file_path(self.faq_path, self.faq_path_fallback) is not None
        }