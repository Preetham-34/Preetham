"""Dataset loading and management utilities."""

import json
import logging
import os
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

class DatasetManager:
    """Manages document datasets with validation and error handling."""

    def __init__(self, data_file_path: str = "data.json"):
        self.data_file_path = data_file_path
        self.sample_data = self._get_sample_documents()

    def _get_sample_documents(self) -> List[Dict]:
        """Minimal sample dataset in case no file is found."""
        return [
            {
                "id": "doc_001",
                "title": "Python Programming Fundamentals",
                "content": "Python is a versatile, high-level programming language known for its simplicity and readability...",
                "category": "Programming",
                "tags": ["python", "programming", "basics"]
            },
            {
                "id": "doc_002",
                "title": "Machine Learning Introduction",
                "content": "Machine learning is a subset of artificial intelligence enabling computers to learn from experience...",
                "category": "AI/ML",
                "tags": ["machine-learning", "ai", "algorithms"]
            }
        ]

    def load_dataset(self, file_path: str = None) -> Tuple[List[str], List[str], List[str], List[Dict]]:
        """
        Load dataset from file or use sample data.

        Args:
            file_path: Path to JSON dataset file

        Returns:
            Tuple of (ids, documents, titles, metadata_list)
        """
        file_path = file_path or self.data_file_path

        try:
            if os.path.exists(file_path):
                data = self._load_from_file(file_path)
                logger.info(f"Loaded {len(data)} documents from {file_path}")
            else:
                data = self.sample_data
                logger.info(f"Using sample dataset with {len(data)} documents")
        except Exception as e:
            logger.warning(f"Could not load from {file_path}: {e}. Using sample data.")
            data = self.sample_data

        return self._extract_components(data)

    def _load_from_file(self, file_path: str) -> List[Dict]:
        """Load and validate JSON dataset file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Dataset must be a list of documents")

        for i, doc in enumerate(data):
            if not isinstance(doc, dict):
                raise ValueError(f"Document {i} must be a dictionary")
            if "content" not in doc:
                raise ValueError(f"Document {i} missing required 'content' field")

        return data

    def _extract_components(self, data: List[Dict]) -> Tuple[List[str], List[str], List[str], List[Dict]]:
        """Extract and organize dataset components with proper variable scoping."""
        ids = []
        documents = []
        titles = []
        metadata_list = []

        for i, doc in enumerate(data):
            doc_id = doc.get("id", f"doc_{i:06d}")
            content = doc.get("content", "")
            title = doc.get("title", f"Document {doc_id}")

            ids.append(doc_id)
            documents.append(content)
            titles.append(title)

            # Metadata excludes content to avoid duplication
            metadata = {k: v for k, v in doc.items() if k != "content"}
            if "title" not in metadata:
                metadata["title"] = title
            metadata_list.append(metadata)

        logger.info(f"Extracted {len(ids)} documents for processing")
        return ids, documents, titles, metadata_list


# Global instance for import convenience
dataset_manager = DatasetManager()
Replace your existing dataset.py with this code, then restart your app. It will fix the title variable error and properly load and extract dataset components.Here is the complete, corrected dataset.py file that fixes the "name 'title' is not defined" issue by properly defining title and content variables inside the loop and returning all four expected values (ids, documents, titles, metadata_list):

python
"""Dataset loading and management utilities."""

import json
import logging
import os
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

class DatasetManager:
    """Manages document datasets with validation and error handling."""

    def __init__(self, data_file_path: str = "data.json"):
        self.data_file_path = data_file_path
        self.sample_data = self._get_sample_documents()

    def _get_sample_documents(self) -> List[Dict]:
        """Minimal sample dataset in case no file is found."""
        return [
            {
                "id": "doc_001",
                "title": "Python Programming Fundamentals",
                "content": "Python is a versatile, high-level programming language known for its simplicity and readability...",
                "category": "Programming",
                "tags": ["python", "programming", "basics"]
            },
            {
                "id": "doc_002",
                "title": "Machine Learning Introduction",
                "content": "Machine learning is a subset of artificial intelligence enabling computers to learn from experience...",
                "category": "AI/ML",
                "tags": ["machine-learning", "ai", "algorithms"]
            }
        ]

    def load_dataset(self, file_path: str = None) -> Tuple[List[str], List[str], List[str], List[Dict]]:
        """
        Load dataset from file or use sample data.

        Args:
            file_path: Path to JSON dataset file

        Returns:
            Tuple of (ids, documents, titles, metadata_list)
        """
        file_path = file_path or self.data_file_path

        try:
            if os.path.exists(file_path):
                data = self._load_from_file(file_path)
                logger.info(f"Loaded {len(data)} documents from {file_path}")
            else:
                data = self.sample_data
                logger.info(f"Using sample dataset with {len(data)} documents")
        except Exception as e:
            logger.warning(f"Could not load from {file_path}: {e}. Using sample data.")
            data = self.sample_data

        return self._extract_components(data)

    def _load_from_file(self, file_path: str) -> List[Dict]:
        """Load and validate JSON dataset file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Dataset must be a list of documents")

        for i, doc in enumerate(data):
            if not isinstance(doc, dict):
                raise ValueError(f"Document {i} must be a dictionary")
            if "content" not in doc:
                raise ValueError(f"Document {i} missing required 'content' field")

        return data

    def _extract_components(self, data: List[Dict]) -> Tuple[List[str], List[str], List[str], List[Dict]]:
        """Extract and organize dataset components with proper variable scoping."""
        ids = []
        documents = []
        titles = []
        metadata_list = []

        for i, doc in enumerate(data):
            doc_id = doc.get("id", f"doc_{i:06d}")
            content = doc.get("content", "")
            title = doc.get("title", f"Document {doc_id}")

            ids.append(doc_id)
            documents.append(content)
            titles.append(title)

            # Metadata excludes content to avoid duplication
            metadata = {k: v for k, v in doc.items() if k != "content"}
            if "title" not in metadata:
                metadata["title"] = title
            metadata_list.append(metadata)

        logger.info(f"Extracted {len(ids)} documents for processing")
        return ids, documents, titles, metadata_list

# Global instance for import convenience
dataset_manager = DatasetManager()
