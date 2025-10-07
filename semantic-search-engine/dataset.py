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
        """Returns sample documents in case data file is missing."""
        return [
            {
                "id": "doc_001",
                "title": "Python Programming Fundamentals",
                "content": "Python is a versatile...",
                "category": "Programming",
                "tags": ["python", "programming", "basics"]
            },
            {
                "id": "doc_002",
                "title": "Machine Learning Introduction",
                "content": "Machine learning is a subset...",
                "category": "AI/ML",
                "tags": ["machine-learning", "ai", "algorithms"]
            }
            # Add more sample documents as needed
        ]

    def load_dataset(self, file_path: str = None) -> Tuple[List[str], List[str], List[str], List[Dict]]:
        """Load dataset and return ids, documents, titles, and metadata."""
        file_path = file_path or self.data_file_path

        try:
            if os.path.exists(file_path):
                data = self._load_from_file(file_path)
                logger.info(f"Loaded {len(data)} documents from {file_path}")
            else:
                data = self.sample_data
                logger.info(f"Using default sample dataset of {len(data)} documents.")
        except Exception as e:
            logger.warning(f"Error reading dataset file: {e}. Using sample data.")
            data = self.sample_data

        return self._extract_components(data)

    def _load_from_file(self, file_path: str) -> List[Dict]:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Dataset must be a list of documents")
        for i, doc in enumerate(data):
            if not isinstance(doc, dict):
                raise ValueError(f"Document {i} is not a JSON object")
            if "content" not in doc:
                raise ValueError(f"Document {i} missing 'content' field")
        return data

    def _extract_components(self, data: List[Dict]) -> Tuple[List[str], List[str], List[str], List[Dict]]:
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
            meta = {k: v for k, v in doc.items() if k != "content"}
            if "title" not in meta:
                meta["title"] = title
            metadata_list.append(meta)
        logger.info(f"Extracted {len(ids)} documents from dataset")
        return ids, documents, titles, metadata_list


# Global singleton instance
dataset_manager = DatasetManager()
