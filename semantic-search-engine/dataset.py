"""Dataset loading and management utilities - Production Ready with Debug Logging."""

import json
import logging
import os
from typing import List, Dict, Tuple
import os
print('[DEBUG] Working directory:', os.getcwd())
print('[DEBUG] Files here:', os.listdir('.'))
print('[DEBUG] File data.json exists?', os.path.exists('data.json'))

logger = logging.getLogger(__name__)

class DatasetManager:
    """Manages document datasets with validation and error handling."""

    def __init__(self, data_file_path: str = "data.json"):
        self.data_file_path = data_file_path
        self.sample_data = self._get_sample_documents()

    def _get_sample_documents(self) -> List[Dict]:
        """Default sample dataset in case no file is found."""
        return [
            {
                "id": "doc_001",
                "title": "Python Programming Fundamentals",
                "content": "Python is a versatile, high-level programming language known for its simplicity and readability. It supports multiple programming paradigms including object-oriented, functional, and procedural programming. Python is widely used in web development, data science, artificial intelligence, automation, and scientific computing. Its extensive standard library and active community make it an excellent choice for both beginners and experienced developers.",
                "category": "Programming",
                "tags": ["python", "programming", "basics", "syntax", "development"]
            },
            {
                "id": "doc_002",
                "title": "Machine Learning Introduction",
                "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It involves algorithms that can identify patterns in data and make predictions or decisions. Common applications include recommendation systems, image recognition, natural language processing, and predictive analytics. Machine learning algorithms can be supervised, unsupervised, or reinforcement-based.",
                "category": "AI/ML",
                "tags": ["machine-learning", "ai", "algorithms", "data-science", "patterns"]
            },
            {
                "id": "doc_003",
                "title": "Vector Databases Explained",
                "content": "Vector databases are specialized database systems designed to store, index, and search high-dimensional vector embeddings efficiently. They enable semantic search by finding similar vectors based on mathematical distance metrics like cosine similarity. Vector databases are essential for building RAG applications, recommendation systems, and similarity search engines. Popular vector databases include Pinecone, Weaviate, Chroma, and Qdrant.",
                "category": "Database",
                "tags": ["vector-database", "embeddings", "semantic-search", "similarity", "rag"]
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
                print(f"[DATASET] Successfully loaded {len(data)} documents from {file_path}")
            else:
                data = self.sample_data
                logger.info(f"File {file_path} not found. Using sample dataset with {len(data)} documents")
                print(f"[DATASET] File not found. Using {len(data)} sample documents")
        except Exception as e:
            logger.warning(f"Could not load from {file_path}: {e}. Using sample data.")
            print(f"[DATASET] Error loading file: {e}. Using sample data.")
            data = self.sample_data

        return self._extract_components(data)

    def _load_from_file(self, file_path: str) -> List[Dict]:
        """Load and validate JSON dataset file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Validate data structure
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
            # Extract fields with proper variable scoping
            doc_id = doc.get("id", f"doc_{i:06d}")
            content = doc.get("content", "")
            title = doc.get("title", f"Document {doc_id}")

            ids.append(doc_id)
            documents.append(content)
            titles.append(title)

            # Prepare metadata (everything except content to avoid duplication)
            metadata = {k: v for k, v in doc.items() if k != "content"}
            if "title" not in metadata:
                metadata["title"] = title
            metadata_list.append(metadata)

        logger.info(f"Extracted {len(ids)} documents for processing")
        print(f"[DATASET] Extracted {len(ids)} documents with {len(set(doc.get('category', 'Unknown') for doc in data))} categories")
        
        # Print sample titles for verification
        sample_titles = titles[:5] if len(titles) >= 5 else titles
        print(f"[DATASET] Sample titles: {sample_titles}")
        
        return ids, documents, titles, metadata_list

# Global instance for import convenience
dataset_manager = DatasetManager()
