"""Dataset loading and management utilities - COMPLETELY FIXED VERSION."""

import json
import logging
import os
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

class DatasetManager:
    """Manages document datasets with validation and error handling."""
    
    def __init__(self, data_file_path: str = "data.json"):
        self.data_file_path = data_file_path
        self.sample_data = self._get_sample_documents()
    
    def _get_sample_documents(self) -> List[Dict]:
        """Default sample dataset covering the required topics."""
        return [
            {
                "id": "doc_001",
                "title": "Python Programming Fundamentals",
                "content": "Python is a versatile, high-level programming language known for its simplicity and readability. It supports multiple programming paradigms including object-oriented, functional, and procedural programming. Python is widely used in web development, data science, artificial intelligence, automation, and scientific computing. Its extensive standard library and active community make it an excellent choice for both beginners and experienced developers.",
                "category": "Programming",
                "tags": ["python", "programming", "basics", "development"]
            },
            {
                "id": "doc_002",
                "title": "Machine Learning Introduction", 
                "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It involves algorithms that can identify patterns in data and make predictions or decisions. Common applications include recommendation systems, image recognition, natural language processing, and predictive analytics. Machine learning can be supervised, unsupervised, or reinforcement-based.",
                "category": "AI/ML",
                "tags": ["machine-learning", "ai", "algorithms", "data-science"]
            },
            {
                "id": "doc_003",
                "title": "Vector Databases Explained",
                "content": "Vector databases are specialized database systems designed to store, index, and search high-dimensional vector embeddings efficiently. They enable semantic search by finding similar vectors based on mathematical distance metrics like cosine similarity. Vector databases are essential for building RAG applications, recommendation systems, and similarity search engines. Popular examples include Pinecone, Weaviate, and Chroma.",
                "category": "Database",
                "tags": ["vector-database", "embeddings", "semantic-search", "similarity"]
            },
            {
                "id": "doc_004",
                "title": "Deep Learning Neural Networks",
                "content": "Deep learning uses artificial neural networks with multiple layers to learn complex patterns in data. It has revolutionized computer vision, natural language processing, and speech recognition. Key architectures include CNNs for image processing, RNNs for sequential data, and Transformers for language understanding. Deep learning automatically extracts features from raw data without manual feature engineering.",
                "category": "AI/ML",
                "tags": ["deep-learning", "neural-networks", "cnn", "rnn", "transformers"]
            },
            {
                "id": "doc_005",
                "title": "Natural Language Processing",
                "content": "Natural Language Processing (NLP) combines computational linguistics with machine learning to help computers understand, interpret, and generate human language. Modern NLP uses transformer models like BERT and GPT for tasks such as sentiment analysis, machine translation, text summarization, and question answering. NLP powers chatbots, search engines, and automated content analysis.",
                "category": "AI/ML",
                "tags": ["nlp", "transformers", "bert", "gpt", "language-models"]
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
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Validate data structure
        if not isinstance(data, list):
            raise ValueError("Dataset must be a list of documents")
        
        for i, doc in enumerate(data):
            if not isinstance(doc, dict):
                raise ValueError(f"Document {i} must be a dictionary")
            if 'content' not in doc:
                raise ValueError(f"Document {i} missing required 'content' field")
        
        return data
    
    def _extract_components(self, data: List[Dict]) -> Tuple[List[str], List[str], List[str], List[Dict]]:
        """Extract and organize dataset components with proper variable scoping."""
        ids = []
        documents = []
        titles = []
        metadata_list = []
        
        for i, doc in enumerate(data):
            # Extract fields with proper variable scoping - define all variables here
            doc_id = doc.get('id', f"doc_{i:03d}")
            document_content = doc.get('content', '')
            document_title = doc.get('title', f'Document {doc_id}')
            
            # Append to lists
            ids.append(doc_id)
            documents.append(document_content)
            titles.append(document_title)
            
            # Prepare metadata (everything except content to avoid duplication)
            metadata = {k: v for k, v in doc.items() if k != 'content'}
            if 'title' not in metadata:
                metadata['title'] = document_title
            metadata_list.append(metadata)
        
        logger.info(f"Extracted {len(ids)} documents for processing")
        return ids, documents, titles, metadata_list

# Global instance for import convenience - MUST BE AT END OF FILE
dataset_manager = DatasetManager()