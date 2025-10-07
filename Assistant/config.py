"""Configuration management for the LMS Assistive AI System."""

import os
from typing import Optional
import streamlit as st

class Config:
    """Central configuration class for the LMS AI system."""
    
    @staticmethod
    def get_secret(key: str, default: Optional[str] = None) -> str:
        """Get secret from environment or Streamlit secrets."""
        try:
            if hasattr(st, 'secrets') and key in st.secrets:
                return st.secrets[key]
        except Exception:
            pass
        return os.getenv(key, default or "")

    # Vector Database Configuration
    PINECONE_API_KEY = get_secret.__func__("PINECONE_API_KEY", "")
    PINECONE_ENVIRONMENT = get_secret.__func__("PINECONE_ENVIRONMENT", "gcp-starter")

    # Index Names for Different Data Types
    CONTENT_INDEX = get_secret.__func__("CONTENT_INDEX", "lms-content")
    USER_INDEX = get_secret.__func__("USER_INDEX", "lms-users") 
    FAQ_INDEX = get_secret.__func__("FAQ_INDEX", "lms-faq")

    # Embedding Model Configuration
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION = 384
    SIMILARITY_METRIC = "cosine"

    # Logging Level
    LOG_LEVEL = "INFO"

    # API Configuration
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    MAX_QUERY_LENGTH = 500
    DEFAULT_TOP_K = 5
    MAX_TOP_K = 20

    # Content Categories
    CONTENT_TYPES = [
        "video", "document", "audio", "image", "quiz", 
        "assignment", "forum", "lecture", "tutorial"
    ]
    
    SUBJECT_AREAS = [
        "Computer Science", "Mathematics", "Physics", "Chemistry", 
        "Biology", "History", "Literature", "Business", "Psychology", 
        "Engineering", "Art", "Music", "Languages"
    ]
    
    DIFFICULTY_LEVELS = ["Beginner", "Intermediate", "Advanced", "Expert"]

    # User Interaction Types
    INTERACTION_TYPES = [
        "view", "complete", "bookmark", "like", "share", 
        "comment", "quiz_attempt", "search", "download"
    ]

    @classmethod
    def validate(cls) -> bool:
        """Validate required configuration."""
        return bool(cls.PINECONE_API_KEY)

    @classmethod
    def get_deployment_info(cls) -> dict:
        """Get deployment configuration info."""
        return {
            "pinecone_configured": bool(cls.PINECONE_API_KEY),
            "environment": cls.PINECONE_ENVIRONMENT,
            "content_index": cls.CONTENT_INDEX,
            "user_index": cls.USER_INDEX,
            "faq_index": cls.FAQ_INDEX,
            "embedding_model": cls.EMBEDDING_MODEL,
            "embedding_dimension": cls.EMBEDDING_DIMENSION
        }
