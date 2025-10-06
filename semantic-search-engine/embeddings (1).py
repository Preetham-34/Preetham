"""Embedding generation using Sentence Transformers - Final version."""

import logging
import numpy as np
import torch
from typing import List, Union
from sentence_transformers import SentenceTransformer
from config import Config

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Generates embeddings using sentence transformers following best practices."""
    
    def __init__(self, model_name: str = None):
        """
        Initialize embedding generator.
        
        Args:
            model_name: Sentence transformer model name
        """
        self.model_name = model_name or Config.EMBEDDING_MODEL
        self.device = self._get_optimal_device()
        self.model = None
        self.dimension = Config.EMBEDDING_DIMENSION
        
        logger.info(f"Initializing embedding generator with {self.model_name} on {self.device}")
        self._load_model()
    
    def _get_optimal_device(self) -> str:
        """Determine the best available device for inference."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"  # Apple Silicon
        else:
            return "cpu"
    
    def _load_model(self) -> None:
        """Load the sentence transformer model with error handling."""
        try:
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device
            )
            # Verify dimension matches configuration
            actual_dim = self.model.get_sentence_embedding_dimension()
            if actual_dim != self.dimension:
                logger.warning(f"Model dimension {actual_dim} differs from config {self.dimension}")
                self.dimension = actual_dim
            
            logger.info(f"Model loaded successfully: {self.dimension}D embeddings")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Could not initialize embedding model: {e}")
    
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text inputs.
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            Normalized embeddings as float32 numpy array
        """
        if not self.model:
            raise RuntimeError("Model not initialized")
        
        # Ensure input is a list
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            raise ValueError("Cannot encode empty text list")
        
        try:
            # Generate embeddings with normalization (important for cosine similarity)
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,  # Critical for vector search
                show_progress_bar=len(texts) > 10
            )
            
            # Ensure float32 for Pinecone compatibility
            embeddings = embeddings.astype(np.float32)
            
            logger.debug(f"Generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise RuntimeError(f"Could not generate embeddings: {e}")
    
    def encode_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        embeddings = self.encode([text])
        return embeddings[0]
    
    def get_model_info(self) -> dict:
        """Get model information for debugging."""
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "device": self.device,
            "max_seq_length": getattr(self.model, 'max_seq_length', 512) if self.model else None
        }