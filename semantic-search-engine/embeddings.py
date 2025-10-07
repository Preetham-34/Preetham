"""Embedding generation with batch processing - Production Ready with Enhanced Debugging."""

import logging
import numpy as np
import torch
from typing import List, Union
from sentence_transformers import SentenceTransformer
from config import Config

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Generates embeddings using sentence transformers with batch processing support."""
    
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
        print(f"[EMBEDDING] Initializing model {self.model_name} on device {self.device}")
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
                print(f"[EMBEDDING] Warning: Model dimension {actual_dim} != config {self.dimension}")
                self.dimension = actual_dim
            
            logger.info(f"Model loaded successfully: {self.dimension}D embeddings")
            print(f"[EMBEDDING] Model loaded successfully: {self.dimension}D embeddings")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            print(f"[EMBEDDING] ERROR: Failed to load model: {e}")
            raise RuntimeError(f"Could not initialize embedding model: {e}")
    
    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for text inputs with batch processing.
        
        Args:
            texts: Single text or list of texts
            batch_size: Number of texts to process in each batch
            
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
        
        print(f"[EMBEDDING] Starting embedding generation for {len(texts)} texts in batches of {batch_size}")
        
        try:
            embeddings = []
            
            # Process in batches for memory efficiency
            for start_idx in range(0, len(texts), batch_size):
                batch_texts = texts[start_idx:start_idx + batch_size]
                batch_num = (start_idx // batch_size) + 1
                total_batches = (len(texts) - 1) // batch_size + 1
                
                print(f"[EMBEDDING] Processing batch {batch_num}/{total_batches}: {len(batch_texts)} texts")
                
                # Generate embeddings for batch
                batch_embeddings = self.model.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,  # Critical for vector search
                    show_progress_bar=False
                )
                
                embeddings.append(batch_embeddings)
            
            # Combine all batches
            if len(embeddings) == 1:
                embeddings = embeddings[0]
            else:
                embeddings = np.vstack(embeddings)
            
            # Ensure float32 for Pinecone compatibility
            embeddings = embeddings.astype(np.float32)
            
            logger.info(f"Generated {len(embeddings)} embeddings of shape {embeddings.shape}")
            print(f"[EMBEDDING] Successfully generated {len(embeddings)} embeddings of shape {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            print(f"[EMBEDDING] ERROR: Embedding generation failed: {e}")
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