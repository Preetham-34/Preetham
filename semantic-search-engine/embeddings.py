"""Embeddings generation with batch processing support for large datasets."""

import logging
import numpy as np
import torch
from typing import List, Union
from sentence_transformers import SentenceTransformer
from config import Config

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Generates embeddings, supports batch processing for scalability."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or Config.EMBEDDING_MODEL
        self.device = self._get_device()
        self.model = None
        self.dimension = Config.EMBEDDING_DIMENSION

        logger.info(f"Loading model {self.model_name} on device {self.device}")
        self._load_model()

    def _get_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _load_model(self):
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            model_dim = self.model.get_sentence_embedding_dimension()
            if model_dim != self.dimension:
                logger.warning(f"Model embedding dimension {model_dim} differs from configured {self.dimension}")
                self.dimension = model_dim
            logger.info(f"Model loaded successfully with embedding dimension {self.dimension}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def encode(self, texts: Union[List[str], str], batch_size: int = 64) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        embeddings = []
        try:
            for start_idx in range(0, len(texts), batch_size):
                batch_texts = texts[start_idx:start_idx + batch_size]
                batch_embs = self.model.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                embeddings.append(batch_embs)
            embeddings = np.vstack(embeddings)
            embeddings = embeddings.astype(np.float32)
            logger.info(f"Generated embeddings for {len(texts)} texts in batches of {batch_size}")
            return embeddings
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def get_model_info(self) -> dict:
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.dimension,
            "device": self.device,
        }
