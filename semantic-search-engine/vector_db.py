import os
import logging
import time
from typing import List, Dict, Any
import numpy as np

from pinecone import Pinecone, ServerlessSpec
from config import Config

logger = logging.getLogger(__name__)

class PineconeDB:
    def __init__(self,
                 api_key: str = None,
                 environment: str = None,
                 index_name: str = None):
        self.api_key = api_key or Config.PINECONE_API_KEY
        self.environment = environment or Config.PINECONE_ENVIRONMENT
        self.index_name = index_name or Config.PINECONE_INDEX_NAME
        self.dimension = Config.EMBEDDING_DIMENSION
        self.metric = Config.SIMILARITY_METRIC

        logger.info(f"Initializing PineconeDB instance for index {self.index_name} in env {self.environment}")

        # Create Pinecone class instance
        self.pc = Pinecone(api_key=self.api_key)  # Pass API key here

        # Ensure index exists
        self._ensure_index_exists()

        # Connect to index
        self.index = self.pc.Index(self.index_name)

    def _ensure_index_exists(self):
            indexes = self.pc.list_indexes().names()
            if self.index_name not in indexes:
            self.pc.create_index(
            name=self.index_name,
            dimension=self.dimension,
            metric=self.metric,
            spec=ServerlessSpec(
                cloud='gcp',     # Make sure this matches your Pinecone environment
                region='us-west1'  # Change to supported region
            )
        )
        self._wait_for_index_ready()

    def _wait_for_index_ready(self, timeout: int = 60):
        start = time.time()
        while time.time() - start < timeout:
            try:
                info = self.pc.describe_index(self.index_name)
                if info.status['ready']:
                    return
            except Exception:
                time.sleep(2)
        logger.warning(f"Index {self.index_name} not fully ready after {timeout} seconds")

    def add_embeddings(self,
                       ids: List[str],
                       embeddings: np.ndarray,
                       metadata_list: List[Dict[str, Any]] = None) -> bool:
        try:
            vectors = []
            for idx, emb in enumerate(embeddings):
                meta = metadata_list[idx] if metadata_list else {}
                vectors.append((ids[idx], emb.tolist(), meta))
            res = self.index.upsert(vectors=vectors)
            logger.info(f"Upserted {getattr(res, 'upserted_count', len(vectors))} vectors")
            return True
        except Exception as e:
            logger.error(f"Upsert failed: {e}")
            return False

    def search(self,
               query_embedding: np.ndarray,
               top_k: int = Config.DEFAULT_TOP_K,
               include_metadata: bool = True) -> List[Dict[str, Any]]:
        try:
            query_vector = query_embedding.tolist()
            res = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=include_metadata,
            )
            results = []
            if res.matches:
                for m in res.matches:
                    result = {'id': m.id, 'score': m.score}
                    if m.metadata:
                        result['metadata'] = m.metadata
                    results.append(result)
            return results
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
