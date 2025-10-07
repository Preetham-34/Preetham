"""Pinecone vector database management - Complete final version."""

import logging
import time
from typing import List, Dict, Any, Optional
import numpy as np
from pinecone import Pinecone, ServerlessSpec, PodSpec
from config import Config

logger = logging.getLogger(__name__)

class PineconeDB:
    """Pinecone vector database manager using the official pinecone package."""
    
    def __init__(self, 
                 api_key: str = None,
                 environment: str = None,
                 index_name: str = None):
        """
        Initialize Pinecone connection.
        
        Args:
            api_key: Pinecone API key
            environment: Pinecone environment
            index_name: Index name
        """
        self.api_key = api_key or Config.PINECONE_API_KEY
        self.environment = environment or Config.PINECONE_ENVIRONMENT
        self.index_name = index_name or Config.PINECONE_INDEX_NAME
        self.dimension = Config.EMBEDDING_DIMENSION
        self.metric = Config.SIMILARITY_METRIC
        
        self.pc = None
        self.index = None
        
        logger.info(f"Initializing Pinecone DB: {self.index_name}")
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize Pinecone client using the official pinecone package."""
        try:
            # Initialize Pinecone using the new API
            self.pc = Pinecone(api_key=self.api_key)
            
            # Ensure index exists
            self._ensure_index_exists()
            
            # Connect to index
            self.index = self.pc.Index(self.index_name)
            
            logger.info("Pinecone initialized successfully")
            
        except Exception as e:
            logger.error(f"Pinecone initialization failed: {e}")
            raise RuntimeError(f"Could not initialize Pinecone: {e}")
    
    def _ensure_index_exists(self) -> None:
        """Create index if it doesn't exist."""
        try:
            # Get existing indexes
            existing_indexes = self.pc.list_indexes()
            index_names = [index.name for index in existing_indexes] if existing_indexes else []
            
            if self.index_name not in index_names:
                logger.info(f"Creating index: {self.index_name}")
                
                # Try serverless first (for free tier)
                try:
                    self.pc.create_index(
                        name=self.index_name,
                        dimension=self.dimension,
                        metric=self.metric,
                        spec=ServerlessSpec(
                            cloud='aws',
                            region='us-east-1'  # Most widely supported free tier region
                        )
                    )
                except Exception as serverless_error:
                    logger.warning(f"Serverless creation failed: {serverless_error}")
                    logger.info("Trying pod-based index creation...")
                    
                    # Fallback to pod-based for older accounts
                    self.pc.create_index(
                        name=self.index_name,
                        dimension=self.dimension,
                        metric=self.metric,
                        spec=PodSpec(
                            environment=self.environment,
                            pod_type="p1.x1"
                        )
                    )
                
                # Wait for index to be ready
                self._wait_for_index_ready()
            
            logger.info(f"Index {self.index_name} is ready")
            
        except Exception as e:
            logger.error(f"Error ensuring index exists: {e}")
            raise
    
    def _wait_for_index_ready(self, timeout: int = 60) -> None:
        """Wait for index to become ready."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                index_info = self.pc.describe_index(self.index_name)
                if hasattr(index_info, 'status') and index_info.status.get('ready'):
                    return
                time.sleep(2)
            except Exception:
                time.sleep(2)
        
        logger.warning(f"Index {self.index_name} readiness timeout - proceeding anyway")
    
    def add_embeddings(self, 
                      ids: List[str], 
                      embeddings: np.ndarray, 
                      metadata_list: List[Dict] = None) -> bool:
        """
        Add embeddings to Pinecone index.
        
        Args:
            ids: Document IDs
            embeddings: Embedding vectors
            metadata_list: Optional metadata for each embedding
            
        Returns:
            Success status
        """
        try:
            if len(ids) != len(embeddings):
                raise ValueError("IDs and embeddings length mismatch")
            
            # Prepare vectors for upsert
            vectors = []
            for i, (doc_id, embedding) in enumerate(zip(ids, embeddings)):
                metadata = metadata_list[i] if metadata_list else {}
                vector_tuple = (
                    str(doc_id),
                    embedding.tolist(),
                    metadata
                )
                vectors.append(vector_tuple)
            
            # Upsert vectors in batches
            batch_size = 100
            total_upserted = 0
            
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                response = self.index.upsert(vectors=batch)
                
                # Handle response format
                if hasattr(response, 'upserted_count'):
                    total_upserted += response.upserted_count
                else:
                    total_upserted += len(batch)
            
            logger.info(f"Successfully upserted {total_upserted} embeddings")
            return True
            
        except Exception as e:
            logger.error(f"Error adding embeddings: {e}")
            return False
    
    def search(self, 
              query_embedding: np.ndarray, 
              top_k: int = None, 
              include_metadata: bool = True) -> List[Dict]:
        """
        Search for similar vectors.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results
            include_metadata: Include metadata in results
            
        Returns:
            List of search results with scores
        """
        try:
            top_k = min(top_k or Config.DEFAULT_TOP_K, Config.MAX_TOP_K)
            
            # Ensure proper shape and type
            if len(query_embedding.shape) > 1:
                query_embedding = query_embedding.flatten()
            
            query_list = query_embedding.astype(np.float32).tolist()
            
            # Perform search
            response = self.index.query(
                vector=query_list,
                top_k=top_k,
                include_metadata=include_metadata
            )
            
            # Parse results
            results = []
            if hasattr(response, 'matches') and response.matches:
                for match in response.matches:
                    result = {
                        'id': match.id,
                        'score': float(match.score)
                    }
                    
                    if hasattr(match, 'metadata') and match.metadata:
                        result['metadata'] = match.metadata
                    
                    results.append(result)
            
            logger.debug(f"Search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        try:
            stats = self.index.describe_index_stats()
            return {
                'total_vectors': getattr(stats, 'total_vector_count', 0),
                'dimension': getattr(stats, 'dimension', self.dimension),
                'index_fullness': getattr(stats, 'index_fullness', 0.0)
            }
        except Exception as e:
            logger.warning(f"Could not get stats: {e}")
            return {}
    
    def health_check(self) -> bool:
        """Check database health."""
        try:
            stats = self.get_stats()
            return bool(stats)
        except Exception:
            return False