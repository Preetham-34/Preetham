"""Pinecone vector database management - Production Ready with Enhanced Debugging."""

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
        print(f"[PINECONE] Initializing connection to index: {self.index_name}")
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize Pinecone client using the official pinecone package."""
        try:
            # Initialize Pinecone using the new API
            self.pc = Pinecone(api_key=self.api_key)
            print(f"[PINECONE] Connected to Pinecone API")
            
            # Ensure index exists
            self._ensure_index_exists()
            
            # Connect to index
            self.index = self.pc.Index(self.index_name)
            
            logger.info("Pinecone initialized successfully")
            print(f"[PINECONE] Successfully connected to index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Pinecone initialization failed: {e}")
            print(f"[PINECONE] ERROR: Initialization failed: {e}")
            raise RuntimeError(f"Could not initialize Pinecone: {e}")
    
    def _ensure_index_exists(self) -> None:
        """Create index if it doesn't exist."""
        try:
            # Get existing indexes
            existing_indexes = self.pc.list_indexes()
            index_names = [index.name for index in existing_indexes] if existing_indexes else []
            
            if self.index_name not in index_names:
                logger.info(f"Creating index: {self.index_name}")
                print(f"[PINECONE] Creating new index: {self.index_name}")
                
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
                    print(f"[PINECONE] Created serverless index successfully")
                except Exception as serverless_error:
                    logger.warning(f"Serverless creation failed: {serverless_error}")
                    print(f"[PINECONE] Serverless creation failed, trying pod-based...")
                    
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
                    print(f"[PINECONE] Created pod-based index successfully")
                
                # Wait for index to be ready
                self._wait_for_index_ready()
            else:
                print(f"[PINECONE] Index {self.index_name} already exists")
            
            logger.info(f"Index {self.index_name} is ready")
            
        except Exception as e:
            logger.error(f"Error ensuring index exists: {e}")
            print(f"[PINECONE] ERROR: Failed to ensure index exists: {e}")
            raise
    
    def _wait_for_index_ready(self, timeout: int = 60) -> None:
        """Wait for index to become ready."""
        start_time = time.time()
        print(f"[PINECONE] Waiting for index to be ready (timeout: {timeout}s)...")
        
        while time.time() - start_time < timeout:
            try:
                index_info = self.pc.describe_index(self.index_name)
                if hasattr(index_info, 'status') and index_info.status.get('ready'):
                    print(f"[PINECONE] Index is ready!")
                    return
                time.sleep(2)
            except Exception:
                time.sleep(2)
        
        logger.warning(f"Index {self.index_name} readiness timeout - proceeding anyway")
        print(f"[PINECONE] WARNING: Index readiness timeout, proceeding anyway")
    
    def add_embeddings(self, 
                      ids: List[str], 
                      embeddings: np.ndarray, 
                      metadata_list: List[Dict] = None) -> bool:
        """
        Add embeddings to Pinecone index with robust error handling.
        
        Args:
            ids: Document IDs
            embeddings: Embedding vectors
            metadata_list: Optional metadata for each embedding
            
        Returns:
            Success status
        """
        try:
            if len(ids) != len(embeddings):
                raise ValueError(f"IDs ({len(ids)}) and embeddings ({len(embeddings)}) length mismatch")
            
            print(f"[PINECONE] Preparing {len(ids)} vectors for upload...")
            
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
                batch_num = (i // batch_size) + 1
                total_batches = (len(vectors) - 1) // batch_size + 1
                
                print(f"[PINECONE] Uploading batch {batch_num}/{total_batches}: {len(batch)} vectors")
                
                try:
                    response = self.index.upsert(vectors=batch)
                    
                    # Handle response format
                    if hasattr(response, 'upserted_count'):
                        batch_uploaded = response.upserted_count
                    else:
                        batch_uploaded = len(batch)
                    
                    total_upserted += batch_uploaded
                    print(f"[PINECONE] Batch {batch_num} uploaded successfully: {batch_uploaded} vectors")
                    
                except Exception as batch_error:
                    logger.error(f"Batch {batch_num} upload failed: {batch_error}")
                    print(f"[PINECONE] ERROR: Batch {batch_num} upload failed: {batch_error}")
                    return False
            
            logger.info(f"Successfully upserted {total_upserted} embeddings")
            print(f"[PINECONE] Upload complete! Total vectors uploaded: {total_upserted}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding embeddings: {e}")
            print(f"[PINECONE] ERROR: Failed to add embeddings: {e}")
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
            print(f"[PINECONE] ERROR: Search failed: {e}")
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