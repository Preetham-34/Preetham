"""Vector Database Management for LMS AI System using Pinecone."""

import logging
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pinecone import Pinecone, ServerlessSpec, PodSpec
from config import Config

logger = logging.getLogger(__name__)

class VectorDBManager:
    """Enhanced vector database manager supporting multiple indexes for different data types."""
    
    def __init__(self):
        """Initialize Pinecone connection and indexes."""
        self.pc = None
        self.indexes = {}
        self._initialize_client()
        
    def _initialize_client(self) -> None:
        """Initialize Pinecone client and ensure all required indexes exist."""
        try:
            if not Config.PINECONE_API_KEY:
                raise ValueError("Pinecone API key is required but not provided")
                
            self.pc = Pinecone(api_key=Config.PINECONE_API_KEY)
            logger.info("Connected to Pinecone successfully")
            
            # Ensure all required indexes exist
            self._ensure_indexes_exist()
            
            # Connect to all indexes
            self.indexes = {
                'content': self.pc.Index(Config.CONTENT_INDEX),
                'users': self.pc.Index(Config.USER_INDEX),
                'faq': self.pc.Index(Config.FAQ_INDEX)
            }
            
            logger.info("All vector indexes initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise RuntimeError(f"Vector database initialization failed: {e}")
    
    def _ensure_indexes_exist(self) -> None:
        """Create indexes if they don't exist."""
        indexes_to_create = [
            Config.CONTENT_INDEX,
            Config.USER_INDEX, 
            Config.FAQ_INDEX
        ]
        
        try:
            existing_indexes = self.pc.list_indexes()
            existing_names = [idx.name for idx in existing_indexes] if existing_indexes else []
        except Exception as e:
            logger.warning(f"Could not list existing indexes: {e}")
            existing_names = []
        
        for index_name in indexes_to_create:
            if index_name not in existing_names:
                logger.info(f"Creating index: {index_name}")
                try:
                    # Try serverless first
                    self.pc.create_index(
                        name=index_name,
                        dimension=Config.EMBEDDING_DIMENSION,
                        metric=Config.SIMILARITY_METRIC,
                        spec=ServerlessSpec(
                            cloud='aws',
                            region='us-east-1'
                        )
                    )
                    self._wait_for_index_ready(index_name)
                except Exception as e:
                    logger.warning(f"Serverless creation failed for {index_name}: {e}")
                    # Fallback to pod-based
                    try:
                        self.pc.create_index(
                            name=index_name,
                            dimension=Config.EMBEDDING_DIMENSION,
                            metric=Config.SIMILARITY_METRIC,
                            spec=PodSpec(
                                environment=Config.PINECONE_ENVIRONMENT,
                                pod_type="p1.x1"
                            )
                        )
                        self._wait_for_index_ready(index_name)
                    except Exception as pod_error:
                        logger.error(f"Failed to create index {index_name}: {pod_error}")
                        raise
            else:
                logger.info(f"Index {index_name} already exists")
    
    def _wait_for_index_ready(self, index_name: str, timeout: int = 60) -> None:
        """Wait for index to become ready."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                index_info = self.pc.describe_index(index_name)
                if hasattr(index_info, 'status') and index_info.status.get('ready'):
                    logger.info(f"Index {index_name} is ready")
                    return
                time.sleep(2)
            except Exception:
                time.sleep(2)
        logger.warning(f"Index {index_name} readiness timeout")
    
    def upsert_content_embeddings(self, 
                                content_data: List[Dict],
                                embeddings: np.ndarray,
                                batch_size: int = 100) -> bool:
        """Upsert content embeddings with metadata."""
        return self._upsert_embeddings('content', content_data, embeddings, batch_size)
    
    def upsert_user_embeddings(self,
                             user_data: List[Dict], 
                             embeddings: np.ndarray,
                             batch_size: int = 100) -> bool:
        """Upsert user profile embeddings."""
        return self._upsert_embeddings('users', user_data, embeddings, batch_size)
    
    def upsert_faq_embeddings(self,
                            faq_data: List[Dict],
                            embeddings: np.ndarray, 
                            batch_size: int = 100) -> bool:
        """Upsert FAQ embeddings."""
        return self._upsert_embeddings('faq', faq_data, embeddings, batch_size)
    
    def _upsert_embeddings(self,
                          index_type: str,
                          data: List[Dict],
                          embeddings: np.ndarray,
                          batch_size: int) -> bool:
        """Generic method to upsert embeddings to specified index."""
        try:
            if len(data) != len(embeddings):
                raise ValueError(f"Data length ({len(data)}) doesn't match embeddings length ({len(embeddings)})")
            
            index = self.indexes[index_type]
            total_upserted = 0
            
            for i in range(0, len(data), batch_size):
                batch_data = data[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size]
                
                vectors = []
                for j, (item, embedding) in enumerate(zip(batch_data, batch_embeddings)):
                    vector_id = self._get_vector_id(index_type, item)
                    metadata = self._prepare_metadata(index_type, item)
                    
                    vectors.append((
                        vector_id,
                        embedding.astype(np.float32).tolist(),
                        metadata
                    ))
                
                batch_num = (i // batch_size) + 1
                total_batches = (len(data) - 1) // batch_size + 1
                
                logger.info(f"Upserting batch {batch_num}/{total_batches} to {index_type} index")
                
                response = index.upsert(vectors=vectors)
                
                if hasattr(response, 'upserted_count'):
                    batch_upserted = response.upserted_count
                else:
                    batch_upserted = len(vectors)
                
                total_upserted += batch_upserted
                logger.info(f"Batch {batch_num} upserted: {batch_upserted} vectors")
            
            logger.info(f"Successfully upserted {total_upserted} vectors to {index_type} index")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upsert to {index_type} index: {e}")
            return False
    
    def _get_vector_id(self, index_type: str, item: Dict) -> str:
        """Generate vector ID based on index type and item."""
        if index_type == 'content':
            return item.get('material_id', item.get('course_id', f"content_{hash(str(item))}"))
        elif index_type == 'users':
            return item.get('user_id', f"user_{hash(str(item))}")
        elif index_type == 'faq':
            return item.get('faq_id', f"faq_{hash(str(item))}")
        else:
            return f"{index_type}_{hash(str(item))}"
    
    def _prepare_metadata(self, index_type: str, item: Dict) -> Dict:
        """Prepare metadata for vector storage based on index type."""
        metadata = {}
        
        if index_type == 'content':
            metadata.update({
                'type': 'content',
                'title': item.get('title', ''),
                'course_id': item.get('course_id', ''),
                'content_type': item.get('type', ''),
                'subject_area': item.get('subject_area', ''),
                'difficulty_level': item.get('difficulty_level', ''),
                'instructor': item.get('instructor', ''),
                'tags': item.get('tags', [])
            })
        elif index_type == 'users':
            profile = item.get('profile', {})
            metadata.update({
                'type': 'user',
                'username': item.get('username', ''),
                'role': item.get('role', ''),
                'major': profile.get('major', ''),
                'interests': profile.get('interests', []),
                'learning_style': profile.get('learning_style', ''),
                'preferred_difficulty': profile.get('preferred_difficulty', '')
            })
        elif index_type == 'faq':
            metadata.update({
                'type': 'faq',
                'category': item.get('category', ''),
                'question': item.get('question', ''),
                'keywords': item.get('keywords', []),
                'popularity_score': item.get('popularity_score', 0.0)
            })
        
        return metadata
    
    def search_content(self, 
                      query_embedding: np.ndarray,
                      top_k: int = None,
                      filters: Dict = None) -> List[Dict]:
        """Search content index."""
        return self._search_index('content', query_embedding, top_k, filters)
    
    def search_users(self,
                    query_embedding: np.ndarray, 
                    top_k: int = None,
                    filters: Dict = None) -> List[Dict]:
        """Search users index."""
        return self._search_index('users', query_embedding, top_k, filters)
    
    def search_faq(self,
                  query_embedding: np.ndarray,
                  top_k: int = None, 
                  filters: Dict = None) -> List[Dict]:
        """Search FAQ index."""
        return self._search_index('faq', query_embedding, top_k, filters)
    
    def _search_index(self,
                     index_type: str,
                     query_embedding: np.ndarray,
                     top_k: int = None,
                     filters: Dict = None) -> List[Dict]:
        """Generic search method for any index."""
        try:
            top_k = min(top_k or Config.DEFAULT_TOP_K, Config.MAX_TOP_K)
            index = self.indexes[index_type]
            
            # Prepare query vector
            if len(query_embedding.shape) > 1:
                query_embedding = query_embedding.flatten()
            query_vector = query_embedding.astype(np.float32).tolist()
            
            # Perform search
            response = index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
                filter=filters
            )
            
            # Parse results
            results = []
            if hasattr(response, 'matches') and response.matches:
                for match in response.matches:
                    result = {
                        'id': match.id,
                        'score': float(match.score),
                        'metadata': match.metadata if hasattr(match, 'metadata') else {}
                    }
                    results.append(result)
            
            logger.info(f"Search in {index_type} returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Search failed in {index_type} index: {e}")
            return []
    
    def get_index_stats(self, index_type: str) -> Dict[str, Any]:
        """Get statistics for a specific index."""
        try:
            index = self.indexes.get(index_type)
            if not index:
                return {}
            
            stats = index.describe_index_stats()
            return {
                'total_vectors': getattr(stats, 'total_vector_count', 0),
                'dimension': getattr(stats, 'dimension', Config.EMBEDDING_DIMENSION),
                'index_fullness': getattr(stats, 'index_fullness', 0.0)
            }
        except Exception as e:
            logger.warning(f"Could not get stats for {index_type}: {e}")
            return {}
    
    def health_check(self) -> Dict[str, bool]:
        """Check health of all indexes."""
        health_status = {}
        for index_type in ['content', 'users', 'faq']:
            try:
                stats = self.get_index_stats(index_type)
                health_status[index_type] = bool(stats)
            except Exception:
                health_status[index_type] = False
        return health_status