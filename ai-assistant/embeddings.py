"""Multi-modal embedding generation for LMS content, users, and FAQs."""

import logging
import numpy as np
import torch
from typing import List, Dict, Union, Any
from sentence_transformers import SentenceTransformer
from config import Config

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Advanced embedding generator supporting different content types and user profiles."""
    
    def __init__(self, model_name: str = None):
        """Initialize embedding generator with specified model."""
        self.model_name = model_name or Config.EMBEDDING_MODEL
        self.device = self._get_optimal_device()
        self.model = None
        self.dimension = Config.EMBEDDING_DIMENSION
        
        logger.info(f"Initializing embedding generator: {self.model_name} on {self.device}")
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
        """Load the sentence transformer model."""
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            
            # Verify dimension
            actual_dim = self.model.get_sentence_embedding_dimension()
            if actual_dim != self.dimension:
                logger.warning(f"Model dimension {actual_dim} differs from config {self.dimension}")
                self.dimension = actual_dim
            
            logger.info(f"Model loaded successfully: {self.dimension}D embeddings")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Could not initialize embedding model: {e}")
    
    def generate_content_embeddings(self, courses_data: List[Dict]) -> tuple[np.ndarray, List[Dict]]:
        """Generate embeddings for course content and materials."""
        content_items = []
        texts_to_embed = []
        
        logger.info(f"Processing {len(courses_data)} courses for embedding generation")
        
        # Extract all content items with enhanced text representation
        for course in courses_data:
            course_base = {
                'course_id': course['course_id'],
                'course_title': course['course_title'],
                'instructor': course['instructor'],
                'subject_area': course['subject_area'],
                'difficulty_level': course['difficulty_level']
            }
            
            # Add course-level content
            course_text = self._prepare_course_text(course)
            content_items.append({
                **course_base,
                'material_id': course['course_id'],
                'title': course['course_title'],
                'type': 'course',
                'tags': []
            })
            texts_to_embed.append(course_text)
            
            # Add material-level content
            for material in course.get('materials', []):
                material_text = self._prepare_material_text(material, course)
                content_items.append({
                    **course_base,
                    **material,
                    'material_id': material.get('material_id', f"{course['course_id']}_{len(content_items)}")
                })
                texts_to_embed.append(material_text)
        
        if not texts_to_embed:
            raise ValueError("No content found to embed")
        
        logger.info(f"Preparing to embed {len(texts_to_embed)} content items")
        embeddings = self._generate_embeddings(texts_to_embed, "content")
        
        return embeddings, content_items
    
    def generate_user_embeddings(self, users_data: List[Dict]) -> tuple[np.ndarray, List[Dict]]:
        """Generate embeddings for user profiles."""
        texts_to_embed = []
        
        for user in users_data:
            user_text = self._prepare_user_text(user)
            texts_to_embed.append(user_text)
        
        if not texts_to_embed:
            raise ValueError("No users found to embed")
        
        logger.info(f"Preparing to embed {len(texts_to_embed)} user profiles")
        embeddings = self._generate_embeddings(texts_to_embed, "users")
        
        return embeddings, users_data
    
    def generate_faq_embeddings(self, faq_data: List[Dict]) -> tuple[np.ndarray, List[Dict]]:
        """Generate embeddings for FAQ items."""
        texts_to_embed = []
        
        for faq in faq_data:
            faq_text = self._prepare_faq_text(faq)
            texts_to_embed.append(faq_text)
        
        if not texts_to_embed:
            raise ValueError("No FAQs found to embed")
        
        logger.info(f"Preparing to embed {len(texts_to_embed)} FAQ items")
        embeddings = self._generate_embeddings(texts_to_embed, "faq")
        
        return embeddings, faq_data
    
    def _prepare_course_text(self, course: Dict) -> str:
        """Prepare comprehensive text representation of a course."""
        parts = [
            f"Course: {course.get('course_title', '')}",
            f"Subject: {course.get('subject_area', '')}",
            f"Level: {course.get('difficulty_level', '')}",
            f"Instructor: {course.get('instructor', '')}",
            f"Description: {course.get('course_description', '')}"
        ]
        
        # Add material summaries
        materials = course.get('materials', [])
        if materials:
            material_types = list(set(m.get('type', '') for m in materials))
            parts.append(f"Materials include: {', '.join(material_types)}")
            
            # Add sample content
            for material in materials[:2]:  # First 2 materials for context
                if material.get('content'):
                    content_preview = material['content'][:200] + "..." if len(material['content']) > 200 else material['content']
                    parts.append(f"Content sample: {content_preview}")
        
        return " ".join(parts)
    
    def _prepare_material_text(self, material: Dict, course: Dict) -> str:
        """Prepare text representation of course material."""
        parts = [
            f"Title: {material.get('title', '')}",
            f"Type: {material.get('type', '')}",
            f"Course: {course.get('course_title', '')}",
            f"Subject: {course.get('subject_area', '')}",
            f"Level: {course.get('difficulty_level', '')}"
        ]
        
        # Add content
        if material.get('content'):
            parts.append(f"Content: {material['content']}")
        
        # Add transcript if available
        if material.get('transcript'):
            parts.append(f"Transcript: {material['transcript']}")
        
        # Add tags
        tags = material.get('tags', [])
        if tags:
            parts.append(f"Tags: {', '.join(tags)}")
        
        return " ".join(parts)
    
    def _prepare_user_text(self, user: Dict) -> str:
        """Prepare text representation of user profile for embedding."""
        profile = user.get('profile', {})
        parts = [
            f"Role: {user.get('role', '')}",
            f"Major: {profile.get('major', '')}",
            f"Year: {profile.get('year', '')}",
            f"Learning style: {profile.get('learning_style', '')}",
            f"Preferred difficulty: {profile.get('preferred_difficulty', '')}"
        ]
        
        # Add interests
        interests = profile.get('interests', [])
        if interests:
            parts.append(f"Interests: {', '.join(interests)}")
        
        # Add goals
        goals = profile.get('goals', [])
        if goals:
            parts.append(f"Goals: {', '.join(goals)}")
        
        # Add enrolled courses context
        enrolled = user.get('enrolled_courses', [])
        if enrolled:
            parts.append(f"Currently studying: {', '.join(enrolled)}")
        
        # Add interaction history context (recent searches)
        recent_searches = []
        for interaction in user.get('interaction_history', [])[-5:]:  # Last 5 interactions
            if interaction.get('action') == 'search':
                recent_searches.append(interaction.get('query', ''))
        
        if recent_searches:
            parts.append(f"Recent searches: {', '.join(recent_searches)}")
        
        return " ".join(parts)
    
    def _prepare_faq_text(self, faq: Dict) -> str:
        """Prepare text representation of FAQ for embedding."""
        parts = [
            f"Category: {faq.get('category', '')}",
            f"Question: {faq.get('question', '')}",
            f"Answer: {faq.get('answer', '')}"
        ]
        
        # Add keywords
        keywords = faq.get('keywords', [])
        if keywords:
            parts.append(f"Keywords: {', '.join(keywords)}")
        
        return " ".join(parts)
    
    def _generate_embeddings(self, texts: List[str], content_type: str, batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if not self.model:
            raise RuntimeError("Model not initialized")
        
        if not texts:
            raise ValueError("Cannot embed empty text list")
        
        try:
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (len(texts) - 1) // batch_size + 1
                
                logger.info(f"Embedding {content_type} batch {batch_num}/{total_batches}: {len(batch_texts)} items")
                
                # Generate embeddings for batch
                batch_embeddings = self.model.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                
                all_embeddings.append(batch_embeddings)
            
            # Combine all batches
            if len(all_embeddings) == 1:
                embeddings = all_embeddings[0]
            else:
                embeddings = np.vstack(all_embeddings)
            
            # Ensure float32 for Pinecone compatibility
            embeddings = embeddings.astype(np.float32)
            
            logger.info(f"Generated {len(embeddings)} {content_type} embeddings of shape {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate {content_type} embeddings: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}")
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for a search query."""
        if not self.model:
            raise RuntimeError("Model not initialized")
        
        try:
            embedding = self.model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            
            return embedding[0].astype(np.float32)
            
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            raise RuntimeError(f"Query embedding failed: {e}")
    
    def update_user_embedding(self, user_data: Dict, new_interactions: List[Dict]) -> np.ndarray:
        """Update user embedding based on new interactions."""
        # Add new interactions to user history
        updated_user = user_data.copy()
        existing_history = updated_user.get('interaction_history', [])
        updated_user['interaction_history'] = existing_history + new_interactions
        
        # Generate new embedding
        user_text = self._prepare_user_text(updated_user)
        embedding = self._generate_embeddings([user_text], "user_update")
        
        return embedding[0]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for debugging."""
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "device": self.device,
            "max_seq_length": getattr(self.model, 'max_seq_length', 512) if self.model else None
        }