"""Recommendation engine for personalized content suggestions."""

import logging
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
from config import Config

logger = logging.getLogger(__name__)

class RecommendationEngine:
    """Advanced recommendation engine using vector similarity and collaborative filtering."""
    
    def __init__(self, vector_db_manager, embedding_generator):
        """Initialize recommendation engine with vector database and embedding generator."""
        self.vector_db = vector_db_manager
        self.embedding_gen = embedding_generator
    
    def get_content_recommendations(self, 
                                  user_id: str,
                                  num_recommendations: int = 5,
                                  content_filters: Dict = None) -> List[Dict]:
        """Get personalized content recommendations for a user."""
        try:
            # Get user embedding
            user_results = self.vector_db.search_users(
                query_embedding=np.zeros(Config.EMBEDDING_DIMENSION),  # Placeholder
                top_k=1,
                filters={'user_id': user_id}
            )
            
            if not user_results:
                logger.warning(f"User {user_id} not found, using fallback recommendations")
                return self._get_fallback_recommendations(num_recommendations, content_filters)
            
            # Use user's interests and profile to find relevant content
            user_metadata = user_results[0]['metadata']
            
            # Create search filters based on user profile
            search_filters = self._build_content_filters(user_metadata, content_filters)
            
            # Search for similar content using a zero vector (we'll improve this)
            content_results = self.vector_db.search_content(
                query_embedding=np.zeros(Config.EMBEDDING_DIMENSION),
                top_k=num_recommendations * 2,  # Get more to filter
                filters=search_filters
            )
            
            # Score and rank recommendations
            recommendations = self._score_recommendations(content_results, user_metadata)
            
            # Return top recommendations
            return recommendations[:num_recommendations]
            
        except Exception as e:
            logger.error(f"Failed to get content recommendations: {e}")
            return self._get_fallback_recommendations(num_recommendations, content_filters)
    
    def get_similar_content(self, 
                           content_id: str,
                           num_recommendations: int = 5) -> List[Dict]:
        """Get content similar to a specific piece of content."""
        try:
            # Get the content embedding by searching for it
            content_results = self.vector_db.search_content(
                query_embedding=np.zeros(Config.EMBEDDING_DIMENSION),
                top_k=1,
                filters={'material_id': content_id}
            )
            
            if not content_results:
                logger.warning(f"Content {content_id} not found")
                return []
            
            # For now, return related content based on metadata
            content_metadata = content_results[0]['metadata']
            
            # Find similar content by subject area and difficulty
            filters = {
                'subject_area': content_metadata.get('subject_area'),
                'difficulty_level': content_metadata.get('difficulty_level')
            }
            
            similar_results = self.vector_db.search_content(
                query_embedding=np.zeros(Config.EMBEDDING_DIMENSION),
                top_k=num_recommendations + 1,  # +1 to exclude original
                filters=filters
            )
            
            # Remove the original content from results
            similar_content = [r for r in similar_results if r['id'] != content_id]
            
            return similar_content[:num_recommendations]
            
        except Exception as e:
            logger.error(f"Failed to get similar content: {e}")
            return []
    
    def get_user_recommendations(self, 
                               user_id: str,
                               num_recommendations: int = 5) -> List[Dict]:
        """Get recommendations for similar users or study groups."""
        try:
            # Get current user
            user_results = self.vector_db.search_users(
                query_embedding=np.zeros(Config.EMBEDDING_DIMENSION),
                top_k=1,
                filters={'user_id': user_id}
            )
            
            if not user_results:
                return []
            
            user_metadata = user_results[0]['metadata']
            
            # Find users with similar interests and major
            filters = {
                'major': user_metadata.get('major'),
                'role': 'student'  # Only recommend other students
            }
            
            similar_users = self.vector_db.search_users(
                query_embedding=np.zeros(Config.EMBEDDING_DIMENSION),
                top_k=num_recommendations + 1,
                filters=filters
            )
            
            # Remove the current user from results
            similar_users = [u for u in similar_users if u['id'] != user_id]
            
            return similar_users[:num_recommendations]
            
        except Exception as e:
            logger.error(f"Failed to get user recommendations: {e}")
            return []
    
    def update_user_preferences(self, 
                              user_id: str,
                              interaction_data: Dict) -> bool:
        """Update user preferences based on new interaction."""
        try:
            # This would normally update the user embedding in the vector database
            # For now, we'll just log the interaction
            logger.info(f"Recording interaction for user {user_id}: {interaction_data}")
            
            # In a full implementation, this would:
            # 1. Retrieve current user embedding
            # 2. Update based on interaction (viewed content, ratings, etc.)
            # 3. Re-embed the user profile with new data
            # 4. Update the vector database
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update user preferences: {e}")
            return False
    
    def _build_content_filters(self, user_metadata: Dict, additional_filters: Dict = None) -> Dict:
        """Build search filters based on user profile and additional criteria."""
        filters = {}
        
        # Filter by user's preferred difficulty if available
        preferred_difficulty = user_metadata.get('preferred_difficulty')
        if preferred_difficulty:
            filters['difficulty_level'] = preferred_difficulty
        
        # Filter by user's major/interests
        major = user_metadata.get('major')
        if major:
            # Map majors to subject areas
            major_to_subject = {
                'Computer Science': 'Computer Science',
                'Business Administration': 'Business',
                'Physics': 'Physics',
                'Art and Design': 'Art',
                'History': 'History',
                'Psychology': 'Psychology'
            }
            subject_area = major_to_subject.get(major)
            if subject_area:
                filters['subject_area'] = subject_area
        
        # Apply additional filters
        if additional_filters:
            filters.update(additional_filters)
        
        return filters
    
    def _score_recommendations(self, content_results: List[Dict], user_metadata: Dict) -> List[Dict]:
        """Score and rank recommendations based on user profile."""
        scored_recommendations = []
        
        user_interests = user_metadata.get('interests', [])
        user_difficulty = user_metadata.get('preferred_difficulty', '')
        
        for result in content_results:
            metadata = result.get('metadata', {})
            score = result.get('score', 0.0)
            
            # Boost score based on interest overlap
            content_tags = metadata.get('tags', [])
            interest_overlap = len(set(user_interests) & set(content_tags))
            score += interest_overlap * 0.1  # Boost for each matching interest
            
            # Boost score for matching difficulty preference
            if metadata.get('difficulty_level') == user_difficulty:
                score += 0.15
            
            # Boost score for popular content (if we had popularity data)
            # score += metadata.get('popularity_score', 0) * 0.05
            
            scored_recommendations.append({
                **result,
                'recommendation_score': score,
                'recommendation_reason': self._generate_recommendation_reason(metadata, user_metadata)
            })
        
        # Sort by recommendation score
        scored_recommendations.sort(key=lambda x: x['recommendation_score'], reverse=True)
        
        return scored_recommendations
    
    def _generate_recommendation_reason(self, content_metadata: Dict, user_metadata: Dict) -> str:
        """Generate a human-readable reason for the recommendation."""
        reasons = []
        
        # Check for major/subject match
        user_major = user_metadata.get('major', '')
        content_subject = content_metadata.get('subject_area', '')
        if user_major.lower() in content_subject.lower() or content_subject.lower() in user_major.lower():
            reasons.append(f"matches your {user_major} major")
        
        # Check for difficulty match
        user_difficulty = user_metadata.get('preferred_difficulty', '')
        content_difficulty = content_metadata.get('difficulty_level', '')
        if user_difficulty == content_difficulty:
            reasons.append(f"matches your preferred {content_difficulty} level")
        
        # Check for interest overlap
        user_interests = set(user_metadata.get('interests', []))
        content_tags = set(content_metadata.get('tags', []))
        common_interests = user_interests & content_tags
        if common_interests:
            reasons.append(f"aligns with your interests in {', '.join(list(common_interests)[:2])}")
        
        if not reasons:
            reasons.append("recommended based on your profile")
        
        return "Recommended because it " + " and ".join(reasons)
    
    def _get_fallback_recommendations(self, num_recommendations: int, filters: Dict = None) -> List[Dict]:
        """Get fallback recommendations when user-specific recommendations fail."""
        try:
            # Get popular/general content
            fallback_results = self.vector_db.search_content(
                query_embedding=np.zeros(Config.EMBEDDING_DIMENSION),
                top_k=num_recommendations,
                filters=filters
            )
            
            # Add generic recommendation reasons
            for result in fallback_results:
                result['recommendation_reason'] = "Popular content you might find interesting"
                result['recommendation_score'] = result.get('score', 0.5)
            
            return fallback_results
            
        except Exception as e:
            logger.error(f"Failed to get fallback recommendations: {e}")
            return []
    
    def get_trending_content(self, num_items: int = 10) -> List[Dict]:
        """Get trending/popular content across the platform."""
        try:
            # For now, return recent content across different subjects
            trending_results = self.vector_db.search_content(
                query_embedding=np.zeros(Config.EMBEDDING_DIMENSION),
                top_k=num_items
            )
            
            for result in trending_results:
                result['recommendation_reason'] = "Trending content on the platform"
                result['recommendation_score'] = result.get('score', 0.7)
            
            return trending_results
            
        except Exception as e:
            logger.error(f"Failed to get trending content: {e}")
            return []
    
    def get_learning_path_recommendations(self, user_id: str) -> List[Dict]:
        """Recommend a learning path based on user's current progress and goals."""
        try:
            # Get user profile
            user_results = self.vector_db.search_users(
                query_embedding=np.zeros(Config.EMBEDDING_DIMENSION),
                top_k=1,
                filters={'user_id': user_id}
            )
            
            if not user_results:
                return []
            
            user_metadata = user_results[0]['metadata']
            
            # Create a progression path based on difficulty
            difficulties = ['Beginner', 'Intermediate', 'Advanced']
            current_level = user_metadata.get('preferred_difficulty', 'Beginner')
            
            try:
                current_index = difficulties.index(current_level)
                next_level = difficulties[min(current_index + 1, len(difficulties) - 1)]
            except ValueError:
                next_level = 'Intermediate'
            
            # Recommend content at the next difficulty level
            path_recommendations = self.vector_db.search_content(
                query_embedding=np.zeros(Config.EMBEDDING_DIMENSION),
                top_k=5,
                filters={
                    'subject_area': user_metadata.get('major', ''),
                    'difficulty_level': next_level
                }
            )
            
            for result in path_recommendations:
                result['recommendation_reason'] = f"Next step in your learning journey to {next_level} level"
                result['recommendation_score'] = 0.8
            
            return path_recommendations
            
        except Exception as e:
            logger.error(f"Failed to get learning path recommendations: {e}")
            return []