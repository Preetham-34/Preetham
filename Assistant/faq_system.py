"""FAQ system with semantic search and chatbot functionality."""

import logging
import numpy as np
from typing import List, Dict, Any, Optional
from config import Config

logger = logging.getLogger(__name__)

class FAQSystem:
    """Intelligent FAQ system using vector search for accurate answer retrieval."""
    
    def __init__(self, vector_db_manager, embedding_generator):
        """Initialize FAQ system with vector database and embedding generator."""
        self.vector_db = vector_db_manager
        self.embedding_gen = embedding_generator
    
    def search_faq(self, 
                   query: str,
                   top_k: int = 3,
                   confidence_threshold: float = 0.7) -> List[Dict]:
        """Search FAQ database for relevant answers."""
        try:
            # Generate query embedding
            query_embedding = self.embedding_gen.generate_query_embedding(query)
            
            # Search FAQ index
            results = self.vector_db.search_faq(
                query_embedding=query_embedding,
                top_k=top_k
            )
            
            # Filter by confidence threshold
            confident_results = []
            for result in results:
                if result.get('score', 0) >= confidence_threshold:
                    confident_results.append({
                        **result,
                        'confidence': self._calculate_confidence(result.get('score', 0)),
                        'formatted_answer': self._format_answer(result)
                    })
            
            logger.info(f"FAQ search for '{query}' returned {len(confident_results)} confident results")
            return confident_results
            
        except Exception as e:
            logger.error(f"FAQ search failed: {e}")
            return []
    
    def get_answer(self, query: str) -> Dict[str, Any]:
        """Get the best answer for a query with confidence scoring."""
        try:
            results = self.search_faq(query, top_k=1, confidence_threshold=0.5)
            
            if not results:
                return {
                    'found': False,
                    'message': "I couldn't find a specific answer to your question. Please try rephrasing or contact support for assistance.",
                    'suggestions': self._get_related_faqs(query)
                }
            
            best_result = results[0]
            metadata = best_result.get('metadata', {})
            
            return {
                'found': True,
                'question': metadata.get('question', ''),
                'answer': metadata.get('answer', ''),
                'category': metadata.get('category', ''),
                'confidence': best_result.get('confidence', 'medium'),
                'faq_id': best_result.get('id', ''),
                'related_questions': self._get_related_questions(query, best_result.get('id', ''))
            }
            
        except Exception as e:
            logger.error(f"Failed to get answer: {e}")
            return {
                'found': False,
                'message': "An error occurred while searching for answers. Please try again.",
                'suggestions': []
            }
    
    def get_category_faqs(self, category: str, limit: int = 10) -> List[Dict]:
        """Get all FAQs in a specific category."""
        try:
            results = self.vector_db.search_faq(
                query_embedding=np.zeros(Config.EMBEDDING_DIMENSION),  # Get all
                top_k=limit,
                filters={'category': category}
            )
            
            formatted_results = []
            for result in results:
                metadata = result.get('metadata', {})
                formatted_results.append({
                    'faq_id': result.get('id', ''),
                    'question': metadata.get('question', ''),
                    'answer': metadata.get('answer', ''),
                    'category': metadata.get('category', ''),
                    'keywords': metadata.get('keywords', []),
                    'popularity_score': metadata.get('popularity_score', 0)
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to get category FAQs: {e}")
            return []
    
    def get_popular_faqs(self, limit: int = 5) -> List[Dict]:
        """Get most popular/frequently accessed FAQs."""
        try:
            # Search all FAQs and sort by popularity
            results = self.vector_db.search_faq(
                query_embedding=np.zeros(Config.EMBEDDING_DIMENSION),
                top_k=limit * 2  # Get more to sort by popularity
            )
            
            # Sort by popularity score
            sorted_results = sorted(
                results, 
                key=lambda x: x.get('metadata', {}).get('popularity_score', 0), 
                reverse=True
            )
            
            formatted_results = []
            for result in sorted_results[:limit]:
                metadata = result.get('metadata', {})
                formatted_results.append({
                    'faq_id': result.get('id', ''),
                    'question': metadata.get('question', ''),
                    'answer': metadata.get('answer', ''),
                    'category': metadata.get('category', ''),
                    'popularity_score': metadata.get('popularity_score', 0)
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to get popular FAQs: {e}")
            return []
    
    def suggest_questions(self, partial_query: str, limit: int = 5) -> List[str]:
        """Suggest FAQ questions based on partial user input."""
        try:
            if len(partial_query) < 3:  # Too short to search
                return []
            
            # Search for similar questions
            query_embedding = self.embedding_gen.generate_query_embedding(partial_query)
            results = self.vector_db.search_faq(
                query_embedding=query_embedding,
                top_k=limit
            )
            
            suggestions = []
            for result in results:
                metadata = result.get('metadata', {})
                question = metadata.get('question', '')
                if question and question not in suggestions:
                    suggestions.append(question)
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Failed to get question suggestions: {e}")
            return []
    
    def chatbot_response(self, user_message: str, conversation_history: List[Dict] = None) -> Dict:
        """Generate chatbot response with context awareness."""
        try:
            # Clean and prepare the user message
            query = user_message.strip()
            
            if not query:
                return {
                    'response': "Hello! How can I help you today? You can ask me questions about courses, technical issues, or anything else related to the platform.",
                    'type': 'greeting',
                    'suggestions': ["How do I reset my password?", "How do I access my courses?", "How do I get a certificate?"]
                }
            
            # Check for greeting patterns
            greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'help']
            if any(greeting in query.lower() for greeting in greetings) and len(query.split()) <= 3:
                return {
                    'response': "Hello! I'm here to help you with any questions about the learning platform. What would you like to know?",
                    'type': 'greeting',
                    'suggestions': self._get_popular_questions()
                }
            
            # Search for answers
            answer_data = self.get_answer(query)
            
            if answer_data['found']:
                response = f"**{answer_data['question']}**\n\n{answer_data['answer']}"
                
                # Add follow-up suggestions if confidence is not very high
                suggestions = []
                if answer_data.get('confidence') != 'high':
                    suggestions = [q for q in answer_data.get('related_questions', [])[:3]]
                
                return {
                    'response': response,
                    'type': 'answer',
                    'confidence': answer_data.get('confidence'),
                    'category': answer_data.get('category'),
                    'suggestions': suggestions
                }
            else:
                return {
                    'response': answer_data['message'],
                    'type': 'no_answer',
                    'suggestions': answer_data.get('suggestions', [])
                }
                
        except Exception as e:
            logger.error(f"Chatbot response failed: {e}")
            return {
                'response': "I'm sorry, I encountered an error while processing your question. Please try again or contact support if the issue persists.",
                'type': 'error',
                'suggestions': []
            }
    
    def _calculate_confidence(self, similarity_score: float) -> str:
        """Convert similarity score to confidence level."""
        if similarity_score >= 0.85:
            return 'high'
        elif similarity_score >= 0.70:
            return 'medium'
        else:
            return 'low'
    
    def _format_answer(self, result: Dict) -> str:
        """Format FAQ answer for display."""
        metadata = result.get('metadata', {})
        answer = metadata.get('answer', '')
        
        # Add category prefix if helpful
        category = metadata.get('category', '')
        if category and category not in answer:
            return f"[{category}] {answer}"
        
        return answer
    
    def _get_related_faqs(self, query: str, limit: int = 3) -> List[str]:
        """Get related FAQ questions for suggestions."""
        try:
            results = self.search_faq(query, top_k=limit + 2, confidence_threshold=0.4)
            
            suggestions = []
            for result in results[:limit]:
                metadata = result.get('metadata', {})
                question = metadata.get('question', '')
                if question:
                    suggestions.append(question)
            
            return suggestions
            
        except Exception:
            return []
    
    def _get_related_questions(self, query: str, exclude_id: str, limit: int = 3) -> List[str]:
        """Get related questions, excluding the current one."""
        try:
            results = self.search_faq(query, top_k=limit + 3, confidence_threshold=0.4)
            
            related = []
            for result in results:
                if result.get('id') != exclude_id:
                    metadata = result.get('metadata', {})
                    question = metadata.get('question', '')
                    if question and len(related) < limit:
                        related.append(question)
            
            return related
            
        except Exception:
            return []
    
    def _get_popular_questions(self, limit: int = 3) -> List[str]:
        """Get popular questions for suggestions."""
        try:
            popular_faqs = self.get_popular_faqs(limit)
            return [faq['question'] for faq in popular_faqs if faq.get('question')]
        except Exception:
            return ["How do I reset my password?", "How do I access my courses?", "How do I get help?"]
    
    def get_categories(self) -> List[str]:
        """Get all available FAQ categories."""
        try:
            # This would ideally query the database for unique categories
            # For now, return common categories
            return [
                "Technical Support",
                "Course Access", 
                "Video Playback",
                "Certificates",
                "Payment & Refunds",
                "Mobile App",
                "Account Management",
                "Assignments",
                "Discussion Forums",
                "Grading",
                "Course Selection",
                "Accessibility",
                "Study Tips",
                "Group Projects"
            ]
        except Exception as e:
            logger.error(f"Failed to get categories: {e}")
            return []
    
    def record_feedback(self, faq_id: str, user_query: str, helpful: bool) -> bool:
        """Record user feedback on FAQ helpfulness."""
        try:
            # In a full implementation, this would update the FAQ popularity score
            # and potentially trigger retraining of the search model
            logger.info(f"FAQ feedback recorded: {faq_id} was {'helpful' if helpful else 'not helpful'} for query '{user_query}'")
            return True
        except Exception as e:
            logger.error(f"Failed to record feedback: {e}")
            return False