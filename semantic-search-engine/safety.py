"""Safety and validation utilities - Production Ready."""

import re
import logging
from typing import List, Tuple, Optional
from config import Config

logger = logging.getLogger(__name__)

class InputSanitizer:
    """Handles input validation and sanitization for LLM application safety."""
    
    def __init__(self):
        self.blocked_terms = Config.BLOCKED_TERMS
        self.max_length = Config.MAX_QUERY_LENGTH
        self.min_length = Config.MIN_QUERY_LENGTH
    
    def sanitize_query(self, query: str) -> Tuple[str, List[str]]:
        """
        Sanitize and validate user query following security best practices.
        
        Args:
            query: Raw user input
            
        Returns:
            Tuple of (cleaned_query, validation_errors)
        """
        errors = []
        
        if not query or not query.strip():
            errors.append("Query cannot be empty")
            return "", errors
        
        # Strip and normalize whitespace
        clean_query = query.strip()
        
        # Length validation
        if len(clean_query) < self.min_length:
            errors.append(f"Query must be at least {self.min_length} characters")
        
        if len(clean_query) > self.max_length:
            errors.append(f"Query too long (max {self.max_length} characters)")
            clean_query = clean_query[:self.max_length]
        
        # Content filtering for safety
        if Config.ENABLE_CONTENT_FILTER:
            if self._contains_blocked_content(clean_query):
                errors.append("Query contains inappropriate content")
        
        # Remove potentially dangerous characters
        clean_query = self._sanitize_text(clean_query)
        
        return clean_query, errors
    
    def _contains_blocked_content(self, text: str) -> bool:
        """Check for blocked terms."""
        text_lower = text.lower()
        return any(term in text_lower for term in self.blocked_terms)
    
    def _sanitize_text(self, text: str) -> str:
        """Remove potentially dangerous characters while preserving readability."""
        # Remove control characters but preserve basic punctuation
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

class ContentFilter:
    """Filters and validates search results for safety."""
    
    def __init__(self):
        self.max_content_display = 300
    
    def filter_results(self, results: List[dict]) -> List[dict]:
        """Filter and sanitize search results."""
        filtered = []
        
        for result in results:
            try:
                if self._validate_result(result):
                    filtered_result = self._sanitize_result(result)
                    filtered.append(filtered_result)
            except Exception as e:
                logger.warning(f"Error filtering result: {e}")
                continue
        
        return filtered
    
    def _validate_result(self, result: dict) -> bool:
        """Validate result structure."""
        required_fields = ['id', 'score']
        return all(field in result for field in required_fields)
    
    def _sanitize_result(self, result: dict) -> dict:
        """Sanitize individual result for safe display."""
        sanitized = result.copy()
        
        # Ensure score is numeric
        try:
            sanitized['score'] = float(result.get('score', 0.0))
        except (ValueError, TypeError):
            sanitized['score'] = 0.0
        
        # Sanitize metadata content
        if 'metadata' in result and 'content' in result['metadata']:
            content = str(result['metadata']['content'])
            if len(content) > self.max_content_display:
                sanitized['metadata']['content'] = content[:self.max_content_display] + "..."
        
        return sanitized

# Global instances
input_sanitizer = InputSanitizer()
content_filter = ContentFilter()