"""
Custom exceptions for the Alliander EA Assistant.

This module defines all custom exceptions used throughout the application
to provide clear error handling and debugging information.
"""

from datetime import datetime
from typing import List, Optional


class UngroundedReplyError(Exception):
    """
    Raised when a response lacks required citations.

    This safety exception prevents ungrounded responses
    from being generated. Every response requires at least one valid
    citation from the required prefixes.
    """
    
    def __init__(self, message: str, required_prefixes: List[str], suggestions: List[str] = None):
        """
        Initialize ungrounded reply error.
        
        Args:
            message: Error message
            required_prefixes: List of required citation prefixes
            suggestions: Optional list of suggested citations
        """
        super().__init__(message)
        self.required_prefixes = required_prefixes
        self.suggestions = suggestions or []
        self.timestamp = datetime.utcnow().isoformat()
    
    def __str__(self):
        base = super().__str__()
        if self.suggestions:
            return f"{base} | Suggestions: {', '.join(self.suggestions[:3])}"
        return base


class FakeCitationError(Exception):
    """
    Raised when response contains fabricated citations not in knowledge sources.
    
    This critical exception prevents hallucinated citations from reaching users.
    It is raised when the system detects citations that don't exist in the
    knowledge graph, ArchiMate models, or TOGAF documents.
    
    Examples of fake citations:
    - archi:id-cap-001 (doesn't exist in loaded models)
    - iec:GridCongestion (not in knowledge graph)
    - iec:61968 (standard number, not a concept URI)
    """
    
    def __init__(self, message: str, fake_citations: List[str], valid_pool: List[str]):
        """
        Initialize fake citation error.
        
        Args:
            message: Error message
            fake_citations: List of fake citations detected
            valid_pool: List of valid citations that should have been used
        """
        super().__init__(message)
        self.fake_citations = fake_citations
        self.valid_pool = valid_pool
        self.timestamp = datetime.utcnow().isoformat()
    
    def __str__(self):
        return (f"{super().__str__()} | "
                f"Fake citations: {', '.join(self.fake_citations[:5])} | "
                f"Valid pool size: {len(self.valid_pool)}")


class LowConfidenceError(Exception):
    """
    Raised when response confidence falls below required threshold.
    
    This exception triggers human review when the system is not confident
    enough in its response. The default threshold is 0.75.
    """
    
    def __init__(self, message: str, confidence: float, threshold: float = 0.75):
        """
        Initialize low confidence error.
        
        Args:
            message: Error message
            confidence: Actual confidence score
            threshold: Required confidence threshold
        """
        super().__init__(message)
        self.confidence = confidence
        self.threshold = threshold
        self.timestamp = datetime.utcnow().isoformat()
    
    def __str__(self):
        return f"{super().__str__()} | Confidence: {self.confidence:.2f} | Threshold: {self.threshold:.2f}"


class KnowledgeGraphError(Exception):
    """
    Raised when knowledge graph operations fail.
    
    This exception is raised for errors related to knowledge graph loading,
    querying, or validation.
    """
    
    def __init__(self, message: str, graph_path: Optional[str] = None):
        """
        Initialize knowledge graph error.
        
        Args:
            message: Error message
            graph_path: Optional path to the graph file
        """
        super().__init__(message)
        self.graph_path = graph_path
        self.timestamp = datetime.utcnow().isoformat()


class PerformanceError(Exception):
    """
    Raised when performance SLOs are violated.
    
    This exception is raised when operations exceed their performance targets,
    such as SPARQL queries taking longer than 300ms.
    """
    
    def __init__(self, message: str, actual_time_ms: float, threshold_ms: float):
        """
        Initialize performance error.
        
        Args:
            message: Error message
            actual_time_ms: Actual execution time in milliseconds
            threshold_ms: Performance threshold in milliseconds
        """
        super().__init__(message)
        self.actual_time_ms = actual_time_ms
        self.threshold_ms = threshold_ms
        self.timestamp = datetime.utcnow().isoformat()
    
    def __str__(self):
        return f"{super().__str__()} | Actual: {self.actual_time_ms:.1f}ms | Threshold: {self.threshold_ms:.1f}ms"


class InvalidModelChangeError(Exception):
    """
    Raised when attempting to modify models directly.
    
    All model changes must go through Pull Request workflow, not direct modification.
    This exception enforces that architectural governance requirement.
    """
    
    def __init__(self, message: str, model_path: str, attempted_change: str):
        """
        Initialize invalid model change error.
        
        Args:
            message: Error message
            model_path: Path to the model being modified
            attempted_change: Description of the attempted change
        """
        super().__init__(message)
        self.model_path = model_path
        self.attempted_change = attempted_change
        self.timestamp = datetime.utcnow().isoformat()


class RouterError(Exception):
    """
    Raised when query routing fails.
    
    This exception is raised when the router cannot determine the appropriate
    knowledge source for a query.
    """
    
    def __init__(self, message: str, query: str, attempted_routes: List[str] = None):
        """
        Initialize router error.
        
        Args:
            message: Error message
            query: The query that failed to route
            attempted_routes: List of routes that were attempted
        """
        super().__init__(message)
        self.query = query
        self.attempted_routes = attempted_routes or []
        self.timestamp = datetime.utcnow().isoformat()


class ArchiMateParseError(Exception):
    """
    Raised when ArchiMate model parsing fails.
    
    This exception is raised for errors during XML parsing or model validation.
    """
    
    def __init__(self, message: str, model_path: str, line_number: Optional[int] = None):
        """
        Initialize ArchiMate parse error.
        
        Args:
            message: Error message
            model_path: Path to the ArchiMate model file
            line_number: Optional line number where error occurred
        """
        super().__init__(message)
        self.model_path = model_path
        self.line_number = line_number
        self.timestamp = datetime.utcnow().isoformat()


class TOGAFValidationError(Exception):
    """
    Raised when TOGAF compliance validation fails.
    
    This exception is raised when architectural elements don't align with
    TOGAF ADM phase requirements.
    """
    
    def __init__(
        self,
        message: str,
        element_id: str,
        phase: str,
        rule_id: str,
        severity: str = "error"
    ):
        """
        Initialize TOGAF validation error.
        
        Args:
            message: Error message
            element_id: ID of the element that failed validation
            phase: TOGAF ADM phase
            rule_id: ID of the violated rule
            severity: Severity level (error, warning, info)
        """
        super().__init__(message)
        self.element_id = element_id
        self.phase = phase
        self.rule_id = rule_id
        self.severity = severity
        self.timestamp = datetime.utcnow().isoformat()


class LLMProviderError(Exception):
    """
    Raised when LLM provider operations fail.
    
    This exception is raised for errors related to LLM API calls,
    authentication, or response parsing.
    """
    
    def __init__(self, message: str, provider_name: str, error_code: Optional[str] = None):
        """
        Initialize LLM provider error.
        
        Args:
            message: Error message
            provider_name: Name of the LLM provider (groq, ollama, openai)
            error_code: Optional error code from the provider
        """
        super().__init__(message)
        self.provider_name = provider_name
        self.error_code = error_code
        self.timestamp = datetime.utcnow().isoformat()


class CitationValidationError(Exception):
    """
    Raised when citation validation fails.
    
    This is a general citation validation error, distinct from FakeCitationError.
    Used for validation logic failures rather than fake citation detection.
    """
    
    def __init__(self, message: str, citation: str, reason: str):
        """
        Initialize citation validation error.
        
        Args:
            message: Error message
            citation: The citation that failed validation
            reason: Reason for validation failure
        """
        super().__init__(message)
        self.citation = citation
        self.reason = reason
        self.timestamp = datetime.utcnow().isoformat()


# Convenience function for error logging
def log_exception(exception: Exception, logger, context: dict = None):
    """
    Log exception with full context.
    
    Args:
        exception: Exception to log
        logger: Logger instance
        context: Optional context dictionary
    """
    error_type = type(exception).__name__
    error_msg = str(exception)
    
    log_data = {
        "error_type": error_type,
        "error_message": error_msg,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if context:
        log_data.update(context)
    
    # Add exception-specific attributes
    if hasattr(exception, 'timestamp'):
        log_data["exception_timestamp"] = exception.timestamp
    
    if isinstance(exception, FakeCitationError):
        log_data["fake_citations"] = exception.fake_citations
        log_data["valid_pool_size"] = len(exception.valid_pool)
    
    elif isinstance(exception, UngroundedReplyError):
        log_data["required_prefixes"] = exception.required_prefixes
        log_data["suggestions_count"] = len(exception.suggestions)
    
    elif isinstance(exception, LowConfidenceError):
        log_data["confidence"] = exception.confidence
        log_data["threshold"] = exception.threshold
    
    elif isinstance(exception, PerformanceError):
        log_data["actual_time_ms"] = exception.actual_time_ms
        log_data["threshold_ms"] = exception.threshold_ms
    
    logger.error(f"Exception occurred: {error_type}", extra=log_data)