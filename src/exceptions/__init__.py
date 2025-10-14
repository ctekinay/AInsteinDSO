# src/exceptions/__init__.py
"""Exception classes for EA Assistant."""

class KnowledgeGraphError(Exception):
    """Knowledge graph related errors."""
    pass

class UngroundedReplyError(Exception):
    """Response lacks required citations."""
    def __init__(self, message: str, required_prefixes=None):
        self.required_prefixes = required_prefixes or []
        super().__init__(message)

class FakeCitationError(Exception):
    """Fake citation detected."""
    def __init__(self, message: str, fake_citations=None, valid_pool=None):
        self.fake_citations = fake_citations or []
        self.valid_pool = valid_pool or []
        super().__init__(message)

class LowConfidenceError(Exception):
    """Confidence below threshold."""
    pass

class PerformanceError(Exception):
    """Performance threshold exceeded."""
    pass