"""
Routing module for query classification and knowledge source selection.

This module implements the critical input protection layer that ensures
queries are routed to the most appropriate knowledge source before falling
back to vector search.
"""

from .query_router import QueryRouter

__all__ = ["QueryRouter"]