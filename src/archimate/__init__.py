"""
ArchiMate module for parsing model files and enabling real citations.

This module provides the ArchiMate parser that extracts elements from XML files
to enable real citations (archi:id-xxx) and TOGAF phase alignment validation.
"""

from .parser import ArchiMateParser, ArchiMateElement

__all__ = ["ArchiMateParser", "ArchiMateElement"]