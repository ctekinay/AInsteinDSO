"""
Validation module for confidence assessment and quality control.

This module implements the Critic system that prevents overconfident wrong answers
by enforcing strict confidence thresholds and identifying irrelevant suggestions.
"""

from .critic import Critic, CriticAssessment

__all__ = ["Critic", "CriticAssessment"]