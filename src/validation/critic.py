"""
Critic Module - Confidence Assessment and Irrelevance Detection.

This module addresses the research finding that LLMs are overconfident about
irrelevant elements. It prevents false positives by enforcing strict confidence
thresholds and explicitly identifying irrelevant suggestions.

Based on research showing 18% irrelevance rate in human experts, the Critic:
1. Ranks suggestions by confidence
2. Identifies irrelevant items with reasoning
3. Forces human review when confidence < 0.75
4. Enforces 18% irrelevance budget to prevent over-filtering

This ensures the system admits uncertainty rather than providing overconfident
wrong answers.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CriticAssessment:
    """
    Assessment result from the Critic module.

    This dataclass captures the complete confidence assessment including
    top suggestions, irrelevant items, and human review requirements.
    """
    top_suggestions: List[Dict]         # Top 3 suggestions with confidence scores
    irrelevant_items: List[Dict]        # Explicitly excluded items with reasoning
    confidence: float                   # Overall confidence score (0.0-1.0)
    citations: List[str]               # Required citations for suggestions
    requires_human_review: bool        # True if confidence < threshold
    reasoning: str                     # Explanation of assessment decision

    def __post_init__(self):
        """Validate assessment data after initialization."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")

        if len(self.top_suggestions) > 3:
            logger.warning(f"Top suggestions should be limited to 3, got {len(self.top_suggestions)}")


class Critic:
    """
    Confidence assessment and irrelevance detection system.

    The Critic prevents overconfident wrong answers by:
    1. Enforcing strict confidence thresholds (0.75)
    2. Identifying irrelevant elements (max 18% budget)
    3. Forcing human review for uncertain cases
    4. Providing clear reasoning for decisions
    """

    # Research-based thresholds
    CONFIDENCE_THRESHOLD = 0.75     # Below this requires human review
    IRRELEVANCE_BUDGET = 0.18       # Max 18% can be marked irrelevant
    MAX_TOP_SUGGESTIONS = 3         # Limit to top 3 for clarity

    def __init__(self):
        """Initialize the Critic with research-based parameters."""
        self.confidence_threshold = self.CONFIDENCE_THRESHOLD
        self.irrelevance_budget = self.IRRELEVANCE_BUDGET
        self.max_top_suggestions = self.MAX_TOP_SUGGESTIONS

        logger.info(f"Critic initialized with confidence_threshold={self.confidence_threshold}, "
                   f"irrelevance_budget={self.irrelevance_budget}")

    def assess(self, candidates: List[Dict], context: Optional[Dict] = None) -> CriticAssessment:
        """
        Assess confidence and identify irrelevant elements from candidates.

        Args:
            candidates: List of candidate elements with confidence scores
                       Each candidate should have: {'element': str, 'confidence': float, 'citations': List[str]}
            context: Optional context for assessment (retrieval context, query, etc.)

        Returns:
            CriticAssessment with top suggestions, irrelevant items, and review requirements

        Raises:
            ValueError: If candidates format is invalid
        """
        if not candidates:
            logger.warning("No candidates provided for assessment")
            return CriticAssessment(
                top_suggestions=[],
                irrelevant_items=[],
                confidence=0.0,
                citations=[],
                requires_human_review=True,
                reasoning="No candidates available for assessment - human review required"
            )

        # Validate candidate format
        self._validate_candidates(candidates)

        # Sort candidates by confidence (highest first)
        sorted_candidates = sorted(candidates, key=lambda x: x['confidence'], reverse=True)

        # Select top suggestions (up to 3)
        top_suggestions = sorted_candidates[:self.max_top_suggestions]

        # Identify irrelevant items within budget
        irrelevant_items = self._identify_irrelevant_items(sorted_candidates)

        # Calculate overall confidence (average of top suggestions)
        overall_confidence = self._calculate_overall_confidence(top_suggestions)

        # Determine if human review is required
        requires_human_review = overall_confidence < self.confidence_threshold

        # Collect all citations from top suggestions
        all_citations = []
        for suggestion in top_suggestions:
            all_citations.extend(suggestion.get('citations', []))
        unique_citations = list(set(all_citations))

        # Generate reasoning
        reasoning = self._generate_reasoning(
            len(candidates), len(top_suggestions), len(irrelevant_items),
            overall_confidence, requires_human_review
        )

        assessment = CriticAssessment(
            top_suggestions=top_suggestions,
            irrelevant_items=irrelevant_items,
            confidence=overall_confidence,
            citations=unique_citations,
            requires_human_review=requires_human_review,
            reasoning=reasoning
        )

        # Log assessment decision
        self._log_assessment(assessment)

        return assessment

    def _validate_candidates(self, candidates: List[Dict]) -> None:
        """
        Validate candidate format and required fields.

        Args:
            candidates: List of candidate dictionaries

        Raises:
            ValueError: If candidate format is invalid
        """
        required_fields = ['element', 'confidence']

        for i, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                raise ValueError(f"Candidate {i} must be a dictionary, got {type(candidate)}")

            for field in required_fields:
                if field not in candidate:
                    raise ValueError(f"Candidate {i} missing required field: {field}")

            confidence = candidate['confidence']
            if not isinstance(confidence, (int, float)) or not 0.0 <= confidence <= 1.0:
                raise ValueError(f"Candidate {i} confidence must be float between 0.0-1.0, got {confidence}")

    def _identify_irrelevant_items(self, sorted_candidates: List[Dict]) -> List[Dict]:
        """
        Identify irrelevant items within the 18% budget.

        Args:
            sorted_candidates: Candidates sorted by confidence (highest first)

        Returns:
            List of irrelevant items with reasoning
        """
        total_candidates = len(sorted_candidates)
        max_irrelevant = int(total_candidates * self.irrelevance_budget)

        if max_irrelevant == 0:
            logger.debug("No irrelevant items allowed within budget")
            return []

        # Identify lowest confidence items as potentially irrelevant
        # Start from the end (lowest confidence) and work backwards
        irrelevant_items = []

        for candidate in reversed(sorted_candidates):
            if len(irrelevant_items) >= max_irrelevant:
                break

            # Mark as irrelevant if confidence is very low (< 0.3) or if we need to fill budget
            if candidate['confidence'] < 0.3:
                irrelevant_item = candidate.copy()
                irrelevant_item['irrelevance_reason'] = (
                    f"Low confidence ({candidate['confidence']:.2f}) indicates irrelevance"
                )
                irrelevant_items.append(irrelevant_item)
                logger.debug(f"Marked {candidate['element']} as irrelevant (confidence: {candidate['confidence']:.2f})")

        logger.info(f"Identified {len(irrelevant_items)}/{max_irrelevant} irrelevant items within {self.irrelevance_budget:.0%} budget")
        return irrelevant_items

    def _calculate_overall_confidence(self, top_suggestions: List[Dict]) -> float:
        """
        Calculate overall confidence from top suggestions.

        Args:
            top_suggestions: List of top suggestion dictionaries

        Returns:
            Overall confidence score (0.0-1.0)
        """
        if not top_suggestions:
            return 0.0

        total_confidence = sum(suggestion['confidence'] for suggestion in top_suggestions)
        overall_confidence = total_confidence / len(top_suggestions)

        logger.debug(f"Overall confidence: {overall_confidence:.3f} (average of {len(top_suggestions)} top suggestions)")
        return round(overall_confidence, 3)

    def _generate_reasoning(self, total_candidates: int, top_count: int, irrelevant_count: int,
                          confidence: float, requires_review: bool) -> str:
        """
        Generate human-readable reasoning for the assessment.

        Args:
            total_candidates: Total number of input candidates
            top_count: Number of top suggestions selected
            irrelevant_count: Number of irrelevant items identified
            confidence: Overall confidence score
            requires_review: Whether human review is required

        Returns:
            Reasoning string explaining the assessment
        """
        reasoning_parts = [
            f"Assessed {total_candidates} candidates, selected top {top_count} suggestions"
        ]

        if irrelevant_count > 0:
            irrelevance_rate = irrelevant_count / total_candidates
            reasoning_parts.append(
                f"Identified {irrelevant_count} irrelevant items ({irrelevance_rate:.1%} of total)"
            )

        reasoning_parts.append(f"Overall confidence: {confidence:.3f}")

        if requires_review:
            reasoning_parts.append(
                f"Confidence below threshold ({self.confidence_threshold}) - human review required"
            )
        else:
            reasoning_parts.append("Confidence sufficient for automated processing")

        return ". ".join(reasoning_parts) + "."

    def _log_assessment(self, assessment: CriticAssessment) -> None:
        """
        Log assessment results for monitoring and debugging.

        Args:
            assessment: The CriticAssessment to log
        """
        if assessment.requires_human_review:
            logger.warning(f"HUMAN REVIEW REQUIRED: Confidence {assessment.confidence:.3f} "
                         f"below threshold {self.confidence_threshold}")
        else:
            logger.info(f"ASSESSMENT PASSED: Confidence {assessment.confidence:.3f} "
                       f"meets threshold {self.confidence_threshold}")

        logger.info(f"Top suggestions: {len(assessment.top_suggestions)}, "
                   f"Irrelevant items: {len(assessment.irrelevant_items)}, "
                   f"Citations: {len(assessment.citations)}")

    def update_thresholds(self, confidence_threshold: Optional[float] = None,
                         irrelevance_budget: Optional[float] = None) -> None:
        """
        Update assessment thresholds (for testing or tuning).

        Args:
            confidence_threshold: New confidence threshold (0.0-1.0)
            irrelevance_budget: New irrelevance budget (0.0-1.0)
        """
        if confidence_threshold is not None:
            if not 0.0 <= confidence_threshold <= 1.0:
                raise ValueError(f"Confidence threshold must be 0.0-1.0, got {confidence_threshold}")
            self.confidence_threshold = confidence_threshold
            logger.info(f"Updated confidence threshold to {confidence_threshold}")

        if irrelevance_budget is not None:
            if not 0.0 <= irrelevance_budget <= 1.0:
                raise ValueError(f"Irrelevance budget must be 0.0-1.0, got {irrelevance_budget}")
            self.irrelevance_budget = irrelevance_budget
            logger.info(f"Updated irrelevance budget to {irrelevance_budget}")

    def get_assessment_stats(self) -> Dict:
        """
        Get current assessment configuration and statistics.

        Returns:
            Dictionary with assessment statistics
        """
        return {
            "confidence_threshold": self.confidence_threshold,
            "irrelevance_budget": self.irrelevance_budget,
            "max_top_suggestions": self.max_top_suggestions,
            "thresholds_source": "Research-based (LLM-KG paper findings)"
        }