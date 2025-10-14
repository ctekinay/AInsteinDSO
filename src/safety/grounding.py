"""
Grounding check module for citation validation.

This module ensures that responses include proper citations from authoritative sources.
Every response requires at least one valid citation from the required prefixes.

Features:
- Validates that citations actually exist in knowledge sources
- Prevents fake citations like "archi:id-cap-001", "iec:GridCongestion"
- Uses CitationValidator for bidirectional validation

Enforces the requirement that responses are grounded in authoritative sources.
"""

import logging
import re
from typing import Dict, List, Set, Optional, TYPE_CHECKING

from src.exceptions.exceptions import UngroundedReplyError
from src.utils.trace import get_tracer

# TYPE_CHECKING import to avoid circular dependencies
if TYPE_CHECKING:
    from src.safety.citation_validator import CitationValidator

logger = logging.getLogger(__name__)
tracer = get_tracer()

# REQUIRED citation prefixes - NO RESPONSE without at least one of these
REQUIRED_CITATION_PREFIXES = [
    "archi:id-",
    "skos:",
    "iec:",
    "togaf:adm:",
    "togaf:concepts:",
    "archimate:research:",
    "entsoe:",
    "lido:",
    "doc:",
    "external:"
]

# Comprehensive regex patterns for citation detection
CITATION_PATTERNS = {
    "archi:id-": r"archi:id-[a-zA-Z0-9\-_]+",
    "skos:": r"skos:[a-zA-Z0-9\-_]+",
    "iec:": r"iec:[a-zA-Z0-9\-_\.]+",
    "togaf:adm:": r"togaf:adm:[a-zA-Z0-9\-_]+",
    "togaf:concepts:": r"togaf:concepts:[0-9]{3}",
    "archimate:research:": r"archimate:research:[0-9]{3}",
    "entsoe:": r"entsoe:[a-zA-Z0-9\-_]+",
    "lido:": r"lido:[a-zA-Z0-9\-_]+",
    "doc:": r"doc:[a-zA-Z0-9\-_]+:[0-9]{3}",
    "external:": r"external:[a-zA-Z0-9\-_]+:[a-zA-Z0-9\-_]+"
}

# Enhanced patterns for comparison responses and edge cases
ENHANCED_CITATION_PATTERNS = [
    r'\[([a-zA-Z0-9\-:]+)\]',  # Square bracket format: [archi:id-cap-001]
    r'\*\*.*?\*\*\s*\[([a-zA-Z0-9\-:]+)\]',  # Bold text with citation: **Concept** [citation]
    r'`([a-zA-Z0-9\-:]+)`',  # Backtick format: `archi:id-cap-001`
    r'(?:citation|ref|source):\s*([a-zA-Z0-9\-:]+)',  # Labeled citations: citation: archi:id-cap-001
]


class GroundingCheck:
    """
    Enforces mandatory citation requirements for response validation.

    This class ensures all responses include valid citations.
    Every response must include at least one valid citation or an
    UngroundedReplyError is raised.

    Features:
    - Validates citation format
    - Validates citation existence in knowledge sources
    - Uses citation pool to prevent fake citations
    """

    def __init__(self, citation_validator: Optional['CitationValidator'] = None):
        """
        Initialize the grounding checker.

        Args:
            citation_validator: Optional CitationValidator for authenticity checks
        """
        self.required_prefixes = REQUIRED_CITATION_PREFIXES
        self.citation_patterns = CITATION_PATTERNS
        self.citation_validator = citation_validator
        
        if citation_validator:
            logger.info("GroundingCheck initialized with CitationValidator - STRICT authenticity enforcement active")
        else:
            logger.info("GroundingCheck initialized - citation format enforcement active (authenticity validation disabled)")

    def assert_citations(
        self,
        answer: str,
        retrieval_context: Dict = None,
        citation_pool: List[str] = None,
        trace_id: str = None
    ) -> Dict:
        """
        MANDATORY citation validation - NO EXCEPTIONS.

        This is the critical safety gate that prevents any ungrounded responses
        from being generated. Every response MUST have valid citations.

        ENHANCED: Now validates citation authenticity against knowledge sources.

        Args:
            answer: The response text to validate
            retrieval_context: Context from retrieval with possible citations
            citation_pool: NEW - List of valid citations from retrieval (for authenticity check)
            trace_id: Optional trace ID for logging

        Returns:
            Dict with status and citation info:
            - status: "grounded" if valid citations found
            - status: "needs_citations" if suggestions available
            - citations: List of found citations
            - suggestions: List of suggested citations if available

        Raises:
            UngroundedReplyError: If NO valid citations found (MANDATORY)
            FakeCitationError: If fake citations detected (NEW)
        """
        if trace_id:
            tracer.trace_info(trace_id, "grounding_check", "validation_start",
                            response_length=len(answer) if answer else 0,
                            has_citation_pool=citation_pool is not None)

        if not answer or not answer.strip():
            logger.error("GROUNDING VIOLATION: Empty response provided")
            raise UngroundedReplyError(
                "Empty response provided - all responses must include citations",
                self.required_prefixes
            )

        # Extract existing citations from the answer
        existing_citations = self._extract_existing_citations(answer)

        if trace_id:
            tracer.trace_info(trace_id, "grounding_check", "citations_extracted",
                            citation_count=len(existing_citations),
                            citations=existing_citations)

        # NEW: Validate citation authenticity if validator and pool available
        if existing_citations and self.citation_validator and citation_pool:
            logger.info(f"Validating {len(existing_citations)} citations against pool of {len(citation_pool)}")
            
            validation_result = self.citation_validator.validate_response_citations(
                answer, citation_pool, trace_id
            )
            
            if not validation_result["valid"]:
                fake_cits = validation_result["fake_citations"]
                logger.error(f"FAKE CITATIONS DETECTED: {fake_cits}")
                
                if trace_id:
                    tracer.trace_info(trace_id, "grounding_check", "fake_citations_detected",
                                    fake_count=len(fake_cits),
                                    fake_citations=fake_cits)
                
                # Import here to avoid circular dependency
                from src.exceptions.exceptions import FakeCitationError
                raise FakeCitationError(
                    f"Response contains {len(fake_cits)} fabricated citation(s): {', '.join(fake_cits[:3])}",
                    fake_citations=fake_cits,
                    valid_pool=citation_pool
                )
            
            logger.info(f"Citation authenticity validated: all {len(existing_citations)} citations are valid")

        # Continue with existing logic
        if existing_citations:
            logger.info(f"GROUNDING SUCCESS: Found {len(existing_citations)} citations")
            
            if trace_id:
                tracer.trace_info(trace_id, "grounding_check", "validation_success",
                                citation_count=len(existing_citations))
            
            return {
                "status": "grounded",
                "citations": existing_citations,
                "citation_count": len(existing_citations)
            }

        # NO CITATIONS FOUND - attempt to suggest from context
        suggested_citations = []
        if retrieval_context:
            suggested_citations = self.suggest_citations(retrieval_context)

        if suggested_citations:
            logger.warning(f"GROUNDING PARTIAL: No citations in response, but {len(suggested_citations)} suggestions available")
            
            if trace_id:
                tracer.trace_info(trace_id, "grounding_check", "suggestions_available",
                                suggestion_count=len(suggested_citations))
            
            return {
                "status": "needs_citations",
                "citations": [],
                "suggestions": suggested_citations,
                "message": f"Response lacks required citations. Suggested: {', '.join(suggested_citations[:3])}"
            }

        # COMPLETE FAILURE - NO CITATIONS AND NO SUGGESTIONS
        logger.error("GROUNDING VIOLATION: No citations found and no suggestions available")
        
        if trace_id:
            tracer.trace_info(trace_id, "grounding_check", "validation_failed",
                            reason="no_citations_no_suggestions")
        
        raise UngroundedReplyError(
            "Response lacks required citations. Must include at least one of: " +
            ", ".join(self.required_prefixes),
            self.required_prefixes
        )

    def suggest_citations(self, retrieval_context: Dict) -> List[str]:
        """
        Extract potential citations from retrieval context.

        Note: Only extracts citations that exist in the retrieval context.
        Does NOT generate fake citations.

        Args:
            retrieval_context: Dictionary containing retrieval results

        Returns:
            List of suggested citation strings
        """
        suggestions = set()

        if not retrieval_context:
            return []

        # Extract from various context fields
        context_fields = [
            "retrieved_terms",
            "iec_terms",
            "entsoe_terms",
            "eurlex_terms",
            "archimate_elements",
            "togaf_phases",
            "concepts",
            "sources",
            "kg_results",
            "togaf_docs",
            "document_chunks"
        ]

        for field in context_fields:
            if field in retrieval_context:
                field_data = retrieval_context[field]

                if isinstance(field_data, dict):
                    # Extract from dictionary keys and values
                    for key, value in field_data.items():
                        suggestions.update(self._extract_existing_citations(str(key)))
                        suggestions.update(self._extract_existing_citations(str(value)))

                elif isinstance(field_data, list):
                    # Extract from list items
                    for item in field_data:
                        if isinstance(item, dict):
                            # Extract citation_id if present
                            if "citation_id" in item:
                                suggestions.add(item["citation_id"])
                            # Also check other fields
                            for v in item.values():
                                suggestions.update(self._extract_existing_citations(str(v)))
                        else:
                            suggestions.update(self._extract_existing_citations(str(item)))

                else:
                    # Extract from string content
                    suggestions.update(self._extract_existing_citations(str(field_data)))

        # No fake citation suggestions
        # The old code might have generated fake citations like "archi:id-cap-001"
        # which violated the core requirement: "If you cannot ground a response, raise UngroundedReplyError - do not guess"
        #
        # We only suggest citations that actually exist in the retrieval context,
        # not hardcoded fake ones that mislead the LLM into generating false responses.

        return list(suggestions)

    def _extract_existing_citations(self, text: str) -> List[str]:
        """
        Extract all valid citations from text using regex patterns.

        Args:
            text: Text to search for citations

        Returns:
            List of found citation strings
        """
        if not text:
            return []

        citations = []

        # Apply each pattern to find citations
        for prefix, pattern in self.citation_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            citations.extend(matches)

        # Apply enhanced patterns for comparison responses and edge cases
        for pattern in ENHANCED_CITATION_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            # Filter matches to only include valid citation prefixes
            for match in matches:
                if any(match.lower().startswith(prefix) for prefix in REQUIRED_CITATION_PREFIXES):
                    citations.append(match)

        # Remove duplicates while preserving order
        seen = set()
        unique_citations = []
        for citation in citations:
            if citation.lower() not in seen:
                seen.add(citation.lower())
                unique_citations.append(citation)

        return unique_citations

    def validate_citation_format(self, citation: str) -> bool:
        """
        Validate that a citation follows the correct format.

        Note: This only validates format, not authenticity.
        Use CitationValidator.validate_citation_exists() for authenticity.

        Args:
            citation: Citation string to validate

        Returns:
            True if citation format is valid
        """
        if not citation:
            return False

        for prefix, pattern in self.citation_patterns.items():
            if re.match(f"^{pattern}$", citation, re.IGNORECASE):
                return True

        return False

    def get_citation_statistics(self, text: str) -> Dict:
        """
        Get detailed statistics about citations in text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with citation statistics
        """
        citations = self._extract_existing_citations(text)

        stats = {
            "total_citations": len(citations),
            "unique_citations": len(set(c.lower() for c in citations)),
            "citations": citations,
            "by_prefix": {}
        }

        # Count by prefix
        for prefix in self.required_prefixes:
            prefix_citations = [c for c in citations if c.lower().startswith(prefix)]
            stats["by_prefix"][prefix] = {
                "count": len(prefix_citations),
                "citations": prefix_citations
            }

        stats["is_grounded"] = len(citations) > 0

        return stats

    def enforce_grounding(
        self,
        answer: str,
        retrieval_context: Dict = None,
        citation_pool: List[str] = None,
        trace_id: str = None
    ) -> str:
        """
        Enforce grounding on a response - raises exception if not grounded.

        This is the final enforcement point that should be called before any
        response is returned to the user.

        Args:
            answer: Response to validate
            retrieval_context: Retrieval context for suggestions
            citation_pool: NEW - List of valid citations for authenticity check
            trace_id: Optional trace ID for logging

        Returns:
            The original answer if grounded

        Raises:
            UngroundedReplyError: If response is not properly grounded
            FakeCitationError: If fake citations detected
        """
        result = self.assert_citations(answer, retrieval_context, citation_pool, trace_id)

        if result["status"] == "grounded":
            logger.info(f"GROUNDING ENFORCED: Response validated with {result['citation_count']} citations")
            return answer

        elif result["status"] == "needs_citations":
            # Could return suggestions, but per requirements - MUST raise error
            logger.error("GROUNDING ENFORCEMENT FAILED: Response needs citations")
            raise UngroundedReplyError(
                f"Response must include citations. {result.get('message', '')}",
                self.required_prefixes,
                result.get("suggestions", [])
            )

        else:
            logger.error("GROUNDING ENFORCEMENT FAILED: Unknown status")
            raise UngroundedReplyError(
                "Response validation failed",
                self.required_prefixes
            )


# Factory function for backward compatibility
def create_grounding_check(citation_validator: Optional['CitationValidator'] = None) -> GroundingCheck:
    """
    Factory function to create GroundingCheck instance.

    Args:
        citation_validator: Optional CitationValidator for authenticity checks

    Returns:
        GroundingCheck instance
    """
    return GroundingCheck(citation_validator=citation_validator)