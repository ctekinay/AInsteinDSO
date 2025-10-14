"""
Comprehensive unit tests for the CRITICAL grounding check module.

These tests ensure that the GroundingCheck enforces the #1 safety requirement:
NO RESPONSE can EVER be generated without proper citations.
"""

import pytest

from src.exceptions.exceptions import UngroundedReplyError
from src.safety.grounding import GroundingCheck, REQUIRED_CITATION_PREFIXES


class TestGroundingCheck:
    """
    Test suite for the CRITICAL GroundingCheck safety component.

    These tests verify ZERO TOLERANCE for ungrounded responses.
    """

    def setup_method(self):
        """Set up test fixtures."""
        self.grounding = GroundingCheck()

    def test_initialization(self):
        """Test GroundingCheck initializes with required prefixes."""
        assert self.grounding.required_prefixes == REQUIRED_CITATION_PREFIXES
        assert len(self.grounding.citation_patterns) == len(REQUIRED_CITATION_PREFIXES)

    # TEST CASE 1: CRITICAL - Must raise UngroundedReplyError for uncited text
    def test_uncited_response_raises_error(self):
        """TEST CASE 1: Uncited response MUST raise UngroundedReplyError."""
        uncited_response = "You should use a Business Capability"

        with pytest.raises(UngroundedReplyError) as exc_info:
            self.grounding.assert_citations(uncited_response)

        error = exc_info.value
        assert "lacks required citations" in str(error)
        assert all(prefix in str(error) for prefix in REQUIRED_CITATION_PREFIXES)

    # TEST CASE 2: CRITICAL - Must pass for properly cited responses
    def test_properly_cited_response_passes(self):
        """TEST CASE 2: Properly cited response MUST pass validation."""
        cited_response = "Based on IEC 61968 (iec:GridCongestion), use Capability (archi:id-cap-001)"

        result = self.grounding.assert_citations(cited_response)

        assert result["status"] == "grounded"
        assert len(result["citations"]) >= 2
        assert any("iec:" in citation for citation in result["citations"])
        assert any("archi:id-" in citation for citation in result["citations"])

    # TEST CASE 3: CRITICAL - Empty response must raise error
    def test_empty_response_raises_error(self):
        """TEST CASE 3: Empty response MUST raise UngroundedReplyError."""
        empty_responses = ["", "   ", "\n\t  ", None]

        for empty_response in empty_responses:
            with pytest.raises(UngroundedReplyError) as exc_info:
                self.grounding.assert_citations(empty_response or "")

            assert "Empty response" in str(exc_info.value)

    # TEST CASE 4: Suggestions when context available
    def test_suggestions_when_context_available(self):
        """TEST CASE 4: Should return suggestions when retrieval context available."""
        uncited_response = "You should model this using ArchiMate"
        context = {
            "retrieved_terms": {"iec:ActivePower": "Active Power"},
            "archimate_elements": ["archi:id-cap-001", "archi:id-proc-001"]
        }

        result = self.grounding.assert_citations(uncited_response, context)

        assert result["status"] == "needs_citations"
        assert len(result["suggestions"]) > 0
        assert any("iec:" in suggestion for suggestion in result["suggestions"])

    # TEST CASE 5: Multiple citation types extraction
    def test_multiple_citation_types_extraction(self):
        """TEST CASE 5: Must extract all citation types correctly."""
        multi_cited_response = """
        According to TOGAF Phase B (togaf:adm:B), implement Business Capability
        (archi:id-cap-001) for IEC equipment (iec:Transformer) using SKOS concept
        (skos:Concept) and ENTSOE model (entsoe:GridCongestion) with LIDO mapping (lido:event-001).
        """

        result = self.grounding.assert_citations(multi_cited_response)

        assert result["status"] == "grounded"
        citations = result["citations"]
        assert len(citations) >= 6

        # Verify each prefix type is found
        prefix_found = {prefix: False for prefix in REQUIRED_CITATION_PREFIXES}
        for citation in citations:
            for prefix in REQUIRED_CITATION_PREFIXES:
                if citation.lower().startswith(prefix):
                    prefix_found[prefix] = True

        # All prefixes should be found in this test
        assert all(prefix_found.values()), f"Missing prefixes: {[k for k, v in prefix_found.items() if not v]}"

    def test_citation_extraction_edge_cases(self):
        """Test citation extraction handles edge cases correctly."""
        test_cases = [
            ("IEC:ActivePower", ["IEC:ActivePower"]),  # Uppercase
            ("iec:power-flow", ["iec:power-flow"]),     # Hyphenated
            ("archi:id-cap_001", ["archi:id-cap_001"]), # Underscore
            ("togaf:adm:A", ["togaf:adm:A"]),           # Single char
            ("multiple iec:one and iec:two", ["iec:one", "iec:two"]),  # Multiple
            ("nested (iec:power) in parens", ["iec:power"]),           # Parentheses
        ]

        for text, expected in test_cases:
            citations = self.grounding._extract_existing_citations(text)
            assert len(citations) == len(expected), f"Expected {expected}, got {citations} for '{text}'"

    def test_citation_format_validation(self):
        """Test citation format validation."""
        valid_citations = [
            "archi:id-cap-001",
            "skos:Concept",
            "iec:ActivePower",
            "togaf:adm:B",
            "entsoe:GridCongestion",
            "lido:event-001"
        ]

        invalid_citations = [
            "invalid:format",
            "archi:",
            "iec",
            "togaf:wrong",
            "",
            "random text"
        ]

        for citation in valid_citations:
            assert self.grounding.validate_citation_format(citation), f"Should be valid: {citation}"

        for citation in invalid_citations:
            assert not self.grounding.validate_citation_format(citation), f"Should be invalid: {citation}"

    def test_citation_statistics(self):
        """Test citation statistics calculation."""
        text = "Use iec:ActivePower and archi:id-cap-001 with iec:Equipment and duplicate iec:ActivePower"

        stats = self.grounding.get_citation_statistics(text)

        assert stats["total_citations"] == 3  # Unique citations only
        assert stats["unique_citations"] == 3  # One duplicate
        assert stats["is_grounded"] is True
        assert stats["by_prefix"]["iec:"]["count"] == 2  # ActivePower and Equipment
        assert stats["by_prefix"]["archi:id-"]["count"] == 1

    def test_enforce_grounding_success(self):
        """Test enforce_grounding returns answer when grounded."""
        grounded_answer = "Based on IEC standard (iec:ActivePower), use this approach."

        result = self.grounding.enforce_grounding(grounded_answer)

        assert result == grounded_answer

    def test_enforce_grounding_failure(self):
        """Test enforce_grounding raises error when not grounded."""
        ungrounded_answer = "This is not grounded in any authoritative source."

        with pytest.raises(UngroundedReplyError):
            self.grounding.enforce_grounding(ungrounded_answer)

    def test_suggest_citations_from_context(self):
        """Test citation suggestion from various context structures."""
        contexts = [
            # Dictionary with nested terms
            {"iec_terms": {"iec:ActivePower": "Active Power", "iec:ReactivePower": "Reactive Power"}},

            # List of terms
            {"concepts": ["archi:id-cap-001", "togaf:adm:B"]},

            # String content
            {"description": "This relates to iec:Equipment and archimate modeling"},

            # Mixed structure
            {
                "retrieved_terms": {"skos:Concept": "Concept"},
                "sources": ["entsoe:GridModel", "lido:event-002"]
            }
        ]

        for context in contexts:
            suggestions = self.grounding.suggest_citations(context)
            assert len(suggestions) > 0, f"Should find suggestions in {context}"

    def test_suggest_citations_smart_generation(self):
        """Test smart citation generation based on context content."""
        context = {"content": "This is about IEC standards and ArchiMate modeling with TOGAF methodology"}

        suggestions = self.grounding.suggest_citations(context)

        # Should generate smart suggestions based on keywords
        has_iec = any("iec:" in s for s in suggestions)
        has_archi = any("archi:" in s for s in suggestions)
        has_togaf = any("togaf:" in s for s in suggestions)

        assert has_iec or has_archi or has_togaf, "Should generate smart suggestions based on content"

    def test_no_false_positives(self):
        """Test that similar but invalid patterns don't count as citations."""
        false_positive_text = """
        This text has words like architect and iconic but no real citations.
        Also mentions technical terms and systematic approaches.
        But none of these should be detected as valid citations.
        """

        citations = self.grounding._extract_existing_citations(false_positive_text)
        assert len(citations) == 0, f"Should not find citations in: {false_positive_text}"

        with pytest.raises(UngroundedReplyError):
            self.grounding.assert_citations(false_positive_text)

    def test_case_insensitive_extraction(self):
        """Test that citation extraction is case insensitive."""
        mixed_case_text = "Use IEC:ActivePower and ARCHI:ID-cap-001 with Skos:Concept"

        citations = self.grounding._extract_existing_citations(mixed_case_text)

        assert len(citations) >= 3
        # Should find all regardless of case
        citation_lower = [c.lower() for c in citations]
        assert any("iec:" in c for c in citation_lower)
        assert any("archi:id-" in c for c in citation_lower)
        assert any("skos:" in c for c in citation_lower)

    def test_grounding_with_punctuation(self):
        """Test citation extraction works with various punctuation."""
        punctuated_text = """
        According to TOGAF (togaf:adm:B), implement capability (archi:id-cap-001).
        The IEC standard [iec:Equipment] defines this, per SKOS {skos:Concept}.
        """

        result = self.grounding.assert_citations(punctuated_text)

        assert result["status"] == "grounded"
        assert len(result["citations"]) >= 4

    def test_regex_pattern_completeness(self):
        """Test that all required prefixes have working regex patterns."""
        for prefix in REQUIRED_CITATION_PREFIXES:
            assert prefix in self.grounding.citation_patterns, f"Missing pattern for {prefix}"

            # Test pattern works
            test_citation = f"{prefix}test123"
            citations = self.grounding._extract_existing_citations(test_citation)
            assert len(citations) > 0, f"Pattern for {prefix} should match {test_citation}"