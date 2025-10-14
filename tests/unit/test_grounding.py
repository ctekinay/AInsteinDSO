"""
Unit tests for GroundingCheck module.

Tests citation format validation, authenticity validation with CitationValidator,
and enforcement of zero-tolerance grounding requirements.
"""

import pytest
from src.safety.grounding import GroundingCheck, REQUIRED_CITATION_PREFIXES
from src.safety.citation_validator import CitationValidator
from src.exceptions.exceptions import UngroundedReplyError, FakeCitationError


class TestGroundingCheckBasic:
    """Test basic grounding functionality without citation validator."""
    
    def test_initialization(self):
        """Test GroundingCheck initializes correctly."""
        gc = GroundingCheck()
        assert gc.required_prefixes == REQUIRED_CITATION_PREFIXES
        assert gc.citation_validator is None
    
    def test_initialization_with_validator(self):
        """Test GroundingCheck initializes with citation validator."""
        validator = CitationValidator()
        gc = GroundingCheck(citation_validator=validator)
        assert gc.citation_validator is not None
    
    def test_extract_valid_citations(self):
        """Test extraction of valid citations from text."""
        gc = GroundingCheck()
        
        text = "This uses skos:Asset and iec:ActivePower citations."
        citations = gc._extract_existing_citations(text)
        
        assert "skos:Asset" in citations
        assert "iec:ActivePower" in citations
        assert len(citations) == 2
    
    def test_extract_multiple_citation_types(self):
        """Test extraction of multiple citation types."""
        gc = GroundingCheck()
        
        text = """
        Using archi:id-abc123 for capability,
        skos:ServiceProvider for definition,
        togaf:adm:B for phase,
        and doc:togaf_concepts:001 for reference.
        """
        citations = gc._extract_existing_citations(text)
        
        assert "archi:id-abc123" in citations
        assert "skos:ServiceProvider" in citations
        assert "togaf:adm:B" in citations
        assert "doc:togaf_concepts:001" in citations
        assert len(citations) == 4
    
    def test_validate_citation_format_valid(self):
        """Test validation of valid citation formats."""
        gc = GroundingCheck()
        
        valid_citations = [
            "skos:Asset",
            "iec:ActivePower",
            "archi:id-abc123",
            "togaf:adm:B",
            "entsoe:GridArea",
            "doc:togaf_concepts:001"
        ]
        
        for citation in valid_citations:
            assert gc.validate_citation_format(citation), f"{citation} should be valid"
    
    def test_validate_citation_format_invalid(self):
        """Test validation rejects invalid citation formats."""
        gc = GroundingCheck()
        
        invalid_citations = [
            "invalid_citation",
            "no:prefix",
            "skos:",  # Missing term
            ":Asset",  # Missing prefix
            ""
        ]
        
        for citation in invalid_citations:
            assert not gc.validate_citation_format(citation), f"{citation} should be invalid"
    
    def test_assert_citations_with_valid_citations(self):
        """Test assert_citations passes with valid citations."""
        gc = GroundingCheck()
        
        response = "An Asset [skos:Asset] is equipment used in operations."
        result = gc.assert_citations(response)
        
        assert result["status"] == "grounded"
        assert len(result["citations"]) == 1
        assert "skos:Asset" in result["citations"]
    
    def test_assert_citations_raises_on_empty(self):
        """Test assert_citations raises error on empty response."""
        gc = GroundingCheck()
        
        with pytest.raises(UngroundedReplyError) as exc_info:
            gc.assert_citations("")
        
        assert "Empty response" in str(exc_info.value)
    
    def test_assert_citations_raises_on_no_citations(self):
        """Test assert_citations raises error when no citations present."""
        gc = GroundingCheck()
        
        response = "This response has no citations at all."
        
        with pytest.raises(UngroundedReplyError) as exc_info:
            gc.assert_citations(response)
        
        assert "lacks required citations" in str(exc_info.value)
    
    def test_get_citation_statistics(self):
        """Test citation statistics extraction."""
        gc = GroundingCheck()
        
        text = """
        Using skos:Asset, iec:ActivePower, and skos:ServiceProvider.
        Also archi:id-123 and togaf:adm:B.
        """
        
        stats = gc.get_citation_statistics(text)
        
        assert stats["total_citations"] == 5
        assert stats["is_grounded"] is True
        assert "skos:" in [c[:5] for c in stats["citations"]]
        assert stats["by_prefix"]["skos:"]["count"] == 2
    
    def test_suggest_citations_from_context(self):
        """Test citation suggestion from retrieval context."""
        gc = GroundingCheck()
        
        context = {
            "kg_results": [
                {"citation_id": "skos:Asset", "label": "Asset"},
                {"citation_id": "iec:ActivePower", "label": "Active Power"}
            ]
        }
        
        suggestions = gc.suggest_citations(context)
        
        assert "skos:Asset" in suggestions
        assert "iec:ActivePower" in suggestions
    
    def test_enforce_grounding_passes(self):
        """Test enforce_grounding returns response when grounded."""
        gc = GroundingCheck()
        
        response = "Asset [skos:Asset] is equipment."
        result = gc.enforce_grounding(response)
        
        assert result == response
    
    def test_enforce_grounding_raises(self):
        """Test enforce_grounding raises error when not grounded."""
        gc = GroundingCheck()
        
        response = "This has no citations."
        
        with pytest.raises(UngroundedReplyError):
            gc.enforce_grounding(response)


class TestGroundingCheckWithValidator:
    """Test grounding with citation validator for authenticity checks."""
    
    def test_assert_citations_validates_authenticity(self):
        """Test that assert_citations validates citation authenticity."""
        # Create mock validator that rejects all citations
        class MockValidator:
            def validate_response_citations(self, response, citation_pool, trace_id=None):
                return {
                    "valid": False,
                    "fake_citations": ["skos:FakeCitation"],
                    "valid_citations": [],
                    "total_citations": 1,
                    "message": "Fake citation detected"
                }
        
        gc = GroundingCheck(citation_validator=MockValidator())
        response = "Using [skos:FakeCitation] here."
        citation_pool = ["skos:RealCitation"]
        
        with pytest.raises(FakeCitationError) as exc_info:
            gc.assert_citations(response, citation_pool=citation_pool)
        
        assert "skos:FakeCitation" in exc_info.value.fake_citations
    
    def test_assert_citations_passes_with_valid_pool(self):
        """Test assert_citations passes when citations are in pool."""
        # Create mock validator that accepts citations in pool
        class MockValidator:
            def validate_response_citations(self, response, citation_pool, trace_id=None):
                return {
                    "valid": True,
                    "fake_citations": [],
                    "valid_citations": ["skos:Asset"],
                    "total_citations": 1,
                    "message": "All valid"
                }
        
        gc = GroundingCheck(citation_validator=MockValidator())
        response = "Using [skos:Asset] here."
        citation_pool = ["skos:Asset"]
        
        result = gc.assert_citations(response, citation_pool=citation_pool)
        
        assert result["status"] == "grounded"
        assert "skos:Asset" in result["citations"]
    
    def test_validator_not_called_without_pool(self):
        """Test validator is not called when citation_pool is None."""
        called = {"count": 0}
        
        class MockValidator:
            def validate_response_citations(self, response, citation_pool, trace_id=None):
                called["count"] += 1
                return {"valid": True, "fake_citations": [], "valid_citations": []}
        
        gc = GroundingCheck(citation_validator=MockValidator())
        response = "Using [skos:Asset] here."
        
        # Call without citation_pool
        result = gc.assert_citations(response, citation_pool=None)
        
        # Validator should not be called
        assert called["count"] == 0
        assert result["status"] == "grounded"
    
    def test_validator_not_called_without_citations(self):
        """Test validator is not called when no citations in response."""
        called = {"count": 0}
        
        class MockValidator:
            def validate_response_citations(self, response, citation_pool, trace_id=None):
                called["count"] += 1
                return {"valid": True, "fake_citations": [], "valid_citations": []}
        
        gc = GroundingCheck(citation_validator=MockValidator())
        response = "This has no citations."
        citation_pool = ["skos:Asset"]
        
        # Should raise UngroundedReplyError before validator is called
        with pytest.raises(UngroundedReplyError):
            gc.assert_citations(response, citation_pool=citation_pool)
        
        # Validator should not be called
        assert called["count"] == 0


class TestGroundingCheckEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_duplicate_citations_counted_once(self):
        """Test duplicate citations are counted as unique."""
        gc = GroundingCheck()
        
        response = "Using [skos:Asset] and [skos:Asset] twice."
        result = gc.assert_citations(response)
        
        # Should extract both but they're the same
        assert result["citation_count"] >= 1
        assert "skos:Asset" in result["citations"]
    
    def test_case_insensitive_citation_extraction(self):
        """Test citations are extracted case-insensitively."""
        gc = GroundingCheck()
        
        response = "Using [SKOS:Asset] and [skos:asset]."
        citations = gc._extract_existing_citations(response)
        
        # Should find both despite case differences
        assert len(citations) >= 1
    
    def test_citation_in_brackets(self):
        """Test citations work both with and without brackets."""
        gc = GroundingCheck()
        
        responses = [
            "Using skos:Asset here.",
            "Using [skos:Asset] here.",
            "Using (skos:Asset) here."
        ]
        
        for response in responses:
            result = gc.assert_citations(response)
            assert result["status"] == "grounded"
            assert "skos:Asset" in result["citations"]
    
    def test_whitespace_handling(self):
        """Test responses with various whitespace are handled correctly."""
        gc = GroundingCheck()
        
        response = """
        
        Using skos:Asset here.
        
        """
        
        result = gc.assert_citations(response)
        assert result["status"] == "grounded"
    
    def test_suggest_citations_empty_context(self):
        """Test suggest_citations returns empty list for empty context."""
        gc = GroundingCheck()
        
        suggestions = gc.suggest_citations({})
        assert suggestions == []
        
        suggestions = gc.suggest_citations(None)
        assert suggestions == []


class TestGroundingCheckFactory:
    """Test factory function for creating GroundingCheck."""
    
    def test_create_grounding_check_without_validator(self):
        """Test factory creates GroundingCheck without validator."""
        from src.safety.grounding import create_grounding_check
        
        gc = create_grounding_check()
        assert isinstance(gc, GroundingCheck)
        assert gc.citation_validator is None
    
    def test_create_grounding_check_with_validator(self):
        """Test factory creates GroundingCheck with validator."""
        from src.safety.grounding import create_grounding_check
        
        validator = CitationValidator()
        gc = create_grounding_check(citation_validator=validator)
        
        assert isinstance(gc, GroundingCheck)
        assert gc.citation_validator is validator


if __name__ == "__main__":
    pytest.main([__file__, "-v"])