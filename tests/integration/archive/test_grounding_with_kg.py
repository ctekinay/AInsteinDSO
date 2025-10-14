"""
Integration tests for grounding check with knowledge graph loader.

Tests the interaction between GroundingCheck and KnowledgeGraphLoader
to ensure citations can be suggested from real extracted terms.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from rdflib import Graph, Literal, Namespace, URIRef

from src.exceptions.exceptions import UngroundedReplyError
from src.safety.grounding import GroundingCheck


def create_test_kg_content() -> str:
    """Create a sample TTL content for testing."""
    return """
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix iec: <http://iec.ch/TC57/> .
@prefix entsoe: <http://entsoe.eu/CIM/> .
@prefix eurlex: <http://data.europa.eu/eli/> .

iec:ActivePower skos:prefLabel "Active Power" .
iec:ReactivePower skos:prefLabel "Reactive Power" .
iec:Transformer skos:prefLabel "Power Transformer" .
iec:Equipment skos:prefLabel "Electrical Equipment" .

entsoe:GridCongestion skos:prefLabel "Grid Congestion" .
entsoe:LoadProfile skos:prefLabel "Load Profile" .

eurlex:directive_2019_944 skos:prefLabel "Electricity Market Directive" .

# Add additional triples for various tests
iec:VoltageLevel skos:prefLabel "Voltage Level" .
iec:Substation skos:prefLabel "Electrical Substation" .
entsoe:PowerFlow skos:prefLabel "Power Flow Analysis" .
"""


class TestGroundingWithKnowledgeGraph:
    """Integration tests for grounding with knowledge graph context."""

    def setup_method(self):
        """Set up test fixtures."""
        self.grounding = GroundingCheck()

    def test_grounding_with_kg_extracted_terms(self):
        """Test grounding can suggest citations from KG extracted terms."""
        # Simulate extracted terms from KG loader
        kg_context = {
            "iec_terms": {
                "iec:ActivePower": "Active Power",
                "iec:ReactivePower": "Reactive Power",
                "iec:Transformer": "Power Transformer"
            },
            "entsoe_terms": {
                "entsoe:GridCongestion": "Grid Congestion",
                "entsoe:LoadProfile": "Load Profile"
            },
            "eurlex_terms": {
                "eurlex:directive_2019_944": "Electricity Market Directive"
            }
        }

        uncited_response = "You should model electrical transformers according to standards"

        result = self.grounding.assert_citations(uncited_response, kg_context)

        assert result["status"] == "needs_citations"
        assert len(result["suggestions"]) > 0

        # Should suggest relevant IEC terms
        suggestions = result["suggestions"]
        has_iec_suggestion = any("iec:" in s for s in suggestions)
        assert has_iec_suggestion, f"Should suggest IEC terms, got: {suggestions}"

    def test_grounding_with_real_kg_structure(self):
        """Test grounding with realistic KG loader output structure."""
        # Simulate what KnowledgeGraphLoader would return
        realistic_context = {
            "retrieved_terms": {
                "concepts": [
                    {"uri": "iec:ActivePower", "label": "Active Power"},
                    {"uri": "archi:id-cap-001", "label": "Grid Management Capability"},
                    {"uri": "togaf:adm:B", "label": "Business Architecture Phase"}
                ]
            },
            "extraction_metadata": {
                "query_terms": ["transformer", "grid", "capability"],
                "namespace_distribution": {
                    "iec": 45,
                    "entsoe": 23,
                    "archimate": 12
                }
            }
        }

        uncited_response = "Model grid management capabilities using business architecture"

        result = self.grounding.assert_citations(uncited_response, realistic_context)

        assert result["status"] == "needs_citations"
        suggestions = result["suggestions"]

        # Should find suggestions from the nested structure
        assert len(suggestions) > 0
        has_relevant_suggestion = any(
            any(prefix in s for prefix in ["iec:", "archi:", "togaf:"])
            for s in suggestions
        )
        assert has_relevant_suggestion

    def test_properly_cited_response_with_kg_context(self):
        """Test that properly cited responses pass even with KG context."""
        kg_context = {
            "iec_terms": {"iec:Equipment": "Equipment"},
            "archimate_elements": {"archi:id-cap-001": "Capability"}
        }

        properly_cited = "Based on IEC standards (iec:Equipment), implement capability (archi:id-cap-001)"

        result = self.grounding.assert_citations(properly_cited, kg_context)

        assert result["status"] == "grounded"
        assert len(result["citations"]) >= 2

    def test_no_false_suggestions_from_kg(self):
        """Test that KG context doesn't create false citation suggestions."""
        kg_context = {
            "non_citation_data": {
                "descriptions": ["This is about power systems"],
                "metadata": {"source": "IEC standards", "version": "2023"}
            },
            "unrelated_terms": ["power", "grid", "system"]
        }

        uncited_response = "This response has no citations"

        # Should raise error since no valid citations can be suggested
        with pytest.raises(UngroundedReplyError):
            self.grounding.assert_citations(uncited_response, kg_context)

    def test_kg_context_smart_suggestions(self):
        """Test smart suggestion generation based on KG context content."""
        kg_context = {
            "query_context": "User asked about IEC transformer modeling in ArchiMate",
            "retrieved_documents": [
                "Document about iec:Transformer specifications",
                "ArchiMate modeling guide with archi:id-dev-001 examples"
            ]
        }

        uncited_response = "Model transformers using standard approaches"

        result = self.grounding.assert_citations(uncited_response, kg_context)

        assert result["status"] == "needs_citations"
        suggestions = result["suggestions"]

        # Should generate smart suggestions based on context keywords
        has_iec = any("iec:" in s for s in suggestions)
        has_archi = any("archi:" in s for s in suggestions)

        assert has_iec or has_archi, f"Should generate smart suggestions, got: {suggestions}"

    def test_partial_kg_integration(self):
        """Test grounding with partial/incomplete KG context."""
        partial_context = {
            "iec_terms": {},  # Empty
            "partial_data": "some iec:Equipment mention",
            "incomplete": ["archi:id-cap"]  # Incomplete citation
        }

        uncited_response = "Use equipment modeling approaches"

        result = self.grounding.assert_citations(uncited_response, partial_context)

        # Should still find the valid citation in partial_data
        assert result["status"] == "needs_citations"
        suggestions = result["suggestions"]
        has_equipment_suggestion = any("iec:Equipment" in s for s in suggestions)
        assert has_equipment_suggestion

    def test_empty_kg_context_behavior(self):
        """Test grounding behavior with empty KG context."""
        empty_contexts = [{}, {"empty_field": []}, None]

        uncited_response = "This response lacks citations"

        for context in empty_contexts:
            with pytest.raises(UngroundedReplyError) as exc_info:
                self.grounding.assert_citations(uncited_response, context)

            # Should still provide the standard error message
            assert "lacks required citations" in str(exc_info.value)

    @patch('src.knowledge.kg_loader.KnowledgeGraphLoader')
    def test_full_integration_simulation(self, mock_loader_class):
        """Test simulated full integration with KG loader."""
        # Mock the KG loader behavior
        mock_loader = mock_loader_class.return_value
        mock_loader.iec_terms = {
            "iec:ActivePower": "Active Power",
            "iec:Equipment": "Electrical Equipment"
        }
        mock_loader.entsoe_terms = {
            "entsoe:GridCongestion": "Grid Congestion"
        }

        # Simulate retrieval context that would come from the loader
        retrieval_context = {
            "kg_terms": {
                "iec_terms": mock_loader.iec_terms,
                "entsoe_terms": mock_loader.entsoe_terms
            },
            "query_match_score": 0.85,
            "retrieved_concepts": ["iec:Equipment", "entsoe:GridCongestion"]
        }

        # Test both success and failure cases
        grounded_response = "Use IEC equipment (iec:Equipment) for grid management"
        ungrounded_response = "Use standard equipment approaches"

        # Grounded response should pass
        result_grounded = self.grounding.assert_citations(grounded_response, retrieval_context)
        assert result_grounded["status"] == "grounded"

        # Ungrounded response should get suggestions
        result_ungrounded = self.grounding.assert_citations(ungrounded_response, retrieval_context)
        assert result_ungrounded["status"] == "needs_citations"
        assert len(result_ungrounded["suggestions"]) > 0

    def test_complex_kg_nested_structure(self):
        """Test grounding with complex nested KG context structure."""
        complex_context = {
            "sparql_results": [
                {
                    "concept": {"value": "iec:ActivePower"},
                    "label": {"value": "Active Power"},
                    "definition": {"value": "Power consumed by resistive components"}
                },
                {
                    "concept": {"value": "archi:id-cap-001"},
                    "label": {"value": "Grid Management Capability"},
                    "type": {"value": "BusinessCapability"}
                }
            ],
            "inference_results": {
                "related_concepts": ["iec:ReactivePower", "entsoe:LoadProfile"],
                "semantic_similarity": 0.92
            }
        }

        uncited_response = "Implement power management using business capabilities"

        result = self.grounding.assert_citations(uncited_response, complex_context)

        assert result["status"] == "needs_citations"
        suggestions = result["suggestions"]

        # Should extract from nested structure
        has_iec = any("iec:" in s for s in suggestions)
        has_archi = any("archi:" in s for s in suggestions)
        assert has_iec or has_archi, "Should extract from complex nested structure"

    def test_performance_with_large_kg_context(self):
        """Test grounding performance with large KG context."""
        # Simulate large context from KG
        large_context = {
            "iec_terms": {f"iec:term_{i}": f"Term {i}" for i in range(1000)},
            "entsoe_terms": {f"entsoe:concept_{i}": f"Concept {i}" for i in range(500)},
            "descriptions": [f"Description {i} with iec:Equipment_{i}" for i in range(100)]
        }

        uncited_response = "This is a test response without citations"

        # Should handle large context without performance issues
        import time
        start_time = time.time()

        result = self.grounding.assert_citations(uncited_response, large_context)

        elapsed_time = time.time() - start_time

        # Should complete quickly even with large context
        assert elapsed_time < 1.0, f"Grounding took too long: {elapsed_time}s"
        assert result["status"] == "needs_citations"
        assert len(result["suggestions"]) > 0