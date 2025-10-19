"""
Comprehensive test suite for comparison query handling.

Tests ensure that comparison queries ALWAYS return two distinct concepts
with different citations.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from src.agents.ea_assistant import ProductionEAAgent


class TestComparisonQueryDistinctness:
    """Test that comparison queries return distinct concepts."""

    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        # Mock the agent with minimal dependencies
        with patch('src.agents.ea_assistant.KnowledgeGraphLoader'):
            with patch('src.agents.ea_assistant.ArchiMateParser'):
                with patch('src.agents.ea_assistant.PDFIndexer'):
                    agent = ProductionEAAgent(
                        llm_provider='groq',
                        vocab_path='data/vocab.alliander.ttl',
                        models_path='data/models',
                        docs_path='data/docs'
                    )
                    return agent

    @pytest.mark.asyncio
    async def test_active_vs_reactive_power(self, agent):
        """
        CRITICAL TEST: The original failing case.

        Query: "What is the difference between active and reactive power?"
        Expected: Two DISTINCT concepts with DIFFERENT citations
        """
        query = "What is the difference between active and reactive power?"

        # Mock candidates (simulating what retrieval might return)
        candidates = [
            {
                'element': 'Active power',
                'definition': 'The real component of apparent power...',
                'citation': 'eurlex:631-20',
                'confidence': 0.85
            },
            {
                'element': 'Reactive power',
                'definition': 'The imaginary component of apparent power...',
                'citation': 'eurlex:631-28',
                'confidence': 0.82
            },
            {
                'element': 'Apparent power',
                'definition': 'The product of RMS voltage and current...',
                'citation': 'eurlex:631-25',
                'confidence': 0.70
            }
        ]

        # Test validation
        concept1, concept2 = await agent._validate_comparison_candidates(candidates, query)

        # CRITICAL ASSERTIONS
        assert concept1 is not None, "Concept 1 should not be None"
        assert concept2 is not None, "Concept 2 should not be None"

        citation1 = concept1.get('citation')
        citation2 = concept2.get('citation')

        assert citation1 is not None, "Concept 1 must have a citation"
        assert citation2 is not None, "Concept 2 must have a citation"
        assert citation1 != citation2, f"Citations must be distinct: {citation1} vs {citation2}"

        # Check that we got the right concepts
        element1 = concept1.get('element', '').lower()
        element2 = concept2.get('element', '').lower()

        assert 'active' in element1 or 'active' in concept1.get('definition', '').lower(), \
            f"Concept 1 should relate to 'active power', got: {element1}"
        assert 'reactive' in element2 or 'reactive' in concept2.get('definition', '').lower(), \
            f"Concept 2 should relate to 'reactive power', got: {element2}"

        print(f"✅ Test passed: {concept1.get('element')} [{citation1}] vs {concept2.get('element')} [{citation2}]")

    @pytest.mark.asyncio
    async def test_duplicate_candidates_triggers_fallback(self, agent):
        """Test that duplicate citations trigger semantic fallback."""
        query = "difference between transformer and conductor?"

        # Mock candidates with DUPLICATE citations (the bug scenario)
        duplicate_candidates = [
            {
                'element': 'Transformer',
                'definition': 'Equipment that changes voltage levels...',
                'citation': 'iec:equipment-001',
                'confidence': 0.80
            },
            {
                'element': 'Transformer',  # DUPLICATE
                'definition': 'Equipment that changes voltage levels...',
                'citation': 'iec:equipment-001',  # SAME CITATION
                'confidence': 0.78
            }
        ]

        # Mock the semantic fallback to return distinct concepts
        async def mock_semantic_fallback(query, candidates):
            return (
                {
                    'element': 'Transformer',
                    'definition': 'Voltage changing device...',
                    'citation': 'iec:transformer-001',
                    'confidence': 0.75
                },
                {
                    'element': 'Conductor',
                    'definition': 'Current carrying component...',
                    'citation': 'iec:conductor-001',
                    'confidence': 0.72
                }
            )

        # Patch the semantic fallback
        agent._semantic_comparison_fallback = mock_semantic_fallback
        agent.embedding_agent = Mock()  # Enable semantic search

        # Should trigger fallback due to duplicate citations
        concept1, concept2 = await agent._validate_comparison_candidates(duplicate_candidates, query)

        # Verify distinct results
        assert concept1.get('citation') != concept2.get('citation'), \
            "Semantic fallback should return distinct citations"

        print(f"✅ Duplicate detection worked, fallback returned: {concept1.get('citation')} vs {concept2.get('citation')}")

    @pytest.mark.asyncio
    async def test_comparison_term_extraction(self, agent):
        """Test extraction of comparison terms from various query formats."""
        test_cases = [
            ("difference between active and reactive power", ["active", "reactive"]),
            ("active power vs reactive power", ["active power", "reactive power"]),
            ("compare transformer with conductor", ["transformer", "conductor"]),
            ("business capability or application component", ["business capability", "application component"]),
        ]

        for query, expected_terms in test_cases:
            extracted = agent._extract_comparison_terms(query)

            # Normalize for comparison
            extracted_lower = [t.lower() for t in extracted]
            expected_lower = [t.lower() for t in expected_terms]

            # Check if we got the expected terms (allow partial matches)
            matches = sum(
                any(exp in ext or ext in exp for ext in extracted_lower)
                for exp in expected_lower
            )

            assert matches >= len(expected_terms) * 0.8, \
                f"Expected terms {expected_terms}, got {extracted} from query: {query}"

            print(f"✅ Extracted {extracted} from: {query}")

    @pytest.mark.asyncio
    async def test_insufficient_candidates_raises_error(self, agent):
        """Test that insufficient candidates raises appropriate error."""
        query = "difference between X and Y?"

        # Only one candidate
        candidates = [
            {
                'element': 'Concept X',
                'definition': 'Definition of X...',
                'citation': 'test:001',
                'confidence': 0.80
            }
        ]

        # Should raise error or trigger fallback
        # If embedding agent is None, should raise ValueError
        agent.embedding_agent = None

        with pytest.raises(ValueError, match="Insufficient candidates"):
            await agent._validate_comparison_candidates(candidates, query)

        print("✅ Correctly raised error for insufficient candidates")

    @pytest.mark.asyncio
    async def test_get_first_two_distinct(self, agent):
        """Test fallback method for getting distinct candidates."""
        candidates = [
            {'element': 'A', 'citation': 'test:001', 'confidence': 0.9},
            {'element': 'A', 'citation': 'test:001', 'confidence': 0.85},  # Duplicate
            {'element': 'B', 'citation': 'test:002', 'confidence': 0.80},
            {'element': 'C', 'citation': 'test:003', 'confidence': 0.75},
        ]

        c1, c2 = agent._get_first_two_distinct(candidates)

        assert c1.get('citation') != c2.get('citation'), "Should return distinct citations"
        assert c1.get('citation') == 'test:001', "Should get first distinct"
        assert c2.get('citation') == 'test:002', "Should get second distinct"

        print(f"✅ First two distinct: {c1.get('citation')} and {c2.get('citation')}")


class TestComparisonQueryEndToEnd:
    """End-to-end tests for comparison queries."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_comparison_pipeline(self):
        """
        Full pipeline test with real agent.

        NOTE: This requires actual data files and API keys.
        Mark as integration test.
        """
        # This test would run the full pipeline
        # Skip if not in integration test mode
        pytest.skip("Integration test - requires full setup")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])