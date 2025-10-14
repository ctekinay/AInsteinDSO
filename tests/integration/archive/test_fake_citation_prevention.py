"""
Integration tests for fake citation prevention.

These tests verify the complete pipeline blocks fake citations.
"""

import pytest
from src.agents.ea_assistant import ProductionEAAgent
from src.exceptions.exceptions import FakeCitationError


@pytest.mark.asyncio
async def test_blocks_fake_archi_citations():
    """Verify system rejects fake ArchiMate citations."""
    agent = ProductionEAAgent()
    
    # Ensure models are loaded
    assert len(agent.archimate_parser.elements) > 0
    
    # Query that might trigger fake citations
    query = "What capability should I use for grid congestion management?"
    
    response = await agent.process_query(query)
    
    # Verify no known fake citations
    KNOWN_FAKES = ["archi:id-cap-001", "archi:id-fake", "archi:id-gridcongestion"]
    for fake in KNOWN_FAKES:
        assert fake not in response.response.lower(), \
            f"Fake citation {fake} found in response"
    
    # Verify all archi: citations are valid
    for citation in response.citations:
        if citation.startswith("archi:id-"):
            assert agent.archimate_parser.citation_exists(citation), \
                f"Invalid ArchiMate citation: {citation}"


@pytest.mark.asyncio
async def test_blocks_fake_iec_citations():
    """Verify system rejects fake IEC citations."""
    agent = ProductionEAAgent()
    
    query = "What is reactive power according to IEC standards?"
    response = await agent.process_query(query)
    
    # Known fake IEC citations
    KNOWN_FAKES = ["iec:GridCongestion", "iec:61968", "iec:61970"]
    for fake in KNOWN_FAKES:
        assert fake not in response.response, \
            f"Fake IEC citation {fake} found in response"
    
    # Verify all iec: citations exist in KG
    for citation in response.citations:
        if citation.startswith("iec:"):
            assert agent.kg_loader.citation_exists(citation), \
                f"Invalid IEC citation: {citation}"


@pytest.mark.asyncio
async def test_citation_pool_only_valid():
    """Verify citation pool contains only valid citations."""
    agent = ProductionEAAgent()
    
    # Mock retrieval context
    context = {
        "kg_results": [
            {"citation_id": "skos:Asset", "label": "Asset"},
            {"citation_id": "iec:ActivePower", "label": "Active Power"}
        ],
        "archimate_elements": []
    }
    
    citation_pool = agent._build_citation_pool(context, "test-trace")
    
    # Verify all citations in pool exist
    for citation in citation_pool:
        if citation.startswith("skos:") or citation.startswith("iec:"):
            assert agent.kg_loader.citation_exists(citation), \
                f"Invalid citation in pool: {citation}"


@pytest.mark.asyncio
async def test_citation_validator_integration():
    """Test citation validator integration with pipeline."""
    agent = ProductionEAAgent()
    
    # Citation validator should be initialized
    assert agent.citation_validator is not None
    assert agent.citation_validator.kg_loader is not None
    assert agent.citation_validator.archimate_parser is not None
    
    # Test validation
    assert agent.citation_validator.validate_citation_exists("skos:Asset")
    assert not agent.citation_validator.validate_citation_exists("skos:FakeAsset123")


@pytest.mark.asyncio
async def test_empty_citation_pool_handling():
    """Verify graceful handling when citation pool is empty."""
    agent = ProductionEAAgent()
    
    # Context with no valid citations
    context = {
        "kg_results": [],
        "archimate_elements": []
    }
    
    citation_pool = agent._build_citation_pool(context, "test-trace")
    
    # Pool should be empty
    assert len(citation_pool) == 0
    
    # System should handle this gracefully (abstain or use fallback)
    query = "What is a capability?"
    response = await agent.process_query(query)
    
    # Should either abstain or use very generic response
    assert response.requires_human_review or "cannot provide" in response.response.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])