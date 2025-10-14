"""
Unit tests for the Knowledge Graph loader.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rdflib import Graph, Literal, Namespace, URIRef

from src.exceptions.exceptions import KnowledgeGraphError, PerformanceError
from src.knowledge.kg_loader import KnowledgeGraphLoader

SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
IEC = Namespace("http://iec.ch/TC57/")
ENTSOE = Namespace("http://entsoe.eu/CIM/")
EURLEX = Namespace("http://data.europa.eu/eli/")


def create_test_graph() -> Graph:
    """Create a test knowledge graph with sample triples."""
    g = Graph()
    g.bind("skos", SKOS)
    g.bind("iec", IEC)
    g.bind("entsoe", ENTSOE)
    g.bind("eurlex", EURLEX)

    # Add IEC terms
    g.add((IEC.ActivePower, SKOS.prefLabel, Literal("Active Power")))
    g.add((IEC.ReactivePower, SKOS.prefLabel, Literal("Reactive Power")))
    g.add((IEC.Transformer, SKOS.prefLabel, Literal("Transformer")))

    # Add ENTSOE terms
    g.add((ENTSOE.GridCongestion, SKOS.prefLabel, Literal("Grid Congestion")))
    g.add((ENTSOE.LoadProfile, SKOS.prefLabel, Literal("Load Profile")))

    # Add EUR-LEX terms
    g.add((EURLEX.directive_2019_944, SKOS.prefLabel, Literal("Electricity Directive")))

    # Add more triples to meet minimum requirement (39100+)
    # In reality, we'd have the full graph, but for testing we'll mock this
    for i in range(100):
        g.add((URIRef(f"http://example.org/test/{i}"), SKOS.note, Literal(f"Test {i}")))

    return g


class TestKnowledgeGraphLoader:
    """Test suite for KnowledgeGraphLoader."""

    def test_initialization(self):
        """Test loader initialization with default and custom paths."""
        # Default path
        loader = KnowledgeGraphLoader()
        assert loader.kg_path == Path("data/energy_knowledge_graph.ttl")
        assert loader.graph is None

        # Custom path
        custom_path = Path("/custom/path/kg.ttl")
        loader = KnowledgeGraphLoader(custom_path)
        assert loader.kg_path == custom_path

    def test_load_missing_file(self):
        """Test loading from non-existent file raises error."""
        loader = KnowledgeGraphLoader(Path("/nonexistent/file.ttl"))
        with pytest.raises(KnowledgeGraphError) as exc_info:
            loader.load()
        assert "not found" in str(exc_info.value)

    def test_load_and_validate_graph(self):
        """Test loading and validating a graph from TTL file."""
        with tempfile.NamedTemporaryFile(suffix=".ttl", mode="w", delete=False) as f:
            test_graph = create_test_graph()
            test_graph.serialize(f, format="turtle")
            temp_path = Path(f.name)

        try:
            loader = KnowledgeGraphLoader(temp_path)

            # Mock the validation to accept our smaller test graph
            with patch.object(loader, "_validate_graph"):
                loader.load()

            assert loader.graph is not None
            assert len(loader.graph) > 0
            assert loader.load_time_ms > 0

        finally:
            temp_path.unlink()

    def test_validate_graph_minimum_triples(self):
        """Test graph validation fails with insufficient triples."""
        loader = KnowledgeGraphLoader()
        loader.graph = Graph()

        # Add fewer than required triples
        for i in range(100):
            loader.graph.add((URIRef(f"http://test/{i}"), SKOS.note, Literal(f"Test {i}")))

        with pytest.raises(KnowledgeGraphError) as exc_info:
            loader._validate_graph()
        assert "39100" in str(exc_info.value)

    def test_extract_terms(self):
        """Test extracting IEC, ENTSOE, and EUR-LEX terms."""
        loader = KnowledgeGraphLoader()
        loader.graph = create_test_graph()

        loader.extract_terms()

        # Check IEC terms
        assert len(loader.iec_terms) == 3
        assert any("ActivePower" in uri for uri in loader.iec_terms)
        assert "Active Power" in loader.iec_terms.values()

        # Check ENTSOE terms
        assert len(loader.entsoe_terms) == 2
        assert any("GridCongestion" in uri for uri in loader.entsoe_terms)
        assert "Grid Congestion" in loader.entsoe_terms.values()

        # Check EUR-LEX terms
        assert len(loader.eurlex_terms) == 1
        assert "Electricity Directive" in loader.eurlex_terms.values()

    def test_extract_terms_without_graph(self):
        """Test extracting terms without loading graph raises error."""
        loader = KnowledgeGraphLoader()
        with pytest.raises(KnowledgeGraphError) as exc_info:
            loader.extract_terms()
        assert "must be loaded" in str(exc_info.value)

    def test_save_vocabularies(self):
        """Test saving extracted vocabularies to JSON."""
        loader = KnowledgeGraphLoader()
        loader.graph = create_test_graph()
        loader.extract_terms()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_vocab.json"
            loader.save_vocabularies(output_path)

            assert output_path.exists()

            with open(output_path) as f:
                data = json.load(f)

            assert "extracted_terms" in data
            assert "iec_terms" in data["extracted_terms"]
            assert "entsoe_terms" in data["extracted_terms"]
            assert "eurlex_terms" in data["extracted_terms"]
            assert "extraction_timestamp" in data["extracted_terms"]
            assert data["extracted_terms"]["triple_count"] > 0

    def test_get_statistics(self):
        """Test getting statistics about the knowledge graph."""
        loader = KnowledgeGraphLoader()
        loader.graph = create_test_graph()
        loader.extract_terms()

        stats = loader.get_statistics()

        assert "total_triples" in stats
        assert stats["total_triples"] > 0
        assert "namespaces" in stats
        assert "top_predicates" in stats
        assert "iec_terms_count" in stats
        assert stats["iec_terms_count"] == 3
        assert stats["entsoe_terms_count"] == 2
        assert stats["eurlex_terms_count"] == 1

    def test_query(self):
        """Test executing SPARQL queries on the graph."""
        loader = KnowledgeGraphLoader()
        loader.graph = create_test_graph()

        # Query for IEC terms
        query = """
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        SELECT ?concept ?label WHERE {
            ?concept skos:prefLabel ?label .
            FILTER(CONTAINS(STR(?concept), "IEC"))
        }
        """

        results = loader.query(query)

        assert len(results) == 3
        assert all("concept" in r and "label" in r for r in results)
        assert any("Active Power" in r["label"] for r in results)

    def test_query_without_graph(self):
        """Test querying without loading graph raises error."""
        loader = KnowledgeGraphLoader()
        with pytest.raises(KnowledgeGraphError) as exc_info:
            loader.query("SELECT * WHERE { ?s ?p ?o }")
        assert "not loaded" in str(exc_info.value)

    def test_performance_load_timeout(self):
        """Test that slow loading raises PerformanceError."""
        loader = KnowledgeGraphLoader()

        # Mock slow parsing
        with patch("time.time") as mock_time:
            mock_time.side_effect = [0, 0.6]  # 600ms elapsed
            with patch.object(Graph, "parse"):
                loader.graph = Graph()
                with pytest.raises(PerformanceError) as exc_info:
                    loader.load()
                assert "500ms" in str(exc_info.value)

    def test_performance_query_timeout(self):
        """Test that slow queries raise PerformanceError."""
        loader = KnowledgeGraphLoader()
        loader.graph = create_test_graph()

        # Mock slow query
        with patch("time.time") as mock_time:
            mock_time.side_effect = [0, 0.35]  # 350ms elapsed
            with pytest.raises(PerformanceError) as exc_info:
                loader.query("SELECT * WHERE { ?s ?p ?o }")
            assert "300ms" in str(exc_info.value)