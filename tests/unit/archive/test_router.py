"""
Comprehensive unit tests for the QueryRouter - Critical input protection layer.

These tests verify that the router correctly implements the STRICT priority order:
1. IEC/Energy + ArchiMate terms → structured_model
2. TOGAF ADM phases + viewpoints → togaf_method
3. Everything else → unstructured_docs

The router is essential for preventing unnecessary vector searches and ensuring
the right knowledge source is queried first.
"""

import json
import tempfile
import time
import unittest
from pathlib import Path

from src.routing.query_router import QueryRouter


class TestQueryRouter(unittest.TestCase):
    """
    Test suite for QueryRouter - verifies correct routing decisions and performance.
    """

    def setUp(self):
        """Set up test fixtures with temporary vocabulary file."""
        # Create test vocabularies
        self.test_vocab = {
            "routing_terms": {
                "iec_energy": [
                    "ActivePower", "ReactivePower", "Equipment", "Conductor",
                    "Breaker", "Transformer", "IEC 61968", "IEC 61970", "CIM",
                    "GridCongestion", "DistributionSystem", "EnergyMeter",
                    "PowerFlow", "LoadProfile", "VoltageLevel", "Substation"
                ],
                "togaf_archimate": [
                    "Actor", "Role", "Process", "Service", "Capability",
                    "Component", "Interface", "Node", "Device", "SystemSoftware",
                    "BusinessArchitecture", "ApplicationArchitecture",
                    "TechnologyArchitecture", "PhaseA", "PhaseB", "PhaseC", "PhaseD", "ADM"
                ]
            }
        }

        # Create temporary vocabulary file
        self.temp_vocab_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        )
        json.dump(self.test_vocab, self.temp_vocab_file, indent=2)
        self.temp_vocab_file.close()

        # Initialize router with test vocabularies
        self.router = QueryRouter(vocab_path=self.temp_vocab_file.name)

    def tearDown(self):
        """Clean up test fixtures."""
        Path(self.temp_vocab_file.name).unlink()

    # PRIORITY 1 TESTS: IEC/Energy + ArchiMate → structured_model
    def test_iec_energy_terms_route_to_structured_model(self):
        """Test IEC/Energy terms correctly route to structured_model."""
        test_cases = [
            "What capability for grid congestion?",  # capability + grid congestion
            "Model ActivePower measurement",         # IEC term
            "How to configure Transformer equipment?", # IEC equipment
            "Implement DistributionSystem monitoring", # IEC system
            "Design PowerFlow analysis using IEC 61968", # IEC standard
            "Configure EnergyMeter for LoadProfile tracking" # IEC terms
        ]

        for query in test_cases:
            with self.subTest(query=query):
                route = self.router.route(query)
                self.assertEqual(route, "structured_model",
                               f"Query '{query}' should route to structured_model")

    def test_archimate_elements_route_to_structured_model(self):
        """Test ArchiMate elements correctly route to structured_model."""
        test_cases = [
            "Define business capability for energy management",
            "Create service interface for grid operations",
            "Model actor roles in distribution system",
            "Design component architecture for metering",
            "Implement process for equipment maintenance",
            "Configure device nodes for substations"
        ]

        for query in test_cases:
            with self.subTest(query=query):
                route = self.router.route(query)
                self.assertEqual(route, "structured_model",
                               f"Query '{query}' should route to structured_model")

    # PRIORITY 2 TESTS: TOGAF ADM + viewpoints → togaf_method
    def test_togaf_adm_phases_route_to_togaf_method(self):
        """Test TOGAF ADM phases correctly route to togaf_method."""
        test_cases = [
            "Phase B business architecture",
            "Implement Phase A architecture vision",
            "Design Phase C information systems",
            "Configure Phase D technology architecture",
            "Follow ADM methodology for EA",
            "Create architecture vision using TOGAF"
        ]

        for query in test_cases:
            with self.subTest(query=query):
                route = self.router.route(query)
                self.assertEqual(route, "togaf_method",
                               f"Query '{query}' should route to togaf_method")

    def test_togaf_viewpoints_route_to_togaf_method(self):
        """Test TOGAF viewpoints correctly route to togaf_method."""
        test_cases = [
            "Layered viewpoint for system overview",
            "Use application usage viewpoint",
            "Create business function viewpoint",
            "Design technology platform viewpoint",
            "Implementation viewpoint for architecture"  # Avoid "service" which is ArchiMate
        ]

        for query in test_cases:
            with self.subTest(query=query):
                route = self.router.route(query)
                self.assertEqual(route, "togaf_method",
                               f"Query '{query}' should route to togaf_method")

    # PRIORITY 3 TESTS: Everything else → unstructured_docs
    def test_general_queries_route_to_unstructured_docs(self):
        """Test general queries correctly route to unstructured_docs."""
        test_cases = [
            "Project management best practices",
            "Team collaboration strategies",
            "Software development lifecycle",
            "Data security guidelines",
            "Cloud migration planning",
            "Performance optimization techniques"
        ]

        for query in test_cases:
            with self.subTest(query=query):
                route = self.router.route(query)
                self.assertEqual(route, "unstructured_docs",
                               f"Query '{query}' should route to unstructured_docs")

    # EDGE CASE TESTS
    def test_empty_queries_route_to_unstructured_docs(self):
        """Test empty queries route to unstructured_docs."""
        empty_queries = ["", "   ", "\n\t  "]

        for query in empty_queries:
            with self.subTest(query=repr(query)):
                route = self.router.route(query)
                self.assertEqual(route, "unstructured_docs",
                               f"Empty query should route to unstructured_docs")

    def test_mixed_domain_queries_follow_priority(self):
        """Test queries with mixed domain terms follow strict priority."""
        # IEC terms should win over TOGAF terms (Priority 1 > Priority 2)
        mixed_query = "Use TOGAF Phase B to model ActivePower capability"
        route = self.router.route(mixed_query)
        self.assertEqual(route, "structured_model",
                        "IEC/ArchiMate terms should take priority over TOGAF terms")

    def test_case_insensitive_routing(self):
        """Test routing is case insensitive."""
        test_cases = [
            ("ACTIVEPOWER measurement", "structured_model"),
            ("phase b architecture", "togaf_method"),
            ("Business CAPABILITY design", "structured_model"),
            ("VIEWPOINT implementation", "togaf_method")
        ]

        for query, expected_route in test_cases:
            with self.subTest(query=query):
                route = self.router.route(query)
                self.assertEqual(route, expected_route,
                               f"Case insensitive routing failed for '{query}'")

    # PERFORMANCE TESTS
    def test_routing_performance_under_50ms(self):
        """Test routing decisions complete under 50ms target."""
        test_queries = [
            "What capability for grid congestion management?",
            "Phase B business architecture implementation",
            "General project management guidelines",
            "Configure ActivePower measurement with IEC 61968",
            "Design layered viewpoint for system overview"
        ]

        for query in test_queries:
            with self.subTest(query=query):
                start_time = time.perf_counter()
                route = self.router.route(query)
                duration_ms = (time.perf_counter() - start_time) * 1000

                self.assertLess(duration_ms, 50,
                              f"Routing took {duration_ms:.1f}ms (> 50ms target) for '{query}'")
                self.assertIn(route, ["structured_model", "togaf_method", "unstructured_docs"])

    def test_batch_routing_performance(self):
        """Test performance with batch of queries."""
        queries = [
            "ActivePower measurement", "Phase B architecture", "Project management",
            "GridCongestion analysis", "TOGAF viewpoint", "Data security",
            "Capability modeling", "ADM methodology", "Team collaboration"
        ] * 10  # 90 queries total

        start_time = time.perf_counter()
        routes = [self.router.route(q) for q in queries]
        total_time_ms = (time.perf_counter() - start_time) * 1000
        avg_time_ms = total_time_ms / len(queries)

        self.assertLess(avg_time_ms, 50,
                       f"Average routing time {avg_time_ms:.1f}ms exceeds 50ms target")
        self.assertEqual(len(routes), len(queries))

    # VOCABULARY TESTS
    def test_vocabulary_loading(self):
        """Test vocabulary loading and statistics."""
        stats = self.router.get_routing_stats()

        self.assertTrue(stats["vocabularies_loaded"])
        self.assertGreater(stats["iec_energy_terms_count"], 0)
        self.assertGreater(stats["togaf_archimate_terms_count"], 0)
        self.assertEqual(stats["performance_target_ms"], 50)

    def test_routing_explanation(self):
        """Test routing explanation for debugging."""
        query = "Model ActivePower capability using IEC 61968"
        explanation = self.router.explain_routing(query)

        self.assertEqual(explanation["query"], query)
        self.assertEqual(explanation["route"], "structured_model")
        self.assertIn("activepower", explanation["analysis"]["iec_energy_matches"])
        self.assertIn("capability", explanation["analysis"]["archimate_matches"])

    def test_vocabulary_reload(self):
        """Test vocabulary reloading functionality."""
        # Should not raise exceptions
        self.router.reload_vocabularies()

        # Verify router still works after reload
        route = self.router.route("ActivePower measurement")
        self.assertEqual(route, "structured_model")

    # COMPREHENSIVE ROUTING VERIFICATION
    def test_all_specified_examples_route_correctly(self):
        """Test all examples from requirements route correctly."""
        required_test_cases = [
            ("What capability for grid congestion?", "structured_model"),
            ("Model ActivePower measurement", "structured_model"),
            ("Phase B business architecture", "togaf_method"),
            ("Layered viewpoint for system overview", "togaf_method"),
            ("Project management best practices", "unstructured_docs")
        ]

        for query, expected_route in required_test_cases:
            with self.subTest(query=query):
                actual_route = self.router.route(query)
                self.assertEqual(actual_route, expected_route,
                               f"Required test case failed: '{query}' → expected {expected_route}, got {actual_route}")

    def test_priority_order_enforcement(self):
        """Test strict priority order is enforced."""
        # Create queries that could match multiple categories
        priority_tests = [
            # IEC term + TOGAF term → should choose IEC (Priority 1)
            ("Phase B ActivePower architecture", "structured_model"),

            # ArchiMate element + TOGAF → should choose ArchiMate (Priority 1)
            ("TOGAF capability viewpoint", "structured_model"),

            # Only TOGAF terms → should choose TOGAF (Priority 2)
            ("Phase B viewpoint methodology", "togaf_method"),

            # No domain terms → should choose unstructured (Priority 3)
            ("Database performance optimization", "unstructured_docs")
        ]

        for query, expected_route in priority_tests:
            with self.subTest(query=query):
                actual_route = self.router.route(query)
                self.assertEqual(actual_route, expected_route,
                               f"Priority order test failed: '{query}' → expected {expected_route}, got {actual_route}")


if __name__ == '__main__':
    unittest.main(verbosity=2)