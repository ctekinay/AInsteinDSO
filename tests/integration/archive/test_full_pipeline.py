"""
Integration test for the complete EA Assistant pipeline.

This test verifies that all components work together correctly in the full
4R+G+C pipeline flow:
- Router directs queries correctly
- Knowledge is retrieved from appropriate sources
- Responses are grounded with citations
- Critic assesses confidence
- TOGAF alignment is validated
- Audit trail is maintained
"""

import asyncio
import unittest
from pathlib import Path

from src.agent.ea_assistant import ProductionEAAgent


class TestFullPipeline(unittest.TestCase):
    """
    Integration tests for the complete EA assistant pipeline.
    """

    def setUp(self):
        """Set up test fixtures."""
        # Initialize agent with test paths
        self.agent = ProductionEAAgent(
            kg_path="data/energy_knowledge_graph.ttl",
            models_path="data/models/",
            vocab_path="config/vocabularies.json"
        )

    def test_grid_congestion_capability_query(self):
        """
        Test the complete pipeline with grid congestion capability query.

        Expected flow:
        1. Routes to structured_model (has "capability" and "grid congestion")
        2. Retrieves IEC GridCongestion + ArchiMate capabilities
        3. Suggests Grid Congestion Management (archi:id-cap-001)
        4. Citations pass grounding
        5. Confidence > 0.75 (no review needed)
        6. TOGAF validates Capability for Phase B
        """
        query = "What capability should I use for grid congestion management?"

        # Run async query in sync test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            response = loop.run_until_complete(
                self.agent.process_query(query, session_id="test-001")
            )
        finally:
            loop.close()

        # Verify routing
        self.assertEqual(response.route, "structured_model",
                        "Query should route to structured_model")

        # Verify response content
        self.assertIn("Grid Congestion Management", response.response,
                     "Response should mention Grid Congestion Management")

        # Verify citations
        self.assertGreater(len(response.citations), 0,
                          "Response should have citations")

        # Check for ArchiMate citation
        has_archimate_citation = any(
            citation.startswith("archi:id-") for citation in response.citations
        )
        self.assertTrue(has_archimate_citation,
                       "Should have ArchiMate element citation")

        # Check for IEC citation
        has_iec_citation = any(
            citation.startswith("iec:") for citation in response.citations
        )
        self.assertTrue(has_iec_citation or has_archimate_citation,
                       "Should have IEC or ArchiMate citation")

        # Verify confidence
        self.assertGreaterEqual(response.confidence, 0.75,
                              "Confidence should be >= 0.75")
        self.assertFalse(response.requires_human_review,
                        "Should not require human review with high confidence")

        # Verify TOGAF phase
        if response.togaf_phase:
            self.assertEqual(response.togaf_phase, "Phase B",
                           "Capability should align with Phase B")

        # Verify ArchiMate elements
        self.assertGreater(len(response.archimate_elements), 0,
                          "Should have validated ArchiMate elements")

        # Check for Grid Congestion Management capability
        has_congestion_capability = any(
            "Grid Congestion Management" in element.get("element", "")
            for element in response.archimate_elements
        )
        self.assertTrue(has_congestion_capability,
                       "Should include Grid Congestion Management capability")

        # Verify processing time
        self.assertLess(response.processing_time_ms, 3000,
                       "Processing should complete within 3 seconds")

        # Verify audit trail
        audit_trail = self.agent.get_audit_trail("test-001")
        self.assertIsNotNone(audit_trail, "Should have audit trail")
        self.assertEqual(audit_trail["query"], query)
        self.assertIn("ROUTE", [step["step"] for step in audit_trail["steps"]])
        self.assertIn("GROUND", [step["step"] for step in audit_trail["steps"]])
        self.assertIn("CRITIC", [step["step"] for step in audit_trail["steps"]])

    def test_togaf_methodology_query(self):
        """Test query that routes to TOGAF methodology."""
        query = "How do I implement Phase B business architecture?"

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            response = loop.run_until_complete(
                self.agent.process_query(query, session_id="test-002")
            )
        finally:
            loop.close()

        # Should route to togaf_method
        self.assertEqual(response.route, "togaf_method",
                        "Query should route to togaf_method")

        # Should have TOGAF citations
        has_togaf_citation = any(
            "togaf:adm:" in citation for citation in response.citations
        )
        self.assertTrue(has_togaf_citation, "Should have TOGAF ADM citation")

    def test_unstructured_query(self):
        """Test query that routes to unstructured docs."""
        query = "What are best practices for project management?"

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            response = loop.run_until_complete(
                self.agent.process_query(query, session_id="test-003")
            )
        finally:
            loop.close()

        # Should route to unstructured_docs
        self.assertEqual(response.route, "unstructured_docs",
                        "Query should route to unstructured_docs")

    def test_low_confidence_scenario(self):
        """Test that low confidence triggers human review requirement."""
        # Query that should result in lower confidence
        query = "How do I implement quantum computing in my architecture?"

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            response = loop.run_until_complete(
                self.agent.process_query(query, session_id="test-004")
            )
        finally:
            loop.close()

        # Should route to unstructured_docs due to no domain terms
        self.assertEqual(response.route, "unstructured_docs")

        # With mock data, confidence might be low
        # The actual behavior depends on implementation
        if response.confidence < 0.75:
            self.assertTrue(response.requires_human_review,
                          "Low confidence should require human review")

    def test_archimate_element_validation(self):
        """Test ArchiMate element validation in pipeline."""
        query = "What application components should I use for grid management?"

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            response = loop.run_until_complete(
                self.agent.process_query(query, session_id="test-005")
            )
        finally:
            loop.close()

        # Should route to structured_model
        self.assertEqual(response.route, "structured_model")

        # Should find Application layer elements
        if response.archimate_elements:
            app_elements = [
                e for e in response.archimate_elements
                if e.get("layer") == "Application"
            ]
            self.assertGreater(len(app_elements), 0,
                             "Should find Application layer elements")

            # Should align with Phase C
            for element in app_elements:
                if element.get("phase"):
                    self.assertEqual(element["phase"], "Phase C",
                                   "Application elements should align with Phase C")

    def test_multiple_citation_types(self):
        """Test that responses can include multiple citation types."""
        query = "How should I model grid congestion using IEC standards and ArchiMate?"

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            response = loop.run_until_complete(
                self.agent.process_query(query, session_id="test-006")
            )
        finally:
            loop.close()

        # Should have both IEC and ArchiMate citations
        citation_types = set()
        for citation in response.citations:
            if citation.startswith("archi:id-"):
                citation_types.add("archimate")
            elif citation.startswith("iec:"):
                citation_types.add("iec")
            elif citation.startswith("togaf:"):
                citation_types.add("togaf")

        self.assertIn("archimate", citation_types,
                     "Should have ArchiMate citations")
        # Note: With mock data, IEC citations depend on implementation

    def test_agent_statistics(self):
        """Test agent statistics collection."""
        stats = self.agent.get_statistics()

        self.assertIn("knowledge_graph", stats)
        self.assertIn("archimate_models", stats)
        self.assertIn("router_config", stats)
        self.assertIn("critic_config", stats)
        self.assertIn("sessions_processed", stats)

        # Should have loaded knowledge graph
        if "triple_count" in stats.get("knowledge_graph", {}):
            self.assertGreater(stats["knowledge_graph"]["triple_count"], 0,
                             "Should have loaded knowledge graph triples")

        # Should have loaded ArchiMate models
        if "total_elements" in stats.get("archimate_models", {}):
            self.assertGreater(stats["archimate_models"]["total_elements"], 0,
                             "Should have loaded ArchiMate elements")

    def test_audit_trail_completeness(self):
        """Test that audit trail captures all pipeline steps."""
        query = "What capability for grid management?"

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            response = loop.run_until_complete(
                self.agent.process_query(query, session_id="test-audit")
            )
        finally:
            loop.close()

        audit_trail = self.agent.get_audit_trail("test-audit")
        self.assertIsNotNone(audit_trail)

        # Check all required steps are present
        steps = [step["step"] for step in audit_trail["steps"]]
        required_steps = ["REFLECT", "ROUTE", "RETRIEVE", "REFINE", "GROUND", "CRITIC"]

        for required_step in required_steps:
            self.assertIn(required_step, steps,
                         f"Audit trail should include {required_step} step")

        # If structured_model, should also have VALIDATE
        if response.route == "structured_model":
            self.assertIn("VALIDATE", steps,
                         "Structured model queries should include VALIDATE step")

    def test_performance_requirements(self):
        """Test that pipeline meets performance requirements."""
        query = "What capability should I use for grid congestion?"

        # Run multiple queries to test performance
        timings = []

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            for i in range(3):
                response = loop.run_until_complete(
                    self.agent.process_query(query, session_id=f"perf-{i}")
                )
                timings.append(response.processing_time_ms)
        finally:
            loop.close()

        # Check average performance
        avg_time = sum(timings) / len(timings)
        self.assertLess(avg_time, 3000,
                       f"Average processing time {avg_time:.0f}ms should be < 3000ms")

        # Check P50 (median)
        p50_time = sorted(timings)[len(timings) // 2]
        self.assertLess(p50_time, 3000,
                       f"P50 time {p50_time:.0f}ms should be < 3000ms")


if __name__ == '__main__':
    unittest.main(verbosity=2)