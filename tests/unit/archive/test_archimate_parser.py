"""
Comprehensive unit tests for ArchiMate parser - Model element extraction and TOGAF validation.

These tests verify that the ArchiMate parser correctly:
1. Loads ArchiMate Exchange Format XML files
2. Extracts elements with id, name, type, layer
3. Validates TOGAF phase alignment
4. Enables real citations (archi:id-xxx)
5. Provides model query capabilities

The parser is essential for grounding responses in actual model elements
and ensuring TOGAF compliance in architectural guidance.
"""

import tempfile
import unittest
from pathlib import Path

from src.archimate.parser import ArchiMateParser, ArchiMateElement


class TestArchiMateParser(unittest.TestCase):
    """
    Test suite for ArchiMate parser functionality.
    """

    def setUp(self):
        """Set up test fixtures."""
        self.parser = ArchiMateParser()
        self.sample_model_path = "data/models/sample.archimate"

    def test_load_sample_model(self):
        """Test loading the sample ArchiMate model."""
        success = self.parser.load_model(self.sample_model_path)

        self.assertTrue(success, "Should successfully load sample model")
        self.assertEqual(self.parser.model_name, "sample")
        self.assertGreater(len(self.parser.elements), 0, "Should extract elements from model")

        # Verify we have the expected minimum elements
        # (2 Business, 2 Application, 1 Technology as per requirements)
        business_elements = self.parser.get_elements_by_layer("Business")
        application_elements = self.parser.get_elements_by_layer("Application")
        technology_elements = self.parser.get_elements_by_layer("Technology")

        self.assertGreaterEqual(len(business_elements), 2, "Should have at least 2 Business elements")
        self.assertGreaterEqual(len(application_elements), 2, "Should have at least 2 Application elements")
        self.assertGreaterEqual(len(technology_elements), 1, "Should have at least 1 Technology element")

    def test_load_nonexistent_model(self):
        """Test loading a non-existent model file."""
        success = self.parser.load_model("nonexistent.archimate")
        self.assertFalse(success, "Should fail to load non-existent model")

    def test_extract_elements_by_layer(self):
        """Test extracting elements by layer."""
        self.parser.load_model(self.sample_model_path)

        # Test Business layer
        business_elements = self.parser.get_elements_by_layer("Business")
        self.assertGreater(len(business_elements), 0, "Should find Business layer elements")

        for element in business_elements:
            self.assertEqual(element.layer, "Business")
            self.assertIn(element.type, ["Capability", "BusinessProcess", "BusinessActor", "BusinessService"])

        # Test Application layer
        application_elements = self.parser.get_elements_by_layer("Application")
        self.assertGreater(len(application_elements), 0, "Should find Application layer elements")

        for element in application_elements:
            self.assertEqual(element.layer, "Application")

        # Test Technology layer
        technology_elements = self.parser.get_elements_by_layer("Technology")
        self.assertGreater(len(technology_elements), 0, "Should find Technology layer elements")

        for element in technology_elements:
            self.assertEqual(element.layer, "Technology")

    def test_find_element_by_id(self):
        """Test finding elements by ID for citation validation."""
        self.parser.load_model(self.sample_model_path)

        # Test finding existing elements
        capability = self.parser.get_element_by_id("cap-001")
        self.assertIsNotNone(capability, "Should find capability element")
        self.assertEqual(capability.name, "Grid Congestion Management")
        self.assertEqual(capability.type, "Capability")
        self.assertEqual(capability.layer, "Business")

        process = self.parser.get_element_by_id("proc-001")
        self.assertIsNotNone(process, "Should find process element")
        self.assertEqual(process.name, "Grid Monitoring Process")
        self.assertEqual(process.type, "BusinessProcess")

        # Test finding non-existent element
        nonexistent = self.parser.get_element_by_id("nonexistent-id")
        self.assertIsNone(nonexistent, "Should return None for non-existent ID")

    def test_validate_togaf_alignment(self):
        """Test TOGAF phase alignment validation."""
        self.parser.load_model(self.sample_model_path)

        # Get test elements
        capability = self.parser.get_element_by_id("cap-001")
        app_component = self.parser.get_element_by_id("comp-001")
        tech_node = self.parser.get_element_by_id("node-001")

        self.assertIsNotNone(capability)
        self.assertIsNotNone(app_component)
        self.assertIsNotNone(tech_node)

        # Test correct alignments
        self.assertTrue(self.parser.validate_togaf_alignment(capability, "Phase B"),
                       "Business Capability should align with Phase B")

        self.assertTrue(self.parser.validate_togaf_alignment(app_component, "Phase C"),
                       "Application Component should align with Phase C")

        self.assertTrue(self.parser.validate_togaf_alignment(tech_node, "Phase D"),
                       "Technology Node should align with Phase D")

        # Test incorrect alignments
        self.assertFalse(self.parser.validate_togaf_alignment(capability, "Phase D"),
                        "Business Capability should NOT align with Phase D")

        self.assertFalse(self.parser.validate_togaf_alignment(tech_node, "Phase B"),
                        "Technology Node should NOT align with Phase B")

        self.assertFalse(self.parser.validate_togaf_alignment(app_component, "Phase B"),
                        "Application Component should NOT align with Phase B")

    def test_citation_validation(self):
        """Test citation validation against model elements."""
        self.parser.load_model(self.sample_model_path)

        # Test valid citations
        valid_citation = "archi:id-cap-001"
        element = self.parser.validate_citation(valid_citation)
        self.assertIsNotNone(element, "Should validate correct citation")
        self.assertEqual(element.id, "cap-001")
        self.assertEqual(element.name, "Grid Congestion Management")

        # Test invalid citation format
        invalid_format = "invalid:citation"
        element = self.parser.validate_citation(invalid_format)
        self.assertIsNone(element, "Should reject invalid citation format")

        # Test non-existent element citation
        nonexistent_citation = "archi:id-nonexistent"
        element = self.parser.validate_citation(nonexistent_citation)
        self.assertIsNone(element, "Should reject citation to non-existent element")

    def test_get_citation_id(self):
        """Test citation ID generation for elements."""
        self.parser.load_model(self.sample_model_path)

        capability = self.parser.get_element_by_id("cap-001")
        self.assertIsNotNone(capability)

        citation_id = capability.get_citation_id()
        self.assertEqual(citation_id, "archi:id-cap-001")

    def test_get_elements_by_type(self):
        """Test getting elements by specific type."""
        self.parser.load_model(self.sample_model_path)

        # Test getting capabilities
        capabilities = self.parser.get_elements_by_type("Capability")
        self.assertGreater(len(capabilities), 0, "Should find capability elements")

        for capability in capabilities:
            self.assertEqual(capability.type, "Capability")
            self.assertEqual(capability.layer, "Business")

        # Test getting application components
        components = self.parser.get_elements_by_type("ApplicationComponent")
        self.assertGreater(len(components), 0, "Should find application components")

        for component in components:
            self.assertEqual(component.type, "ApplicationComponent")
            self.assertEqual(component.layer, "Application")

    def test_get_citation_candidates(self):
        """Test finding citation candidates for query terms."""
        self.parser.load_model(self.sample_model_path)

        # Test finding candidates for "grid" query
        grid_candidates = self.parser.get_citation_candidates(["grid"])
        self.assertGreater(len(grid_candidates), 0, "Should find candidates for 'grid' query")

        grid_names = [c.name.lower() for c in grid_candidates]
        self.assertTrue(any("grid" in name for name in grid_names),
                       "Candidates should contain 'grid' in names")

        # Test finding candidates for "capability" query
        capability_candidates = self.parser.get_citation_candidates(["capability"])
        self.assertGreater(len(capability_candidates), 0, "Should find candidates for 'capability' query")

        # Test with multiple terms
        multi_candidates = self.parser.get_citation_candidates(["grid", "monitoring"])
        self.assertGreater(len(multi_candidates), 0, "Should find candidates for multiple terms")

    def test_model_summary(self):
        """Test model summary statistics."""
        self.parser.load_model(self.sample_model_path)

        summary = self.parser.get_model_summary()

        self.assertEqual(summary["model_name"], "sample")
        self.assertIn("total_elements", summary)
        self.assertGreater(summary["total_elements"], 0)

        self.assertIn("elements_by_layer", summary)
        layer_counts = summary["elements_by_layer"]

        # Should have elements in Business, Application, and Technology layers
        self.assertIn("Business", layer_counts)
        self.assertIn("Application", layer_counts)
        self.assertIn("Technology", layer_counts)

        self.assertGreater(layer_counts["Business"], 0)
        self.assertGreater(layer_counts["Application"], 0)
        self.assertGreater(layer_counts["Technology"], 0)

        # Should include TOGAF mapping
        self.assertIn("togaf_layer_mapping", summary)

    def test_element_properties_and_documentation(self):
        """Test extraction of element properties and documentation."""
        self.parser.load_model(self.sample_model_path)

        capability = self.parser.get_element_by_id("cap-001")
        self.assertIsNotNone(capability)

        # Check documentation
        self.assertIn("grid congestion", capability.documentation.lower())

        # Check properties
        self.assertIn("criticality", capability.properties)
        self.assertEqual(capability.properties["criticality"], "high")

    def test_empty_model_summary(self):
        """Test model summary when no model is loaded."""
        empty_parser = ArchiMateParser()
        summary = empty_parser.get_model_summary()

        self.assertIn("error", summary)
        self.assertEqual(summary["error"], "No model loaded")

    def test_invalid_xml_model(self):
        """Test handling of invalid XML files."""
        # Create temporary invalid XML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write("<?xml version='1.0'?><invalid><unclosed>")
            invalid_path = f.name

        try:
            success = self.parser.load_model(invalid_path)
            self.assertFalse(success, "Should fail to load invalid XML")
        finally:
            Path(invalid_path).unlink()

    def test_archimate_element_string_representation(self):
        """Test ArchiMateElement string representation."""
        element = ArchiMateElement(
            id="test-001",
            name="Test Element",
            type="TestType",
            layer="Test",
            properties={},
            documentation=""
        )

        str_repr = str(element)
        self.assertIn("Test Element", str_repr)
        self.assertIn("TestType", str_repr)
        self.assertIn("Test", str_repr)

    def test_required_sample_model_elements(self):
        """Test that sample model contains all required elements from specification."""
        self.parser.load_model(self.sample_model_path)

        # Should have at least 2 Business elements (Capability, Process)
        business_elements = self.parser.get_elements_by_layer("Business")
        business_types = [e.type for e in business_elements]

        self.assertIn("Capability", business_types, "Should have Business Capability")
        self.assertIn("BusinessProcess", business_types, "Should have Business Process")

        # Should have at least 2 Application elements (Component, Service)
        application_elements = self.parser.get_elements_by_layer("Application")
        app_types = [e.type for e in application_elements]

        self.assertIn("ApplicationComponent", app_types, "Should have Application Component")
        # Note: Sample has ApplicationService, which satisfies "Service" requirement

        # Should have at least 1 Technology element (Node)
        technology_elements = self.parser.get_elements_by_layer("Technology")
        tech_types = [e.type for e in technology_elements]

        self.assertIn("Node", tech_types, "Should have Technology Node")

        # Verify specific required elements exist
        capability = self.parser.get_element_by_id("cap-001")
        self.assertIsNotNone(capability, "Should have capability with ID cap-001")
        self.assertEqual(capability.type, "Capability")

        process = self.parser.get_element_by_id("proc-001")
        self.assertIsNotNone(process, "Should have process with ID proc-001")
        self.assertEqual(process.type, "BusinessProcess")

        component = self.parser.get_element_by_id("comp-001")
        self.assertIsNotNone(component, "Should have component with ID comp-001")
        self.assertEqual(component.type, "ApplicationComponent")

        node = self.parser.get_element_by_id("node-001")
        self.assertIsNotNone(node, "Should have node with ID node-001")
        self.assertEqual(node.type, "Node")


if __name__ == '__main__':
    unittest.main(verbosity=2)