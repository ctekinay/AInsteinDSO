"""
ArchiMate Model Parser - Extract elements for citation and TOGAF validation.

This parser processes ArchiMate Exchange Format XML files to extract model elements
with their IDs, names, types, and layers. It enables:

1. Real citations: archi:id-cap-001 can be validated against actual model elements
2. TOGAF compliance: Ensures Phase B uses Business elements, not Technology
3. Model queries: "What capabilities exist?" can query actual models

The parser supports the full ArchiMate metamodel and provides TOGAF phase alignment
validation to ensure architectural consistency.
"""

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class ArchiMateElement:
    """
    Represents a parsed ArchiMate element.

    This dataclass captures essential element information for citation
    validation and TOGAF alignment checks.
    """
    id: str                    # Element ID for citations (archi:id-xxx)
    name: str                  # Human-readable element name
    type: str                  # ArchiMate element type (e.g., BusinessCapability)
    layer: str                 # ArchiMate layer (Business, Application, Technology)
    properties: Dict[str, str] # Additional element properties
    documentation: str         # Element documentation/description

    def get_citation_id(self) -> str:
        """
        Get the citation ID for this element.
        Normalizes to expected format for grounding check.
        """
        # Normalize to expected format - handle various ID patterns
        if self.id.startswith("id-"):
            return f"archi:{self.id}"
        elif self.id.startswith("archi:"):
            return self.id
        else:
            return f"archi:id-{self.id}"

    def __str__(self) -> str:
        return f"{self.type}: {self.name} ({self.layer} layer)"


class ArchiMateParser:
    """
    Parser for ArchiMate Exchange Format XML files.

    Extracts model elements and provides TOGAF phase alignment validation
    to ensure architectural consistency and enable proper citations.
    """

    # ArchiMate namespace
    ARCHIMATE_NS = "http://www.archimatetool.com/archimate"

    # Layer mappings for TOGAF alignment
    TOGAF_LAYER_MAPPING = {
        "Business": ["Phase B"],
        "Application": ["Phase C"],
        "Technology": ["Phase D"],
        "Physical": ["Phase D"],
        "Strategy": ["Phase A"],
        "Implementation": ["Phase F", "Phase G", "Phase H"],
        "Motivation": ["Phase A"]
    }

    # ArchiMate element type to layer mapping
    ELEMENT_LAYER_MAPPING = {
        # Business Layer
        "BusinessActor": "Business",
        "BusinessRole": "Business",
        "BusinessCollaboration": "Business",
        "BusinessInterface": "Business",
        "BusinessProcess": "Business",
        "BusinessFunction": "Business",
        "BusinessInteraction": "Business",
        "BusinessEvent": "Business",
        "BusinessService": "Business",
        "BusinessObject": "Business",
        "Contract": "Business",
        "Representation": "Business",
        "Product": "Business",
        "Capability": "Business",
        "CourseOfAction": "Business",
        "Resource": "Business",

        # Application Layer
        "ApplicationComponent": "Application",
        "ApplicationCollaboration": "Application",
        "ApplicationInterface": "Application",
        "ApplicationFunction": "Application",
        "ApplicationInteraction": "Application",
        "ApplicationProcess": "Application",
        "ApplicationEvent": "Application",
        "ApplicationService": "Application",
        "DataObject": "Application",

        # Technology Layer
        "Node": "Technology",
        "Device": "Technology",
        "SystemSoftware": "Technology",
        "TechnologyCollaboration": "Technology",
        "TechnologyInterface": "Technology",
        "Path": "Technology",
        "CommunicationNetwork": "Technology",
        "TechnologyFunction": "Technology",
        "TechnologyProcess": "Technology",
        "TechnologyInteraction": "Technology",
        "TechnologyEvent": "Technology",
        "TechnologyService": "Technology",
        "Artifact": "Technology",

        # Physical Layer
        "Equipment": "Physical",
        "Facility": "Physical",
        "DistributionNetwork": "Physical",
        "Material": "Physical",

        # Strategy Layer
        "Stakeholder": "Strategy",
        "Driver": "Strategy",
        "Assessment": "Strategy",
        "Goal": "Strategy",
        "Outcome": "Strategy",
        "Principle": "Strategy",
        "Requirement": "Strategy",
        "Constraint": "Strategy",
        "Meaning": "Strategy",
        "Value": "Strategy"
    }

    def __init__(self):
        """Initialize the ArchiMate parser."""
        self.elements = {}  # id -> ArchiMateElement
        self.model_path = None
        self.model_name = None

        logger.info("ArchiMate parser initialized")

    def load_model(self, path: str) -> bool:
        """
        Load an ArchiMate model from XML file.

        Args:
            path: Path to the ArchiMate XML file

        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            model_path = Path(path)
            if not model_path.exists():
                logger.error(f"Model file not found: {path}")
                return False

            logger.info(f"Loading ArchiMate model from: {path}")

            # Parse XML
            tree = ET.parse(path)
            root = tree.getroot()

            # Store model metadata
            self.model_path = str(model_path)
            self.model_name = model_path.stem

            # Clear existing elements
            self.elements = {}

            # Extract elements
            self._extract_elements(root)

            logger.info(f"Loaded {len(self.elements)} elements from model '{self.model_name}'")
            return True

        except ET.ParseError as e:
            logger.error(f"XML parsing error: {e}")
            return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def _extract_elements(self, root: ET.Element) -> None:
        """
        Extract ArchiMate elements from XML root.

        Args:
            root: XML root element
        """
        # Handle different XML structures - ArchiMate files can vary
        elements_found = 0

        # Look for elements in various XML structures
        for element in root.iter():
            if self._is_archimate_element(element):
                archimate_element = self._parse_element(element)
                if archimate_element:
                    self.elements[archimate_element.id] = archimate_element
                    elements_found += 1

        logger.debug(f"Extracted {elements_found} ArchiMate elements")

    def _is_archimate_element(self, element: ET.Element) -> bool:
        """
        Check if XML element is an ArchiMate model element.

        Args:
            element: XML element to check

        Returns:
            True if this is an ArchiMate model element
        """
        tag = element.tag.split('}')[-1] if '}' in element.tag else element.tag

        # Check if this is an element with xsi:type attribute
        xsi_type = element.get('{http://www.w3.org/2001/XMLSchema-instance}type')
        if not xsi_type:
            xsi_type = element.get('xsi:type')

        if xsi_type:
            # Handle archimate: prefix in type
            if ':' in xsi_type:
                element_type = xsi_type.split(':')[-1]
            else:
                element_type = xsi_type

            # Check if this is a known ArchiMate element type
            if element_type in self.ELEMENT_LAYER_MAPPING:
                return True

        # Check if tag itself is a known ArchiMate element type
        return tag in self.ELEMENT_LAYER_MAPPING

    def _parse_element(self, xml_element: ET.Element) -> Optional[ArchiMateElement]:
        """
        Parse a single ArchiMate element from XML.

        Args:
            xml_element: XML element to parse

        Returns:
            ArchiMateElement instance or None if parsing failed
        """
        try:
            # Get element attributes
            element_id = xml_element.get('id') or xml_element.get('identifier')
            name = xml_element.get('name', '')

            # Determine element type - handle both namespaced and non-namespaced attributes
            element_type = xml_element.get('{http://www.w3.org/2001/XMLSchema-instance}type')
            if not element_type:
                element_type = xml_element.get('xsi:type')

            if not element_type:
                tag = xml_element.tag.split('}')[-1] if '}' in xml_element.tag else xml_element.tag
                element_type = tag

            # Clean up type name (remove namespace prefixes)
            if ':' in element_type:
                element_type = element_type.split(':')[-1]

            # Skip if essential attributes missing
            if not element_id or not element_type:
                return None

            # Get layer from element type
            layer = self.ELEMENT_LAYER_MAPPING.get(element_type, "Unknown")

            # Extract properties
            properties = {}
            for prop in xml_element.findall('.//property'):
                key = prop.get('key', prop.get('name', ''))
                value = prop.get('value', prop.text or '')
                if key:
                    properties[key] = value

            # Extract documentation
            documentation = ""
            doc_element = xml_element.find('.//documentation')
            if doc_element is not None:
                documentation = doc_element.text or ""

            # Use name from nested name element if main name is empty
            if not name:
                name_element = xml_element.find('.//name')
                if name_element is not None:
                    name = name_element.text

            # Handle unnamed elements - use type and ID as fallback
            if not name or name.strip() == "" or name == "unnamed":
                # Create a meaningful name from type and ID
                name = f"{element_type}_{element_id[-8:]}"  # Use type + last 8 chars of ID
                logger.debug(f"Element has no name, using: {name} for {element_id}")

            element = ArchiMateElement(
                id=element_id,
                name=name,
                type=element_type,
                layer=layer,
                properties=properties,
                documentation=documentation
            )

            logger.debug(f"Parsed element: {element}")
            return element

        except Exception as e:
            logger.warning(f"Error parsing element: {e}")
            return None
    
    def citation_exists(self, archi_id: str) -> bool:
        """
        Check if ArchiMate element ID exists in loaded models.
        
        Args:
            archi_id: Citation in format 'archi:id-{element_id}'
            
        Returns:
            True if element exists in loaded models
        """
        if not archi_id.startswith("archi:id-"):
            return False
        
        # Extract element ID
        element_id = archi_id.replace("archi:id-", "")
        
        # Check if element exists in loaded models
        return element_id in self.elements


    def get_valid_citations(self) -> List[str]:
        """
        Get all valid archi:id-* citations from loaded models.
        
        Returns:
            List of archi:id-* citation strings
        """
        return [f"archi:id-{element_id}" for element_id in self.elements.keys()]


    def get_element_by_citation(self, citation: str) -> Optional[ArchiMateElement]:
        """
        Retrieve element by citation ID.
        
        Args:
            citation: Citation in format 'archi:id-{element_id}'
            
        Returns:
            ArchiMateElement if found, None otherwise
        """
        if not self.citation_exists(citation):
            return None
        
        element_id = citation.replace("archi:id-", "")
        return self.elements.get(element_id)

    def get_element_by_id(self, element_id: str) -> Optional[ArchiMateElement]:
        """
        Get element by ID for citation validation.

        Args:
            element_id: Element ID to find

        Returns:
            ArchiMateElement if found, None otherwise
        """
        return self.elements.get(element_id)

    def get_elements_by_layer(self, layer: str) -> List[ArchiMateElement]:
        """
        Get all elements in a specific layer for TOGAF alignment.

        Args:
            layer: Layer name (Business, Application, Technology, etc.)

        Returns:
            List of elements in the specified layer
        """
        return [element for element in self.elements.values() if element.layer == layer]

    def get_elements_by_type(self, element_type: str) -> List[ArchiMateElement]:
        """
        Get all elements of a specific type.

        Args:
            element_type: ArchiMate element type

        Returns:
            List of elements of the specified type
        """
        return [element for element in self.elements.values() if element.type == element_type]

    def validate_togaf_alignment(self, element: ArchiMateElement, phase: str) -> bool:
        """
        Validate if element is appropriate for TOGAF phase.

        Args:
            element: ArchiMate element to validate
            phase: TOGAF ADM phase (e.g., "Phase B", "Phase C")

        Returns:
            True if element is appropriate for the phase
        """
        allowed_phases = self.TOGAF_LAYER_MAPPING.get(element.layer, [])
        is_aligned = phase in allowed_phases

        if not is_aligned:
            logger.debug(f"TOGAF alignment mismatch: {element.type} ({element.layer}) not suitable for {phase}")

        return is_aligned

    def get_citation_candidates(self, query_terms: List[str]) -> List[ArchiMateElement]:
        """
        Find elements that could provide citations for query terms.

        Args:
            query_terms: List of terms to search for

        Returns:
            List of matching elements that could be cited
        """
        candidates = []
        query_terms_lower = [term.lower() for term in query_terms]

        for element in self.elements.values():
            # Check if element name or type matches any query term
            element_text = f"{element.name} {element.type}".lower()

            for term in query_terms_lower:
                if term in element_text:
                    candidates.append(element)
                    break

        return candidates

    def get_model_summary(self) -> Dict:
        """
        Get summary statistics of the loaded model.

        Returns:
            Dictionary with model statistics
        """
        if not self.elements:
            return {"error": "No model loaded"}

        layer_counts = {}
        type_counts = {}

        for element in self.elements.values():
            layer_counts[element.layer] = layer_counts.get(element.layer, 0) + 1
            type_counts[element.type] = type_counts.get(element.type, 0) + 1

        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "total_elements": len(self.elements),
            "elements_by_layer": layer_counts,
            "elements_by_type": type_counts,
            "togaf_layer_mapping": self.TOGAF_LAYER_MAPPING
        }

    def validate_citation(self, citation: str) -> Optional[ArchiMateElement]:
        """
        Validate a citation against loaded model elements.

        Args:
            citation: Citation in format "archi:id-xxx"

        Returns:
            ArchiMateElement if citation is valid, None otherwise
        """
        if not citation.startswith("archi:id-"):
            return None

        element_id = citation[9:]  # Remove "archi:id-" prefix
        return self.get_element_by_id(element_id)