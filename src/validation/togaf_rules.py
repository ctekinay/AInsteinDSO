"""
TOGAF ADM Rules Engine - Rules-as-Code Implementation.

This module implements TOGAF Architecture Development Method (ADM) rules
with explicit exceptions and concrete guidance passages. It moves beyond
simple heuristics to provide detailed validation based on The Open Group
Architecture Framework specification.

Key Features:
- Phase-Layer mapping rules with documented exceptions
- ADM guidance passage references for all rules
- Cross-layer artifact validation for transition architectures
- Migration planning exception handling
- CI/CD integration ready rule violations
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
from src.archimate.parser import ArchiMateElement

logger = logging.getLogger(__name__)


class TOGAFPhase(Enum):
    """TOGAF ADM Phases with official descriptions."""
    PHASE_A = "Phase A: Architecture Vision"
    PHASE_B = "Phase B: Business Architecture"
    PHASE_C = "Phase C: Information Systems Architectures"
    PHASE_D = "Phase D: Technology Architecture"
    PHASE_E = "Phase E: Opportunities and Solutions"
    PHASE_F = "Phase F: Migration Planning"
    PHASE_G = "Phase G: Implementation Governance"
    PHASE_H = "Phase H: Architecture Change Management"
    REQUIREMENTS = "Requirements Management"


class ViolationSeverity(Enum):
    """Severity levels for TOGAF rule violations."""
    ERROR = "error"          # Hard rule violation
    WARNING = "warning"      # Heuristic violation with exceptions
    INFO = "info"           # Best practice recommendation
    EXCEPTION = "exception"  # Documented exception applied


@dataclass
class TOGAFRuleViolation:
    """
    Represents a TOGAF rule violation with context and guidance.
    """
    element_id: str
    element_type: str
    element_layer: str
    phase: str
    rule_id: str
    severity: ViolationSeverity
    message: str
    adm_reference: str
    guidance_passage: str
    suggested_action: str
    exception_applied: Optional[str] = None


class TOGAFRulesEngine:
    """
    TOGAF ADM Rules Engine with exception handling and ADM guidance.

    Implements concrete rules from TOGAF 9.2 specification with
    documented exceptions for real-world architectural scenarios.
    """

    # Core Phase-Layer Mapping (TOGAF 9.2 ADM)
    CORE_PHASE_LAYER_RULES = {
        "Phase A": {
            "primary_layers": ["Strategy", "Motivation"],
            "adm_reference": "TOGAF 9.2 Section 19.3",
            "guidance": "Phase A focuses on architecture vision, stakeholders, and principles"
        },
        "Phase B": {
            "primary_layers": ["Business"],
            "adm_reference": "TOGAF 9.2 Section 20.3",
            "guidance": "Phase B develops business architecture baseline and target"
        },
        "Phase C": {
            "primary_layers": ["Application", "Data"],
            "adm_reference": "TOGAF 9.2 Section 21.3",
            "guidance": "Phase C develops information systems architectures"
        },
        "Phase D": {
            "primary_layers": ["Technology", "Physical"],
            "adm_reference": "TOGAF 9.2 Section 22.3",
            "guidance": "Phase D develops technology architecture"
        },
        "Phase E": {
            "primary_layers": ["Business", "Application", "Technology"],
            "adm_reference": "TOGAF 9.2 Section 23.3",
            "guidance": "Phase E identifies opportunities and solutions across all layers"
        },
        "Phase F": {
            "primary_layers": ["Implementation", "Migration"],
            "adm_reference": "TOGAF 9.2 Section 24.3",
            "guidance": "Phase F focuses on migration planning and implementation sequencing"
        }
    }

    # Documented Exceptions (Real-world architectural scenarios)
    TOGAF_EXCEPTIONS = {
        "Phase B": {
            "ApplicationInterface": {
                "reason": "Business-Application interfaces in transition architectures",
                "adm_reference": "TOGAF 9.2 Section 20.3.6",
                "guidance": "Application interfaces may be defined in Phase B when they represent business services",
                "conditions": ["transition_architecture", "business_service_interface"]
            },
            "ApplicationService": {
                "reason": "Business services exposed through applications",
                "adm_reference": "TOGAF 9.2 Section 20.3.4",
                "guidance": "Application services representing business capabilities",
                "conditions": ["business_capability_exposure"]
            }
        },
        "Phase C": {
            "BusinessProcess": {
                "reason": "Process-application alignment analysis",
                "adm_reference": "TOGAF 9.2 Section 21.3.3",
                "guidance": "Business processes may be referenced in Phase C for application mapping",
                "conditions": ["process_application_mapping"]
            },
            "TechnologyInterface": {
                "reason": "Data integration technology interfaces",
                "adm_reference": "TOGAF 9.2 Section 21.3.5",
                "guidance": "Technology interfaces for data integration may be specified",
                "conditions": ["data_integration_interface"]
            }
        },
        "Phase D": {
            "ApplicationComponent": {
                "reason": "Technology-application integration points",
                "adm_reference": "TOGAF 9.2 Section 22.3.4",
                "guidance": "Application components may be referenced for technology integration",
                "conditions": ["technology_integration_point"]
            },
            "DataObject": {
                "reason": "Technology-specific data representations",
                "adm_reference": "TOGAF 9.2 Section 22.3.3",
                "guidance": "Data objects may be specified for technology implementation",
                "conditions": ["technology_data_representation"]
            }
        },
        "Phase E": {
            # Phase E allows cross-layer artifacts by design
            "*": {
                "reason": "Cross-layer opportunity identification",
                "adm_reference": "TOGAF 9.2 Section 23.3",
                "guidance": "Phase E explicitly works across all architecture layers",
                "conditions": ["opportunity_analysis"]
            }
        },
        "Phase F": {
            "BusinessCapability": {
                "reason": "Migration impact on business capabilities",
                "adm_reference": "TOGAF 9.2 Section 24.3.2",
                "guidance": "Business capabilities affected by migration must be considered",
                "conditions": ["migration_impact_analysis"]
            },
            "BusinessProcess": {
                "reason": "Process migration and transformation",
                "adm_reference": "TOGAF 9.2 Section 24.3.3",
                "guidance": "Business processes undergoing transformation in migration",
                "conditions": ["process_transformation"]
            }
        }
    }

    # Cross-layer relationship rules
    CROSS_LAYER_RULES = {
        "BusinessCapability_ApplicationService": {
            "description": "Business capabilities realized by application services",
            "phases": ["Phase B", "Phase C", "Phase E"],
            "relationship_type": "realization",
            "adm_reference": "TOGAF 9.2 Capability-Based Planning"
        },
        "ApplicationComponent_TechnologyNode": {
            "description": "Application components deployed on technology nodes",
            "phases": ["Phase C", "Phase D", "Phase E"],
            "relationship_type": "assignment",
            "adm_reference": "TOGAF 9.2 Application/Technology Mapping"
        },
        "BusinessProcess_ApplicationFunction": {
            "description": "Business processes supported by application functions",
            "phases": ["Phase B", "Phase C"],
            "relationship_type": "serving",
            "adm_reference": "TOGAF 9.2 Process-Application Alignment"
        }
    }

    def __init__(self):
        """Initialize TOGAF rules engine."""
        self.violations: List[TOGAFRuleViolation] = []
        logger.info("TOGAF Rules Engine initialized")

    def validate_element_phase_alignment(
        self,
        element: ArchiMateElement,
        phase: str,
        context: Optional[Dict] = None
    ) -> List[TOGAFRuleViolation]:
        """
        Validate if element is appropriate for TOGAF phase.

        Args:
            element: ArchiMate element to validate
            phase: TOGAF ADM phase
            context: Additional context for exception evaluation

        Returns:
            List of rule violations (empty if valid)
        """
        violations = []
        context = context or {}

        # Check core phase-layer rules
        if phase in self.CORE_PHASE_LAYER_RULES:
            phase_rules = self.CORE_PHASE_LAYER_RULES[phase]
            primary_layers = phase_rules["primary_layers"]

            if element.layer not in primary_layers:
                # Check for documented exceptions
                exception = self._check_exceptions(element, phase, context)

                if exception:
                    # Exception applied - create informational violation
                    violation = TOGAFRuleViolation(
                        element_id=element.id,
                        element_type=element.type,
                        element_layer=element.layer,
                        phase=phase,
                        rule_id=f"TOGAF_PHASE_LAYER_{phase.replace(' ', '_').upper()}",
                        severity=ViolationSeverity.EXCEPTION,
                        message=f"{element.type} ({element.layer}) allowed in {phase} via exception",
                        adm_reference=exception["adm_reference"],
                        guidance_passage=exception["guidance"],
                        suggested_action="Exception applied - review architectural rationale",
                        exception_applied=exception["reason"]
                    )
                    violations.append(violation)
                else:
                    # Hard rule violation
                    violation = TOGAFRuleViolation(
                        element_id=element.id,
                        element_type=element.type,
                        element_layer=element.layer,
                        phase=phase,
                        rule_id=f"TOGAF_PHASE_LAYER_{phase.replace(' ', '_').upper()}",
                        severity=ViolationSeverity.WARNING,
                        message=f"{element.type} ({element.layer}) not typically used in {phase}",
                        adm_reference=phase_rules["adm_reference"],
                        guidance_passage=phase_rules["guidance"],
                        suggested_action=f"Consider moving to appropriate phase or document exception"
                    )
                    violations.append(violation)

        return violations

    def _check_exceptions(
        self,
        element: ArchiMateElement,
        phase: str,
        context: Dict
    ) -> Optional[Dict]:
        """
        Check if element qualifies for documented exception in phase.

        Args:
            element: ArchiMate element
            phase: TOGAF phase
            context: Context for exception evaluation

        Returns:
            Exception details if applicable, None otherwise
        """
        phase_exceptions = self.TOGAF_EXCEPTIONS.get(phase, {})

        # Check for wildcard exception (Phase E)
        if "*" in phase_exceptions:
            return phase_exceptions["*"]

        # Check for specific element type exception
        if element.type in phase_exceptions:
            exception = phase_exceptions[element.type]

            # Evaluate exception conditions
            if self._evaluate_exception_conditions(exception["conditions"], context):
                return exception

        return None

    def _evaluate_exception_conditions(
        self,
        conditions: List[str],
        context: Dict
    ) -> bool:
        """
        Evaluate if context meets exception conditions.

        Args:
            conditions: List of condition identifiers
            context: Context dictionary

        Returns:
            True if conditions are met
        """
        # For now, return True if any condition keyword is in context
        # In production, implement sophisticated condition evaluation
        for condition in conditions:
            if condition in context.get("scenario_tags", []):
                return True
            if context.get(condition.replace("_", ""), False):
                return True

        return False

    def validate_cross_layer_relationships(
        self,
        elements: List[ArchiMateElement],
        phase: str
    ) -> List[TOGAFRuleViolation]:
        """
        Validate cross-layer relationships for TOGAF compliance.

        Args:
            elements: List of related elements
            phase: Current TOGAF phase

        Returns:
            List of relationship violations
        """
        violations = []

        # Check each pair of elements for valid cross-layer relationships
        for i, element1 in enumerate(elements):
            for element2 in elements[i+1:]:
                if element1.layer != element2.layer:
                    # Cross-layer relationship detected
                    relationship_key = f"{element1.type}_{element2.type}"
                    reverse_key = f"{element2.type}_{element1.type}"

                    # Check if this cross-layer relationship is valid
                    if (relationship_key not in self.CROSS_LAYER_RULES and
                        reverse_key not in self.CROSS_LAYER_RULES):

                        violation = TOGAFRuleViolation(
                            element_id=f"{element1.id}+{element2.id}",
                            element_type=f"{element1.type}-{element2.type}",
                            element_layer=f"{element1.layer}-{element2.layer}",
                            phase=phase,
                            rule_id="TOGAF_CROSS_LAYER_RELATIONSHIP",
                            severity=ViolationSeverity.INFO,
                            message=f"Uncommon cross-layer relationship: {element1.type} ({element1.layer}) ↔ {element2.type} ({element2.layer})",
                            adm_reference="TOGAF 9.2 Cross-Layer Guidelines",
                            guidance_passage="Cross-layer relationships should be explicitly justified",
                            suggested_action="Document architectural rationale for cross-layer relationship"
                        )
                        violations.append(violation)

        return violations

    def validate_phase_completeness(
        self,
        elements: List[ArchiMateElement],
        phase: str
    ) -> List[TOGAFRuleViolation]:
        """
        Validate that phase has appropriate element coverage.

        Args:
            elements: Elements in the phase
            phase: TOGAF phase to validate

        Returns:
            List of completeness violations
        """
        violations = []

        if phase not in self.CORE_PHASE_LAYER_RULES:
            return violations

        phase_rules = self.CORE_PHASE_LAYER_RULES[phase]
        required_layers = set(phase_rules["primary_layers"])
        present_layers = set(element.layer for element in elements)

        missing_layers = required_layers - present_layers

        for missing_layer in missing_layers:
            violation = TOGAFRuleViolation(
                element_id="",
                element_type="Missing",
                element_layer=missing_layer,
                phase=phase,
                rule_id=f"TOGAF_PHASE_COMPLETENESS_{phase.replace(' ', '_').upper()}",
                severity=ViolationSeverity.WARNING,
                message=f"Phase {phase} missing {missing_layer} layer elements",
                adm_reference=phase_rules["adm_reference"],
                guidance_passage=phase_rules["guidance"],
                suggested_action=f"Consider adding {missing_layer} layer elements or document omission rationale"
            )
            violations.append(violation)

        return violations

    def generate_violation_report(self, violations: List[TOGAFRuleViolation]) -> str:
        """
        Generate human-readable violation report.

        Args:
            violations: List of violations to report

        Returns:
            Formatted violation report
        """
        if not violations:
            return "✅ No TOGAF rule violations detected"

        report = f"TOGAF ADM Validation Report - {len(violations)} issues found\n"
        report += "=" * 60 + "\n\n"

        # Group by severity
        by_severity = {}
        for violation in violations:
            severity = violation.severity.value
            if severity not in by_severity:
                by_severity[severity] = []
            by_severity[severity].append(violation)

        for severity in ["error", "warning", "info", "exception"]:
            if severity in by_severity:
                count = len(by_severity[severity])
                report += f"{severity.upper()}: {count} issues\n"

                for violation in by_severity[severity]:
                    report += f"  • {violation.message}\n"
                    report += f"    Element: {violation.element_type} ({violation.element_id})\n"
                    report += f"    Phase: {violation.phase}\n"
                    report += f"    Reference: {violation.adm_reference}\n"
                    report += f"    Action: {violation.suggested_action}\n"
                    if violation.exception_applied:
                        report += f"    Exception: {violation.exception_applied}\n"
                    report += "\n"

        return report

    def create_ci_lint_rules(self) -> Dict:
        """
        Generate CI/CD lint rules for automated validation.

        Returns:
            Dictionary of lint rules for CI integration
        """
        return {
            "togaf_phase_layer_alignment": {
                "description": "Validate phase-layer alignment with exceptions",
                "severity": "warning",
                "exceptions_file": "togaf_exceptions.yaml",
                "adm_references": True
            },
            "togaf_cross_layer_relationships": {
                "description": "Check cross-layer relationship validity",
                "severity": "info",
                "auto_fix": False
            },
            "togaf_phase_completeness": {
                "description": "Ensure phases have appropriate coverage",
                "severity": "warning",
                "threshold": 0.8
            }
        }


def create_togaf_validator() -> TOGAFRulesEngine:
    """Factory function to create TOGAF validator."""
    return TOGAFRulesEngine()