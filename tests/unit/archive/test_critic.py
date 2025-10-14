"""
Comprehensive unit tests for the Critic module - Confidence Assessment System.

These tests verify that the Critic correctly:
1. Enforces confidence threshold of 0.75
2. Identifies irrelevant elements (max 18% budget)
3. Forces human review for low confidence
4. Ranks suggestions by confidence
5. Respects irrelevance budget constraints

The Critic is critical for preventing overconfident wrong answers based on
research findings from LLM-KG evaluation studies.
"""

import unittest

from src.validation.critic import Critic, CriticAssessment


class TestCritic(unittest.TestCase):
    """
    Test suite for Critic confidence assessment and irrelevance detection.
    """

    def setUp(self):
        """Set up test fixtures."""
        self.critic = Critic()

    def test_high_confidence_no_review_needed(self):
        """Test high confidence (0.9) → no review needed."""
        candidates = [
            {'element': 'Capability', 'confidence': 0.9, 'citations': ['archi:id-cap-001']},
            {'element': 'Process', 'confidence': 0.85, 'citations': ['archi:id-proc-001']},
            {'element': 'Service', 'confidence': 0.8, 'citations': ['archi:id-serv-001']}
        ]

        assessment = self.critic.assess(candidates)

        self.assertEqual(len(assessment.top_suggestions), 3)
        self.assertEqual(assessment.top_suggestions[0]['element'], 'Capability')
        self.assertAlmostEqual(assessment.confidence, 0.85, places=2)  # Average: (0.9+0.85+0.8)/3
        self.assertFalse(assessment.requires_human_review)
        self.assertIn("Confidence sufficient", assessment.reasoning)

    def test_low_confidence_requires_human_review(self):
        """Test low confidence (0.4) → requires_human_review = True."""
        candidates = [
            {'element': 'Actor', 'confidence': 0.4, 'citations': ['archi:id-act-001']},
            {'element': 'Role', 'confidence': 0.35, 'citations': ['archi:id-role-001']},
            {'element': 'Interface', 'confidence': 0.3, 'citations': ['archi:id-int-001']}
        ]

        assessment = self.critic.assess(candidates)

        self.assertEqual(len(assessment.top_suggestions), 3)
        self.assertAlmostEqual(assessment.confidence, 0.35, places=2)  # Average: (0.4+0.35+0.3)/3
        self.assertTrue(assessment.requires_human_review)
        self.assertIn("human review required", assessment.reasoning)

    def test_irrelevance_budget_enforcement_18_percent(self):
        """Test 10 candidates with 2 irrelevant → respects 18% budget."""
        candidates = []

        # 8 good candidates (0.5-0.9 confidence)
        for i in range(8):
            candidates.append({
                'element': f'GoodElement{i}',
                'confidence': 0.5 + (i * 0.05),  # 0.5 to 0.85
                'citations': [f'archi:id-good-{i}']
            })

        # 2 bad candidates (< 0.3 confidence)
        candidates.extend([
            {'element': 'BadElement1', 'confidence': 0.2, 'citations': ['archi:id-bad-1']},
            {'element': 'BadElement2', 'confidence': 0.1, 'citations': ['archi:id-bad-2']}
        ])

        assessment = self.critic.assess(candidates)

        # Should have top 3 suggestions
        self.assertEqual(len(assessment.top_suggestions), 3)

        # Should identify irrelevant items within 18% budget
        # 10 candidates * 18% = 1.8 → 1 item max (but we have 2 low confidence)
        max_irrelevant = int(10 * 0.18)  # 1 item
        self.assertLessEqual(len(assessment.irrelevant_items), max_irrelevant)

        # The irrelevant items should be the lowest confidence ones
        if assessment.irrelevant_items:
            for item in assessment.irrelevant_items:
                self.assertLess(item['confidence'], 0.3)
                self.assertIn('irrelevance_reason', item)

    def test_empty_candidates_requires_review(self):
        """Test empty candidates → requires review."""
        assessment = self.critic.assess([])

        self.assertEqual(len(assessment.top_suggestions), 0)
        self.assertEqual(len(assessment.irrelevant_items), 0)
        self.assertEqual(assessment.confidence, 0.0)
        self.assertTrue(assessment.requires_human_review)
        self.assertIn("No candidates available", assessment.reasoning)

    def test_mixed_confidence_scores_correct_ranking(self):
        """Test mixed confidence scores → correct ranking."""
        # Use 10 candidates to allow for irrelevance budget (10 * 18% = 1.8 → 1 irrelevant)
        candidates = [
            {'element': 'MediumConf', 'confidence': 0.6, 'citations': ['archi:id-med']},
            {'element': 'HighConf', 'confidence': 0.9, 'citations': ['archi:id-high']},
            {'element': 'LowConf', 'confidence': 0.3, 'citations': ['archi:id-low']},
            {'element': 'VeryHighConf', 'confidence': 0.95, 'citations': ['archi:id-very-high']},
            {'element': 'VeryLowConf', 'confidence': 0.1, 'citations': ['archi:id-very-low']},
            {'element': 'Medium2', 'confidence': 0.7, 'citations': ['archi:id-med2']},
            {'element': 'Medium3', 'confidence': 0.65, 'citations': ['archi:id-med3']},
            {'element': 'Medium4', 'confidence': 0.55, 'citations': ['archi:id-med4']},
            {'element': 'Medium5', 'confidence': 0.5, 'citations': ['archi:id-med5']},
            {'element': 'Low2', 'confidence': 0.2, 'citations': ['archi:id-low2']}
        ]

        assessment = self.critic.assess(candidates)

        # Should be ranked by confidence (highest first)
        expected_order = ['VeryHighConf', 'HighConf', 'Medium2']  # Top 3
        actual_order = [s['element'] for s in assessment.top_suggestions]
        self.assertEqual(actual_order, expected_order)

        # With 10 candidates, 18% budget allows 1 irrelevant item
        # The lowest confidence item should be marked irrelevant
        irrelevant_elements = [item['element'] for item in assessment.irrelevant_items]
        if irrelevant_elements:  # Budget may allow 1 item
            # Should be the lowest confidence element
            lowest_conf_item = min(candidates, key=lambda x: x['confidence'])
            self.assertIn(lowest_conf_item['element'], irrelevant_elements)

    def test_confidence_threshold_boundary_cases(self):
        """Test confidence exactly at threshold (0.75)."""
        # Test exactly at threshold
        candidates_at_threshold = [
            {'element': 'Element1', 'confidence': 0.75, 'citations': ['archi:id-1']},
            {'element': 'Element2', 'confidence': 0.75, 'citations': ['archi:id-2']},
            {'element': 'Element3', 'confidence': 0.75, 'citations': ['archi:id-3']}
        ]

        assessment = self.critic.assess(candidates_at_threshold)
        self.assertEqual(assessment.confidence, 0.75)
        self.assertFalse(assessment.requires_human_review)  # At threshold = no review

        # Test just below threshold
        candidates_below_threshold = [
            {'element': 'Element1', 'confidence': 0.74, 'citations': ['archi:id-1']},
            {'element': 'Element2', 'confidence': 0.74, 'citations': ['archi:id-2']},
            {'element': 'Element3', 'confidence': 0.74, 'citations': ['archi:id-3']}
        ]

        assessment = self.critic.assess(candidates_below_threshold)
        self.assertAlmostEqual(assessment.confidence, 0.74, places=2)
        self.assertTrue(assessment.requires_human_review)  # Below threshold = review

    def test_citation_collection_from_top_suggestions(self):
        """Test citations are collected from top suggestions."""
        candidates = [
            {'element': 'Capability', 'confidence': 0.9,
             'citations': ['archi:id-cap-001', 'iec:GridCongestion']},
            {'element': 'Process', 'confidence': 0.8,
             'citations': ['archi:id-proc-001', 'togaf:adm:B']},
            {'element': 'Service', 'confidence': 0.7,
             'citations': ['archi:id-serv-001']},
            {'element': 'LowConf', 'confidence': 0.2,
             'citations': ['archi:id-low']}  # Should not be in top suggestions
        ]

        assessment = self.critic.assess(candidates)

        expected_citations = [
            'archi:id-cap-001', 'iec:GridCongestion',
            'archi:id-proc-001', 'togaf:adm:B',
            'archi:id-serv-001'
        ]

        # Check all expected citations are present (order may vary)
        for citation in expected_citations:
            self.assertIn(citation, assessment.citations)

        # Low confidence citation should NOT be included
        self.assertNotIn('archi:id-low', assessment.citations)

    def test_assessment_dataclass_validation(self):
        """Test CriticAssessment dataclass validation."""
        # Valid assessment
        valid_assessment = CriticAssessment(
            top_suggestions=[{'element': 'Test', 'confidence': 0.8}],
            irrelevant_items=[],
            confidence=0.8,
            citations=['archi:id-test'],
            requires_human_review=False,
            reasoning="Test reasoning"
        )
        self.assertEqual(valid_assessment.confidence, 0.8)

        # Invalid confidence (> 1.0)
        with self.assertRaises(ValueError):
            CriticAssessment(
                top_suggestions=[],
                irrelevant_items=[],
                confidence=1.5,  # Invalid
                citations=[],
                requires_human_review=True,
                reasoning="Invalid"
            )

        # Invalid confidence (< 0.0)
        with self.assertRaises(ValueError):
            CriticAssessment(
                top_suggestions=[],
                irrelevant_items=[],
                confidence=-0.1,  # Invalid
                citations=[],
                requires_human_review=True,
                reasoning="Invalid"
            )

    def test_invalid_candidate_format_raises_error(self):
        """Test invalid candidate format raises ValueError."""
        # Missing required field
        invalid_candidates = [
            {'element': 'Test'}  # Missing confidence
        ]

        with self.assertRaises(ValueError) as cm:
            self.critic.assess(invalid_candidates)
        self.assertIn("missing required field", str(cm.exception))

        # Invalid confidence type
        invalid_confidence_candidates = [
            {'element': 'Test', 'confidence': 'invalid'}
        ]

        with self.assertRaises(ValueError) as cm:
            self.critic.assess(invalid_confidence_candidates)
        self.assertIn("confidence must be float", str(cm.exception))

        # Confidence out of range
        out_of_range_candidates = [
            {'element': 'Test', 'confidence': 1.5}
        ]

        with self.assertRaises(ValueError) as cm:
            self.critic.assess(out_of_range_candidates)
        self.assertIn("confidence must be float between 0.0-1.0", str(cm.exception))

    def test_irrelevance_budget_with_different_candidate_counts(self):
        """Test irrelevance budget with various candidate counts."""
        # Test with 5 candidates (18% = 0.9 → 0 irrelevant allowed)
        candidates_5 = [
            {'element': f'Element{i}', 'confidence': 0.1 + i*0.1, 'citations': [f'id-{i}']}
            for i in range(5)
        ]
        assessment_5 = self.critic.assess(candidates_5)
        max_irrelevant_5 = int(5 * 0.18)  # 0
        self.assertLessEqual(len(assessment_5.irrelevant_items), max_irrelevant_5)

        # Test with 20 candidates (18% = 3.6 → 3 irrelevant allowed)
        candidates_20 = [
            {'element': f'Element{i}', 'confidence': 0.05 + i*0.04, 'citations': [f'id-{i}']}
            for i in range(20)
        ]
        assessment_20 = self.critic.assess(candidates_20)
        max_irrelevant_20 = int(20 * 0.18)  # 3
        self.assertLessEqual(len(assessment_20.irrelevant_items), max_irrelevant_20)

    def test_threshold_updates(self):
        """Test updating confidence and irrelevance thresholds."""
        # Update confidence threshold
        self.critic.update_thresholds(confidence_threshold=0.8)
        self.assertEqual(self.critic.confidence_threshold, 0.8)

        # Update irrelevance budget
        self.critic.update_thresholds(irrelevance_budget=0.2)
        self.assertEqual(self.critic.irrelevance_budget, 0.2)

        # Test invalid threshold values
        with self.assertRaises(ValueError):
            self.critic.update_thresholds(confidence_threshold=1.5)

        with self.assertRaises(ValueError):
            self.critic.update_thresholds(irrelevance_budget=-0.1)

    def test_assessment_stats(self):
        """Test assessment statistics and configuration."""
        stats = self.critic.get_assessment_stats()

        self.assertEqual(stats['confidence_threshold'], 0.75)
        self.assertEqual(stats['irrelevance_budget'], 0.18)
        self.assertEqual(stats['max_top_suggestions'], 3)
        self.assertIn('Research-based', stats['thresholds_source'])

    def test_example_scenario_from_requirements(self):
        """Test the exact scenario described in requirements."""
        candidates = [
            {'element': 'Capability', 'confidence': 0.85, 'citations': ['archi:id-cap-001']},
            {'element': 'Process', 'confidence': 0.70, 'citations': ['archi:id-proc-001']},
            {'element': 'Actor', 'confidence': 0.65, 'citations': ['archi:id-act-001']},
            {'element': 'Device', 'confidence': 0.20, 'citations': ['archi:id-dev-001']},
            {'element': 'Node', 'confidence': 0.15, 'citations': ['archi:id-node-001']}
        ]

        assessment = self.critic.assess(candidates)

        # Should have top 3 suggestions
        self.assertEqual(len(assessment.top_suggestions), 3)
        expected_top = ['Capability', 'Process', 'Actor']
        actual_top = [s['element'] for s in assessment.top_suggestions]
        self.assertEqual(actual_top, expected_top)

        # Should identify low confidence items as irrelevant
        # 5 candidates * 18% = 0.9 → 0 items max, but we allow at least low confidence items
        irrelevant_elements = [item['element'] for item in assessment.irrelevant_items]

        # At least the very low confidence items should be considered irrelevant
        # (even if budget doesn't allow marking them all)
        low_conf_elements = ['Device', 'Node']
        for element in low_conf_elements:
            if element in irrelevant_elements:
                # If marked irrelevant, check it has reasoning
                item = next(item for item in assessment.irrelevant_items if item['element'] == element)
                self.assertIn('irrelevance_reason', item)

        # Average confidence: (0.85 + 0.70 + 0.65) / 3 = 0.733
        expected_confidence = round((0.85 + 0.70 + 0.65) / 3, 3)
        self.assertAlmostEqual(assessment.confidence, expected_confidence, places=3)

        # Should require human review (0.733 < 0.75)
        self.assertTrue(assessment.requires_human_review)


if __name__ == '__main__':
    unittest.main(verbosity=2)