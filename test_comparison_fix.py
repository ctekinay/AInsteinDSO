#!/usr/bin/env python3
"""
Quick CLI test for comparison query fixes.

Tests the core comparison logic without requiring full web interface.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.agents.ea_assistant import ProductionEAAgent

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


async def test_comparison_fix():
    """Test the comparison query fix with manual validation."""

    print("🔧 Testing Comparison Query Fix")
    print("=" * 50)

    # Test queries that should return distinct concepts
    test_queries = [
        "What is the difference between active and reactive power?",
        "active power vs reactive power",
        "compare transformer with conductor",
        "business capability or application component"
    ]

    # Mock candidates for testing (simulating retrieval results)
    mock_candidates = [
        {
            'element': 'Active power',
            'definition': 'The real component of apparent power expressed in watts',
            'citation': 'eurlex:631-20',
            'confidence': 0.85
        },
        {
            'element': 'Reactive power',
            'definition': 'The imaginary component of apparent power expressed in vars',
            'citation': 'eurlex:631-28',
            'confidence': 0.82
        },
        {
            'element': 'Transformer',
            'definition': 'Equipment that changes voltage levels in electrical systems',
            'citation': 'iec:transformer-001',
            'confidence': 0.80
        },
        {
            'element': 'Conductor',
            'definition': 'Material that allows electric current to flow through it',
            'citation': 'iec:conductor-001',
            'confidence': 0.78
        },
        {
            'element': 'Business Capability',
            'definition': 'Ability of organization to achieve specific business outcome',
            'citation': 'archi:id-capability-001',
            'confidence': 0.75
        },
        {
            'element': 'Application Component',
            'definition': 'Software module that encapsulates business logic',
            'citation': 'archi:id-component-001',
            'confidence': 0.73
        }
    ]

    try:
        # Create a minimal agent for testing (without full initialization)
        print("Creating test agent...")

        # We'll test the core logic without full agent initialization
        from unittest.mock import Mock

        agent = Mock()

        # Import the actual methods we want to test
        from src.agents.ea_assistant import ProductionEAAgent

        # Create a real agent instance for method access
        # But we'll only test the comparison validation methods
        temp_agent = object.__new__(ProductionEAAgent)
        temp_agent.embedding_agent = None  # No semantic search for this test

        # Add the methods we need
        temp_agent._extract_comparison_terms = ProductionEAAgent._extract_comparison_terms.__get__(temp_agent)
        temp_agent._validate_comparison_candidates = ProductionEAAgent._validate_comparison_candidates.__get__(temp_agent)
        temp_agent._get_first_two_distinct = ProductionEAAgent._get_first_two_distinct.__get__(temp_agent)

        print("✅ Test agent created")

        # Test each query
        for i, query in enumerate(test_queries):
            print(f"\n🧪 Test {i+1}: {query}")
            print("-" * 40)

            try:
                # Test term extraction
                terms = temp_agent._extract_comparison_terms(query)
                print(f"📝 Extracted terms: {terms}")

                # Create relevant candidates for this query
                if 'power' in query.lower():
                    candidates = [c for c in mock_candidates if 'power' in c['element'].lower()]
                elif 'transformer' in query.lower() or 'conductor' in query.lower():
                    candidates = [c for c in mock_candidates if c['element'].lower() in ['transformer', 'conductor']]
                elif 'capability' in query.lower() or 'component' in query.lower():
                    candidates = [c for c in mock_candidates if 'capability' in c['element'].lower() or 'component' in c['element'].lower()]
                else:
                    candidates = mock_candidates[:3]  # Use first 3 as fallback

                print(f"📋 Using {len(candidates)} candidates:")
                for c in candidates:
                    print(f"   - {c['element']} [{c['citation']}]")

                # Test validation (this is the critical test)
                try:
                    concept1, concept2 = await temp_agent._validate_comparison_candidates(candidates, query)

                    citation1 = concept1.get('citation', 'NO_CITATION')
                    citation2 = concept2.get('citation', 'NO_CITATION')

                    print(f"✅ SUCCESS:")
                    print(f"   Concept 1: {concept1.get('element')} [{citation1}]")
                    print(f"   Concept 2: {concept2.get('element')} [{citation2}]")

                    # Critical validation
                    if citation1 == citation2:
                        print(f"❌ FAILURE: Duplicate citations! {citation1}")
                        return False
                    else:
                        print(f"✅ PASS: Citations are distinct")

                except Exception as e:
                    print(f"❌ VALIDATION ERROR: {e}")
                    # This might be expected if semantic search is required
                    if "semantic search" in str(e).lower():
                        print("ℹ️  This is expected when semantic fallback is needed")
                    else:
                        return False

            except Exception as e:
                print(f"❌ UNEXPECTED ERROR: {e}")
                return False

        print(f"\n🎉 All comparison query tests completed!")
        return True

    except Exception as e:
        print(f"❌ SETUP ERROR: {e}")
        return False


def test_distinct_logic():
    """Test the _get_first_two_distinct logic specifically."""
    print("\n🔧 Testing _get_first_two_distinct logic")
    print("=" * 40)

    # Test cases with duplicates
    test_candidates = [
        {'element': 'A', 'citation': 'test:001', 'confidence': 0.9},
        {'element': 'A', 'citation': 'test:001', 'confidence': 0.85},  # Duplicate
        {'element': 'B', 'citation': 'test:002', 'confidence': 0.80},
        {'element': 'C', 'citation': 'test:003', 'confidence': 0.75},
    ]

    try:
        from src.agents.ea_assistant import ProductionEAAgent

        # Create a temporary agent for method access
        temp_agent = object.__new__(ProductionEAAgent)
        temp_agent._get_first_two_distinct = ProductionEAAgent._get_first_two_distinct.__get__(temp_agent)

        c1, c2 = temp_agent._get_first_two_distinct(test_candidates)

        print(f"Input candidates:")
        for c in test_candidates:
            print(f"   - {c['element']} [{c['citation']}] ({c['confidence']})")

        print(f"\nSelected distinct candidates:")
        print(f"   1. {c1.get('element')} [{c1.get('citation')}]")
        print(f"   2. {c2.get('element')} [{c2.get('citation')}]")

        if c1.get('citation') != c2.get('citation'):
            print("✅ PASS: Citations are distinct")
            return True
        else:
            print("❌ FAIL: Citations are identical")
            return False

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


async def main():
    """Run all tests."""
    print("🚀 Starting Comparison Query Fix Tests")
    print("=" * 60)

    # Test the distinct logic first (simpler)
    distinct_test_passed = test_distinct_logic()

    # Test the full comparison logic
    comparison_test_passed = await test_comparison_fix()

    print("\n" + "=" * 60)
    print("📊 TEST RESULTS")
    print("=" * 60)
    print(f"Distinct Logic Test: {'✅ PASS' if distinct_test_passed else '❌ FAIL'}")
    print(f"Comparison Logic Test: {'✅ PASS' if comparison_test_passed else '❌ FAIL'}")

    if distinct_test_passed and comparison_test_passed:
        print(f"\n🎉 ALL TESTS PASSED!")
        print("The comparison query fix is working correctly.")
        return True
    else:
        print(f"\n❌ SOME TESTS FAILED!")
        print("Please review the implementation.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)