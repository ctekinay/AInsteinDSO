#!/usr/bin/env python3
from src.safety.grounding import GroundingCheck, UngroundedReplyError

def test_grounding():
    grounder = GroundingCheck()
    
    print("🧪 TESTING GROUNDING ENFORCEMENT\n")
    
    # Test 1: This MUST fail
    print("Test 1: Uncited text (should fail)")
    try:
        result = grounder.assert_citations("You should use a Business Capability for grid congestion", {})
        print("  ❌ ERROR: Uncited text was accepted! GROUNDING NOT WORKING!")
    except UngroundedReplyError as e:
        print(f"  ✅ Correctly blocked: {str(e)[:50]}...")
    
    # Test 2: This MUST pass
    print("\nTest 2: Properly cited text (should pass)")
    try:
        result = grounder.assert_citations(
            "For grid congestion (iec:61968-GridCongestion), implement Business Capability (archi:id-cap-001) per TOGAF Phase B (togaf:adm:B)",
            {}
        )
        print(f"  ✅ Citations found: {result.get('citations', [])}")
    except UngroundedReplyError:
        print("  ❌ ERROR: Valid citations rejected!")
    
    # Test 3: Edge case - citation-like text that isn't valid
    print("\nTest 3: Fake citations (should fail)")
    try:
        result = grounder.assert_citations("Use iec-like naming or archi-style patterns", {})
        print("  ❌ ERROR: False citations accepted! GROUNDING NOT STRICT ENOUGH!")
    except UngroundedReplyError:
        print("  ✅ Correctly rejected fake citations")
    
    # Test 4: Test with context suggestions
    print("\nTest 4: Citation suggestions from context")
    context = {
        "iec_terms": ["GridCongestion", "ActivePower"],
        "archimate_elements": [{"id": "cap-001", "name": "Grid Management"}]
    }
    try:
        result = grounder.assert_citations("Use capability for congestion", context)
        if result.get('status') == 'needs_citations':
            print(f"  ✅ Status: {result.get('status')}, Suggested: {result.get('suggestions', [])}")

    except UngroundedReplyError as e:
        print(f"  ℹ️  No citations, error raised as expected")
    
    print("\n" + "="*50)
    print("GROUNDING ENFORCEMENT TEST COMPLETE")

if __name__ == "__main__":
    test_grounding()