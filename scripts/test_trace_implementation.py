#!/usr/bin/env python3
"""
Test script to verify trace implementation works correctly.
Run this BEFORE starting the web interface to catch any issues.
"""

import asyncio
import sys
from pathlib import Path

# CRITICAL: Load .env BEFORE importing anything else
from dotenv import load_dotenv
load_dotenv(override=True)  # Force reload of environment variables

# Add project to path
sys.path.append(str(Path(__file__).parent.parent))

from src.agent.ea_assistant import ProductionEAAgent

async def test_trace_implementation():
    """Test that process_query returns tuple and trace is structured correctly."""
    
    print("=" * 70)
    print("🧪 Testing Trace Implementation")
    print("=" * 70)
    
    # Initialize agent
    print("\n1️⃣  Initializing ProductionEAAgent...")
    try:
        agent = ProductionEAAgent()
        print("   ✅ Agent initialized successfully")
    except Exception as e:
        print(f"   ❌ Failed to initialize agent: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Wait for background loading
    print("\n2️⃣  Waiting for Knowledge Graph loading...")
    await asyncio.sleep(5)
    print("   ✅ Initialization complete")
    
    # Test query
    query = "What is reactive power?"
    print(f"\n3️⃣  Processing test query: '{query}'")
    
    try:
        # CRITICAL: This should return a tuple
        result = await agent.process_query(query)
        
        # Check if it's a tuple
        if not isinstance(result, tuple):
            print(f"   ❌ FAILED: Expected tuple, got {type(result)}")
            print(f"      Your process_query() is not returning a tuple!")
            return False
        
        if len(result) != 2:
            print(f"   ❌ FAILED: Expected tuple of length 2, got {len(result)}")
            return False
        
        response, trace = result
        print("   ✅ Returned tuple correctly")
        
    except Exception as e:
        print(f"   ❌ FAILED: Exception during query: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Verify response structure
    print("\n4️⃣  Verifying response structure...")
    
    required_response_fields = [
        'query', 'response', 'route', 'citations', 'confidence',
        'requires_human_review', 'togaf_phase', 'archimate_elements',
        'processing_time_ms', 'session_id', 'timestamp'
    ]
    
    for field in required_response_fields:
        if not hasattr(response, field):
            print(f"   ❌ Missing field: {field}")
            return False
    
    print("   ✅ Response structure valid")
    print(f"      - Query: {response.query}")
    print(f"      - Route: {response.route}")
    print(f"      - Citations: {len(response.citations)}")
    print(f"      - Confidence: {response.confidence:.2f}")
    print(f"      - Processing time: {response.processing_time_ms:.0f}ms")
    
    # Verify trace structure
    print("\n5️⃣  Verifying trace structure...")
    
    if not hasattr(trace, 'to_dict'):
        print("   ❌ Trace missing to_dict() method")
        return False
    
    trace_dict = trace.to_dict()
    
    required_trace_fields = ['session_id', 'query', 'total_duration_ms', 'phases']
    for field in required_trace_fields:
        if field not in trace_dict:
            print(f"   ❌ Missing trace field: {field}")
            return False
    
    print("   ✅ Trace structure valid")
    print(f"      - Session ID: {trace_dict['session_id']}")
    print(f"      - Query: {trace_dict['query']}")
    print(f"      - Total duration: {trace_dict['total_duration_ms']:.0f}ms")
    print(f"      - Phases: {len(trace_dict['phases'])}")
    
    # Verify phases
    print("\n6️⃣  Verifying phases...")
    
    expected_phases = [
        'INITIALIZATION', 'REFLECT', 'ROUTE', 'RETRIEVE',
        'BUILD_CITATION_POOL', 'REFINE', 'GROUND', 'CRITIC'
    ]
    
    # Note: VALIDATE_TOGAF and RESPONSE_ASSEMBLY might not always be present
    
    phases = trace_dict['phases']
    phase_names = [p['name'] for p in phases]
    
    print(f"   Found phases: {phase_names}")
    
    for expected_phase in expected_phases:
        if expected_phase not in phase_names:
            print(f"   ⚠️  Warning: Phase '{expected_phase}' not found")
    
    if len(phases) < 8:
        print(f"   ⚠️  Warning: Expected at least 8 phases, found {len(phases)}")
    else:
        print(f"   ✅ Found {len(phases)} phases")
    
    # Verify phase structure
    print("\n7️⃣  Verifying phase details...")
    
    for phase in phases[:3]:  # Check first 3 phases
        required_phase_fields = ['name', 'duration_ms', 'status', 'details', 'sub_steps']
        for field in required_phase_fields:
            if field not in phase:
                print(f"   ❌ Phase '{phase.get('name', 'unknown')}' missing field: {field}")
                return False
    
    print("   ✅ Phase details valid")
    
    # Print sample phases
    print("\n8️⃣  Sample phases:")
    for i, phase in enumerate(phases[:5], 1):
        print(f"   {i}. {phase['name']}: {phase['duration_ms']:.0f}ms [{phase['status']}]")
        if phase['details']:
            for key, value in list(phase['details'].items())[:2]:
                print(f"      - {key}: {value}")
    
    # Success!
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)
    print("\nYour trace implementation is working correctly.")
    print("You can now start the web interface with:")
    print("  python run_web_demo.py")
    print("\n")
    
    return True


if __name__ == "__main__":
    success = asyncio.run(test_trace_implementation())
    sys.exit(0 if success else 1)