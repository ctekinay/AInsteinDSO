#!/usr/bin/env python3
# test_integration_complete.py
import asyncio
import time
import os
from src.agent.ea_assistant import ProductionEAAgent

async def test_complete_integration():
    print("INTEGRATION TEST SUITE\n")
    results = {"passed": 0, "failed": 0}
    
    # Test 1: Pipeline without LLM (template mode)
    print("1. Testing template fallback...")
    agent = ProductionEAAgent(llm_provider=None)
    resp = await agent.process_query("What capability for grid congestion?", "test-1")
    
    # Use correct attribute name: 'response' not 'answer'
    assert hasattr(resp, 'response'), "Missing response"
    assert hasattr(resp, 'citations'), "Missing citations"
    assert hasattr(resp, 'confidence'), "Missing confidence"
    
    if resp.response and "Based on" in resp.response:
        print(f"   ✓ Template text: {resp.response[:50]}...")
        results["passed"] += 1
    else:
        print(f"   ✗ No template text")
        results["failed"] += 1
    
    # Test 2: Router functionality
    print("\n2. Testing router...")
    test_routes = [
        ("What capability for congestion", "structured_model"),  # Has 'capability'
        ("Phase B architecture", "togaf_method"),
        ("general documentation", "unstructured_docs")
    ]
    
    for query, expected in test_routes:
        resp = await agent.process_query(query, f"route-test")
        if resp.route == expected:
            print(f"   ✓ '{query[:20]}' → {resp.route}")
            results["passed"] += 1
        else:
            print(f"   ✗ '{query[:20]}' → {resp.route} (expected {expected})")
            results["failed"] += 1
    
    # Test 3: Performance
    print("\n3. Testing performance...")
    times = []
    for i in range(5):
        resp = await agent.process_query("test", f"perf-{i}")
        times.append(resp.processing_time_ms)
    
    avg_time = sum(times) / len(times)
    if avg_time < 10:  # Should be <10ms
        print(f"   ✓ Average: {avg_time:.1f}ms")
        results["passed"] += 1
    else:
        print(f"   ✗ Slow: {avg_time:.1f}ms")
        results["failed"] += 1
    
    # Test 4: Confidence assessment
    print("\n4. Testing confidence...")
    resp = await agent.process_query("vague query", "confidence-test")
    if resp.confidence < 0.75 and resp.requires_human_review:
        print(f"   ✓ Low confidence ({resp.confidence:.2f}) triggers review")
        results["passed"] += 1
    else:
        print(f"   ✗ Confidence handling issue")
        results["failed"] += 1
    
    # Test 5: Check response structure
    print("\n5. Testing response structure...")
    resp = await agent.process_query("test structure", "struct-test")
    required_attrs = ['response', 'citations', 'confidence', 'route', 'session_id']
    
    for attr in required_attrs:
        if hasattr(resp, attr):
            print(f"   ✓ Has {attr}")
            results["passed"] += 1
        else:
            print(f"   ✗ Missing {attr}")
            results["failed"] += 1
    
    # Summary
    print("\n" + "="*50)
    total = results['passed'] + results['failed']
    print(f"RESULTS: {results['passed']}/{total} passed")
    print(f"Success rate: {results['passed']/total*100:.1f}%")
    
    return results['failed'] == 0

if __name__ == "__main__":
    success = asyncio.run(test_complete_integration())
    exit(0 if success else 1)