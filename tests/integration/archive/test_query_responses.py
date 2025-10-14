#!/usr/bin/env python3
"""
Direct query testing to verify actual responses from the EA Assistant.
Tests real queries and shows exact outputs to verify quality.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.ea_assistant import ProductionEAAgent


async def test_specific_queries():
    """Test specific queries and display full responses for quality check."""
    
    # Set up paths relative to this test file
    base_path = Path(__file__).parent.parent
    kg_path = base_path / "data" / "energy_knowledge_graph.ttl"
    models_path = base_path / "data" / "models"
    vocab_path = base_path / "config" / "vocabularies.json"
    
    print(f"Using KG path: {kg_path}")
    print(f"KG exists: {kg_path.exists()}")
    
    if not kg_path.exists():
        print(f"ERROR: Knowledge graph not found at {kg_path}")
        print("Please ensure you're running from the project root or tests directory")
        return False
    
    # Initialize agent
    agent = ProductionEAAgent(
        kg_path=str(kg_path),
        models_path=str(models_path),
        vocab_path=str(vocab_path)
    )
    
    # Wait for KG to load
    print("Waiting for knowledge graph to load...")
    await asyncio.sleep(5)
    
    # Test queries
    test_queries = [
        "What is an asset in Alliander terms?",
        "What is asset management?",
        "What is power?",
        "What is a grid?",
        "Define reactive power",
        "What capabilities exist for grid congestion?"
    ]
    
    print("\n" + "="*80)
    print("TESTING ACTUAL QUERY RESPONSES")
    print("="*80)
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"QUERY: {query}")
        print(f"{'='*80}")
        
        try:
            response = await agent.process_query(query)
            
            print(f"\nFULL RESPONSE:")
            print("-"*40)
            print(response.response)
            print("-"*40)
            
            print(f"\nMETADATA:")
            print(f"  Route: {response.route}")
            print(f"  Citations: {response.citations}")
            print(f"  Confidence: {response.confidence}")
            print(f"  Requires Review: {response.requires_human_review}")
            print(f"  Processing Time: {response.processing_time_ms:.1f}ms")
            
            # Quality checks
            print(f"\nQUALITY CHECKS:")
            
            # Check 1: Response should contain key term from query
            key_term = query.split()[2] if "is" in query else query.split()[0]
            key_term = key_term.lower().strip('?')
            if key_term in response.response.lower():
                print(f"  ✓ Response mentions '{key_term}'")
            else:
                print(f"  ✗ Response doesn't mention '{key_term}'")
            
            # Check 2: Should have citations
            if response.citations and len(response.citations) > 0:
                print(f"  ✓ Has {len(response.citations)} citations")
                
                # Check citation format
                for cit in response.citations:
                    if cit.startswith('external:llm'):
                        print(f"    ⚠️  Using LLM fallback: {cit}")
                    elif cit.startswith('archi:id-'):
                        print(f"    ⚠️  Possibly fake ArchiMate citation: {cit}")
                    else:
                        print(f"    ✓ Citation: {cit}")
            else:
                print(f"  ✗ No citations provided")
            
            # Check 3: Response length
            if len(response.response) > 50:
                print(f"  ✓ Response length: {len(response.response)} chars")
            else:
                print(f"  ✗ Response too short: {len(response.response)} chars")
            
        except Exception as e:
            print(f"\n✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("END OF QUERY TESTS")
    print("="*80)
    
    return True


async def test_asset_vs_assetmanagement():
    """Specific test to ensure Asset and AssetManagement are differentiated."""
    
    base_path = Path(__file__).parent.parent
    kg_path = base_path / "data" / "energy_knowledge_graph.ttl"
    models_path = base_path / "data" / "models"
    vocab_path = base_path / "config" / "vocabularies.json"
    
    agent = ProductionEAAgent(
        kg_path=str(kg_path),
        models_path=str(models_path),
        vocab_path=str(vocab_path)
    )
    
    await asyncio.sleep(5)
    
    print("\n" + "="*80)
    print("ASSET vs ASSET MANAGEMENT DIFFERENTIATION TEST")
    print("="*80)
    
    # Query for Asset
    asset_response = await agent.process_query("What is an asset?")
    
    # Query for Asset Management
    mgmt_response = await agent.process_query("What is asset management?")
    
    print("\n--- ASSET Response ---")
    print(asset_response.response[:300])
    print(f"Citations: {asset_response.citations}")
    
    print("\n--- ASSET MANAGEMENT Response ---")
    print(mgmt_response.response[:300])
    print(f"Citations: {mgmt_response.citations}")
    
    print("\nVERIFICATION:")
    
    # Check 1: Asset response should NOT primarily discuss management
    asset_text = asset_response.response.lower()
    if "entity of value" in asset_text:
        print("✓ Asset response contains correct definition")
    else:
        print("✗ Asset response missing 'entity of value' definition")
    
    # Check 2: Responses should be different
    if asset_response.response != mgmt_response.response:
        print("✓ Responses are different")
    else:
        print("✗ WARNING: Same response for both queries!")
    
    # Check 3: Management response should discuss systematic management
    mgmt_text = mgmt_response.response.lower()
    if "systematic" in mgmt_text or "management" in mgmt_text:
        print("✓ Asset Management response discusses management")
    else:
        print("✗ Asset Management response doesn't discuss management")
    
    return True


if __name__ == "__main__":
    print("Starting EA Assistant Query Response Tests...")
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Run tests
        loop.run_until_complete(test_specific_queries())
        loop.run_until_complete(test_asset_vs_assetmanagement())
        
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
    finally:
        loop.close()
    
    print("\nTests complete. Review the responses above for quality issues.")