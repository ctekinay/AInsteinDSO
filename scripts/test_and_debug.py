# test_and_debug.py
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.agent.ea_assistant import ProductionEAAgent

async def debug_asset_query():
    """Full debugging of the asset query issue"""
    
    print("Initializing EA Agent...")
    agent = ProductionEAAgent()
    
    print("=== STEP 1: Check KG Loading ===")
    for i in range(20):  # Wait up to 10 seconds
        if agent.kg_loader.is_full_graph_loaded():
            print(f"✓ KG loaded after {i*0.5}s")
            break
        await asyncio.sleep(0.5)
    else:
        print("✗ KG failed to load within 10s")
        return
    
    print("\n=== STEP 2: Raw SPARQL Query ===")
    sparql = """
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    SELECT ?concept ?label WHERE {
        ?concept skos:prefLabel ?label .
        FILTER(LCASE(STR(?label)) = "asset")
    }
    """
    try:
        results = list(agent.kg_loader.graph.query(sparql))
        print(f"Exact matches for 'asset': {len(results)}")
        for r in results:
            print(f"  - {r.concept}: {r.label}")
    except Exception as e:
        print(f"SPARQL query failed: {e}")
    
    print("\n=== STEP 3: Using query_definitions method ===")
    kg_results = agent.kg_loader.query_definitions(["asset"])
    print(f"Found {len(kg_results)} results via query_definitions:")
    for idx, result in enumerate(kg_results[:5]):  # Show top 5
        print(f"\n  [{idx+1}] Label: {result['label']}")
        print(f"      Score: {result.get('score', 0)}")
        print(f"      Citation: {result.get('citation_id', 'N/A')}")
        print(f"      Definition: {result.get('definition', 'No definition')[:100]}...")
        print(f"      Full URI: {result['concept']}")
    
    print("\n=== STEP 4: Check Router ===")
    route = agent.router.route("What is an asset in Alliander terms?")
    print(f"Route: {route}")
    
    print("\n=== STEP 5: Check Retrieval ===")
    retrieval_context = await agent._retrieve_knowledge("What is an asset in Alliander terms?", route)
    print(f"Candidates found: {len(retrieval_context.get('candidates', []))}")
    for idx, cand in enumerate(retrieval_context.get('candidates', [])[:3]):
        print(f"\n  Candidate {idx+1}:")
        print(f"    Element: {cand.get('element')}")
        print(f"    Citation: {cand.get('citation')}")
        print(f"    Confidence: {cand.get('confidence')}")
        print(f"    Has Definition: {'Yes' if cand.get('definition') else 'No'}")
    
    print("\n=== STEP 6: Full Pipeline ===")
    try:
        response = await agent.process_query("What is an asset in Alliander terms?")
        
        print(f"\nResponse (first 300 chars):")
        print(f"  {response.response[:300]}...")
        print(f"\nCitations: {response.citations}")
        print(f"Confidence: {response.confidence}")
        print(f"Route taken: {response.route}")
        
        # Check if response actually answers the question
        if "asset" in response.response.lower():
            if "management" in response.response.lower() and response.response.lower().find("asset") > response.response.lower().find("management"):
                print("\n⚠️ ISSUE: Response talks about AssetManagement instead of Asset")
            else:
                print("\n✓ Response correctly addresses 'asset'")
        else:
            print("\n✗ ISSUE: Response doesn't mention 'asset' at all")
            
    except Exception as e:
        print(f"\n✗ Pipeline error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_asset_query())