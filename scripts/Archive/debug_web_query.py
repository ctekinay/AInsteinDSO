#!/usr/bin/env python3
import asyncio
from src.agent.ea_assistant import ProductionEAAgent

async def debug_web_query():
    """Debug the exact same query as web UI."""
    agent = ProductionEAAgent()
    await asyncio.sleep(5)  # Wait for loading
    
    query = "What is an asset in Alliander terms?"
    print(f"Query: {query}")
    
    # Check what route it takes
    route = agent.router.route(query)
    print(f"Route: {route}")
    
    # Get retrieval context
    context = await agent._retrieve_knowledge(query, route)
    print(f"\nCandidates found: {len(context['candidates'])}")
    
    # Check first few candidates
    for i, cand in enumerate(context['candidates'][:3]):
        print(f"\n{i+1}. {cand['element']} (priority: {cand.get('priority', 'unknown')})")
        print(f"   Has definition: {'Yes' if cand.get('definition') else 'No'}")
        print(f"   Citation: {cand.get('citation', 'none')}")
    
    # Now test full pipeline
    print("\n=== FULL PIPELINE ===")
    response = await agent.process_query(query)
    print(f"Response preview: {response.response[:200]}...")
    
    # Check if it's using the fix
    if "Entity of value to individuals or organizations" in response.response:
        print("✅ Fix is working in pipeline")
    else:
        print("❌ Fix NOT applied - still using LLM path")

asyncio.run(debug_web_query())