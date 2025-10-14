#!/usr/bin/env python3
import asyncio
from src.agent.ea_assistant import ProductionEAAgent
from src.knowledge.kg_loader import KnowledgeGraphLoader
from pathlib import Path

async def test_asset_query():
    # First test KG directly
    print("=== Testing KG directly ===")
    kg = KnowledgeGraphLoader(Path("data/energy_knowledge_graph.ttl"))
    kg.load()
    
    import time
    for i in range(10):
        if kg.is_full_graph_loaded():
            break
        time.sleep(0.5)
    
    results = kg.query_definitions(["asset"])
    print(f"KG found {len(results)} results for 'asset'")
    if results:
        print(f"First result: {results[0]['label']}: {results[0].get('definition', 'NO DEFINITION')}")
    
    # Now test full pipeline
    print("\n=== Testing full pipeline ===")
    agent = ProductionEAAgent()
    await asyncio.sleep(5)
    
    response = await agent.process_query("What is an asset in Alliander terms?")
    # At the end, change the check to:
    if "Entity of value to individuals or organizations" in response.response:
        print("✅ SUCCESS! Correct definition returned!")
    else:
        print(f"❌ Wrong response: {response.response[:200]}")


asyncio.run(test_asset_query())