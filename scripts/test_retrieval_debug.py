#!/usr/bin/env python3
import asyncio
from src.agent.ea_assistant import ProductionEAAgent

async def test_retrieval():
    agent = ProductionEAAgent()
    await asyncio.sleep(5)  # Wait for loading
    
    # Test retrieval directly
    print("=== TESTING RETRIEVAL ===\n")
    
    query = "Show me capabilities for power management"
    context = await agent._retrieve_knowledge(query, "structured_model")
    
    print(f"Candidates found: {len(context['candidates'])}")
    
    if context['candidates']:
        print("\nFirst 3 candidates:")
        for i, cand in enumerate(context['candidates'][:3], 1):
            print(f"{i}. {cand['element']}")
            print(f"   Citation: {cand['citation']}")
            print(f"   Type: {cand.get('type')}")
            print(f"   Priority: {cand.get('priority')}")
    else:
        print("NO CANDIDATES - This is the problem!")
        
        # Debug what's in the parser
        print(f"\nArchiMate elements loaded: {len(agent.archimate_parser.elements)}")
        caps = agent.archimate_parser.get_elements_by_type("Capability")
        print(f"Capabilities available: {len(caps)}")
        
        if caps:
            print("First capability:")
            print(f"  Name: {caps[0].name}")
            print(f"  ID: {caps[0].id}")

if __name__ == "__main__":
    asyncio.run(test_retrieval())