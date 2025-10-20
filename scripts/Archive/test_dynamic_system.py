#!/usr/bin/env python3
import asyncio
from src.agent.ea_assistant import ProductionEAAgent
from src.archimate.parser import ArchiMateParser
from collections import Counter

async def test_dynamic_discovery():
    """Test that system dynamically adapts to whatever elements are available."""
    
    print("=== DYNAMIC ELEMENT DISCOVERY TEST ===\n")
    
    # First, see what we actually have
    parser = ArchiMateParser()
    parser.load_model("data/models/IEC 61968.xml")
    parser.load_model("data/models/archi-4-archi.xml")
    
    type_counts = Counter(elem.type for elem in parser.elements.values())
    print(f"Found {len(type_counts)} different element types:")
    for elem_type, count in type_counts.most_common(5):
        print(f"  - {elem_type}: {count}")
    
    # Now test the agent
    agent = ProductionEAAgent()
    await asyncio.sleep(5)
    
    # Test generic queries that should work with ANY element types
    test_queries = [
        "What elements are available?",
        "Show me architecture components",
        "List available capabilities",  # Even if no Capabilities exist
        "What can you tell me about the architecture?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        
        try:
            response = await agent.process_query(query)
            
            print(f"Response preview: {response.response[:150]}...")
            print(f"Citations: {response.citations[:2] if response.citations else 'None'}")
            
            # Check if response adapted to available data
            if response.citations and response.citations[0].startswith('archi:'):
                print("✓ System found and used actual ArchiMate elements")
            elif 'external' in str(response.citations):
                print("⚠ System fell back to external LLM")
                
        except Exception as e:
            print(f"Error: {e}")

async def test_future_proof():
    """Test that system can handle queries for non-existent element types."""
    
    print("\n\n=== FUTURE-PROOF TEST ===\n")
    
    agent = ProductionEAAgent()
    await asyncio.sleep(5)
    
    # Query for element types that might not exist yet
    future_queries = [
        "Show me capabilities",  # You might not have these now
        "List all interfaces",   # Might or might not exist
        "What nodes are defined?", # Might or might not exist
        "Show me stakeholders"   # You have 6 of these
    ]
    
    for query in future_queries:
        print(f"\nQuery: {query}")
        
        context = await agent._retrieve_knowledge(query, "structured_model")
        
        if context['candidates']:
            types = set(c.get('type', 'Unknown') for c in context['candidates'][:5])
            print(f"  Found candidates of types: {types}")
        else:
            print(f"  No specific matches, fallback should activate")

if __name__ == "__main__":
    asyncio.run(test_dynamic_discovery())
    asyncio.run(test_future_proof())